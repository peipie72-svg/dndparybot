"""
Telegram-бот «Dungeon Master» для настольной ролевой игры Dungeons & Dragons 5e (2024 / «5.5e»).

СТАТУС: режим ПАРТИИ (мультиплеер). Бот ведёт ОДНО приключение для группы из 2–6 игроков
в ОДНОМ групповом чате Telegram. Мир — только классический D&D 2024 («Забытые Королевства»).

Стек: aiogram 3.x, openai SDK (DeepSeek), python-dotenv, sqlite3.
Хранение: локальная база party_database.db (см. DATA_DIR).

Команды:
    /start       — краткая справка и приглашение в отряд (/join); в ЛС с аргументом
                   «join_<chat_id>» запускает создание героя для указанной группы
    /join        — в группе присылает кнопку для создания героя в личных сообщениях
    /party       — статус отряда: кто в группе, HP и уровень каждого героя
    /sheet       — лист СВОЕГО персонажа
    /roll        — бросок кубиков (кодом), например /roll d20 или /roll 2d6+3
    /inventory   — снаряжение и золото СВОЕГО персонажа
    /check       — меню проверок характеристик
    /spells      — книга заклинаний своего персонажа
    /rest        — отдых: восстановление HP и ячеек
    /reset_party — СБРОС всей партии в этом чате (только администраторы)

Бот отвечает в группе ТОЛЬКО когда его позвали: упоминание @бота, ответ на сообщение бота
или команда. Служебный JSON-блок Мастера относится к тому герою, чей игрок только что действовал.

Создание персонажа («Вариант Б») проходит в личных сообщениях: из группы /join присылает
кнопку-ссылку, игрок создаёт героя в ЛС и подтверждает его — готовый герой появляется в группе.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

from aiogram import BaseMiddleware, Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from dnd2024_reference import (
    ABILITIES,
    ABILITY_FULL_RU,
    ABILITY_GENITIVE_RU,
    ARMOR,
    CLASSES,
    MAX_LEVEL,
    SHIELD_BONUS,
    SKILLS,
    SPELL_LEVEL_RU,
    SPECIES,
    WEAPONS,
    ability_modifier,
    ability_priority_for_class,
    average_hit_points,
    build_reference_digest,
    class_cantrip_list,
    class_name_from_text,
    class_spell_list,
    default_cantrips_for_class,
    default_known_spells_for_class,
    format_modifier,
    hit_die_sides,
    is_pact_caster,
    is_spellcaster_class,
    is_spontaneous_caster,
    max_prepared_spells,
    next_xp_threshold,
    normalize_ability_key,
    proficiency_bonus,
    resolve_class_key,
    species_name,
    species_traits,
    spell_level,
    spell_slots_for_level,
    spellcasting_ability_for_class,
    standard_array_for_class,
    starter_equipment_for_class,
    starting_gold_for_class,
)
from prompts import (
    BASE_DM_PROMPT,
    CONTROL_BLOCK_INSTRUCTIONS,
    DM_ROLL_INSTRUCTION,
    HERO_ALREADY_CONFIRMED_ALERT,
    HERO_ALREADY_CONFIRMED_TEXT,
    HERO_CARD_QUESTION,
    HERO_CARD_TITLE_CREATED,
    HERO_CARD_TITLE_PENDING,
    HERO_CARD_TITLE_RANDOM,
    HERO_CREATION_PROMPT,
    HERO_CREATION_START_PROMPT,
    HERO_NOT_CREATED_ALERT,
)

# ---------------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# Локальная база данных SQLite с партиями, листами персонажей и историей.
DATA_DIR_ENV = os.getenv("DATA_DIR", "").strip()
if DATA_DIR_ENV:
    DATA_DIR_PATH = Path(DATA_DIR_ENV)
    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    DB_PATH = DATA_DIR_PATH / "party_database.db"
else:
    DB_PATH = BASE_DIR / "party_database.db"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

LLM_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or ""
).strip()

LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"

LLM_REQUEST_TIMEOUT = 60.0
LLM_MAX_ATTEMPTS = 4
LLM_RETRY_BASE_DELAY = 2.0

MAX_HISTORY_MESSAGES = 20
COMBAT_HISTORY_SAFETY_LIMIT = 90
BATTLE_SUMMARY_PREFIX = "[ИТОГ БИТВЫ]:"
DM_MAX_TOKENS = 1200
DM_TEMPERATURE = 0.65
TELEGRAM_MESSAGE_LIMIT = 4000
MAX_PLAYER_INPUT = 4000
MAX_DICE_COUNT = 100
MAX_DICE_SIDES = 1000
MAX_PARTY_SIZE = 6

_rng = random.SystemRandom()

# ID и username бота: заполняются один раз при запуске (см. main -> bot.get_me()).
BOT_ID: int = 0
BOT_USERNAME: str = ""

# ---------------------------------------------------------------------------
# 2. ЛОГГИРОВАНИЕ
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dnd-dm-bot")
logging.getLogger("aiogram.event").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 3. ОГРАНИЧЕНИЕ ДОСТУПА И ЧАСТОТЫ (белый список и антифлуд)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Читает целое число из переменной окружения (при ошибке — значение по умолчанию)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("Переменная %s=%r не является числом — беру %d.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Читает число с плавающей точкой из окружения (при ошибке — значение по умолчанию)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("Переменная %s=%r не является числом — беру %r.", name, raw, default)
        return default


def _parse_id_set(value: str) -> frozenset[int]:
    """Разбирает список Telegram user id из строки (через запятую или пробелы)."""
    ids: set[int] = set()
    for chunk in re.split(r"[\s,]+", value or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.isdigit():
            ids.add(int(chunk))
        else:
            logger.warning("Игнорирую некорректный user id: %r.", chunk)
    return frozenset(ids)


# Белый список игроков (пусто — пускаем всех) и администраторы (обходят лимиты).
ALLOWED_USER_IDS = _parse_id_set(os.getenv("ALLOWED_USER_IDS", ""))
ADMIN_USER_IDS = _parse_id_set(os.getenv("ADMIN_USER_IDS", ""))

RATE_LIMIT_MAX_REQUESTS = _env_int("RATE_LIMIT_MAX_REQUESTS", 20)
RATE_LIMIT_WINDOW_SECONDS = _env_float("RATE_LIMIT_WINDOW_SECONDS", 60.0)


def _is_admin(user_id: int) -> bool:
    """True, если пользователь — администратор.

    Пустой ADMIN_USER_IDS трактуется как «все администраторы» (удобно в разработке);
    в продакшене список задаётся в .env, и тогда доступ к /reset_party есть только у него.
    """
    return not ADMIN_USER_IDS or user_id in ADMIN_USER_IDS


# ---------------------------------------------------------------------------
# 4. СИСТЕМНЫЙ ПРОМПТ (DUNGEON MASTER для партии)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def build_system_prompt(*, species_name: Optional[str] = None, hero_ready: bool = False) -> str:
    """Собирает системный промпт Мастера под фазу игры.

    :param species_name: вид (раса) героя; нужен, когда герой уже создан.
    :param hero_ready: True — герои созданы (игра идёт): справочник правил сокращается —
        без раздела «Создание персонажа» и с расовыми особенностями текущего героя.
        False — этап создания героя: справочник полный.

    Функция кэшируется (lru_cache): большой справочник правил подмешивается один раз
    на сочетание «вид + фаза», а не пересобирается на каждый запрос к модели.
    """
    digest = build_reference_digest(full=not hero_ready, species_name=species_name)
    return "\n\n".join((BASE_DM_PROMPT, digest, CONTROL_BLOCK_INSTRUCTIONS))


# ---------------------------------------------------------------------------
# 5. СТАТИЧНЫЕ ТЕКСТЫ И СЛУЖЕБНЫЕ СООБЩЕНИЯ
# ---------------------------------------------------------------------------

WELCOME_TEXT = (
    "🐉 Добро пожаловать за стол, искатели приключений!\n\n"
    "Я — ваш Мастер Подземелий в духе Dungeons & Dragons 5e (2024, «Забытые Королевства»).\n"
    "Этот чат — отряд: я веду ОДНО приключение для всех вас, от 2 до 6 игроков.\n\n"
    "Чтобы вступить в партию, напишите /join — бот пришлёт кнопку и отправит вас в личные\n"
    "сообщения, где вы спокойно создадите героя (чтобы не загромождать общий чат):\n"
    "• назовите имя, вид (раса), класс и пару слов о внешности или характере;\n"
    "• либо напишите «Случайный герой» — код соберёт героя 1-го уровня по правилам PHB 2024;\n"
    "• затем подтвердите героя («да» или кнопка «✅ Подтвердить героя») — и он появится в группе.\n\n"
    "Когда герои готовы — начинается пролог. Дальше пишите, что делают ваши персонажи.\n"
    "Я отвечаю, когда вы обращаетесь ко мне: упомяните @бота, ответьте на моё сообщение "
    "или используйте команду.\n\n"
    "Команды: /join, /party, /sheet, /roll <кубик>, /inventory, /check, /spells, /rest."
)

JOIN_ALREADY_TEXT = "✅ Ты уже в отряде. Твой герой:\n\n{sheet}\n\nЧто ты делаешь?"

JOIN_PRIVATE_HINT = (
    "🐉 Этот бот ведёт партию в ГРУППОВОМ чате. Добавь меня в группу от 2 до 6 игроков "
    "и используй /join, чтобы создать героя."
)

# --- «Вариант Б»: создание персонажа в личных сообщениях ---

# Приглашение в группе: ведёт игрока в ЛС, чтобы не загромождать общий чат.
JOIN_GROUP_DM_TEXT = (
    "⚔️ {mention}, чтобы не загромождать чат, создание персонажа проходит в личных "
    "сообщениях с Мастером."
)

JOIN_DM_BUTTON_TEXT = "👤 Создать персонажа в ЛС"

# Подсказка, если username бота неизвестен (кнопка-ссылка недоступна).
JOIN_DM_NO_USERNAME_TEXT = (
    "⚠️ Не удалось собрать ссылку для перехода в ЛС. Откройте бота @BotFather, возьмите его "
    "username и напишите боту в личные сообщения команду /join."
)

# Сообщение в ЛС в начале создания героя.
JOIN_DM_START_TEXT = (
    "🎭 Создаём твоего героя для отряда этой группы.\n"
    "Опиши его (имя, вид, класс, внешность) или напиши «Случайный герой», а затем подтверди героя."
)

# Итог в ЛС, когда герой доставлен в группу.
PRIVATE_HERO_DONE_TEXT = "✅ Персонаж готов! Возвращайся в беседу."

# Подсказка в ЛС, если игрок уже провёл героя в группу (или у него уже есть герой).
PRIVATE_HERO_DONE_HINT = (
    "✅ Твой герой уже в отряде — продолжай игру в групповом чате, где стоит этот бот."
)

PRIVATE_HERO_ALREADY_TEXT = (
    "ℹ️ У тебя уже есть подтверждённый герой в этом отряде. Продолжай игру в групповом чате."
)

# Подсказка, если игрок открыл ЛС бота без перехода из группы.
PRIVATE_HERO_NO_TARGET_TEXT = (
    "🐉 Создание персонажа начинается из группового чата: напиши «/join» в беседе отряда "
    "и перейди по кнопке «👤 Создать персонажа в ЛС»."
)

# Торжественное объявление в группе о прибытии нового героя.
NEW_HERO_ANNOUNCEMENT = (
    "🎉 В отряд прибыл новый герой: {name} ({race} {class_name}, ур. {level})! "
    "[Игрок: {mention}]"
)

NEW_HERO_ANNOUNCEMENT_EMPTY_MENTION = "новый игрок"

# Deep-link аргумент команды /start: «join_<chat_id>» (chat_id группы, обычно отрицательный).
JOIN_DEEP_LINK_PATTERN = re.compile(r"^join_(-?\d+)$")

NOT_IN_PARTY_TEXT = (
    "Ты ещё не в отряде. Нажми /join, чтобы создать своего героя и присоединиться к партии."
)

PARTY_FULL_TEXT = "🧑‍🤝‍🧑 В отряде уже максимум игроков ({max_size}). Больше не влезет!"

PARTY_STATUS_HEADER = "🧭 ОТРЯД В ЭТОМ ПРИКЛЮЧЕНИИ"

RESET_PARTY_NO_RIGHTS_TEXT = "🔒 /reset_party доступна только администраторам."
RESET_PARTY_DONE_TEXT = (
    "🔄 Партия сброшена: все герои и история этого чата удалены. "
    "Создайте героев заново через /join."
)

NOBODY_CONFIRMED_TEXT = (
    "Отряд только собирается: пока никто не подтвердил героя. Используйте /join."
)

API_ERROR_TEXT = (
    "⚠️ Мастер ненадолго отвлёкся: не удалось связаться с оракулом.\n"
    "Попробуй повторить сообщение через несколько секунд."
)

GENERIC_ERROR_TEXT = (
    "⚠️ Что-то пошло не так при обработке действия. Попробуй ещё раз."
)

ACCESS_DENIED_TEXT = (
    "🔒 Извини, бот закрыт для посторонних. "
    "Попроси владельца добавить твой Telegram ID в список доступа."
)

RATE_LIMIT_TEXT = (
    "⏳ Слишком много сообщений подряд. "
    "Подожди несколько секунд и повтори — так Мастер успеет ответить как следует."
)

STALE_CALLBACK_TEXT = (
    "Эта кнопка устарела (сообщение слишком старое). "
    "Напиши новое действие в чат — под свежим ответом кнопки снова активны."
)

UNKNOWN_BUTTON_TEXT = "Неизвестная кнопка — попробуй ещё раз."

NOT_YOUR_BUTTON_TEXT = "Это действие другого персонажа!"

ACTION_MENU_TEXT = "🎲 Быстрые действия: выбери бросок или нужный раздел."

CHECKS_MENU_TEXT = (
    "🧠 ПРОВЕРКИ И СПАСБРОСКИ ХАРАКТЕРИСТИК\n"
    "Нажми характеристику — я брошу d20, добавлю её модификатор из твоего листа "
    "и передам Мастеру готовый итог.\n"
    "Кнопки со 🛡 — спасброски: к характеристике прибавляется бонус мастерства, "
    "если класс владеет этим спасброском."
)

ROLL_USAGE_TEXT = (
    "Формат броска: /roll <кубик>\n"
    "Примеры: /roll d20, /roll 2d6+3, /roll 1d10-1, /roll 4d6"
)

SPELLS_MENU_TEXT = (
    "🪄 КНИГА ЗАКЛИНАНИЙ\n"
    "Применяй заклинания, готовь их на день или отдыхай, чтобы восстановить ячейки."
)

NOT_SPELLCASTER_TEXT = (
    "🪄 У этого героя нет магии: его класс не владеет заклинаниями.\n"
    "Магией пользуются Бард, Жрец, Друид, Паладин, Следопыт, Чародей, Колдун и Волшебник."
)

SPELL_MENU_STALE_TEXT = "Меню заклинаний устарело. Открой его заново кнопкой «📜 Заклинания»."

SPELL_NOT_AVAILABLE_ALERT = (
    "Это заклинание сейчас недоступно: его нет в списке заговоров или оно не "
    "заготовлено на сегодня."
)

SPONTANEOUS_PREP_ALERT = (
    "Твой класс — спонтанный заклинатель: все изученные заклинания всегда готовы к "
    "применению, менять список подготовки не нужно."
)

CAST_MENU_TEXT = (
    "🔥 ЧТО ПРИМЕНИТЬ\n"
    "Нажми заклинание — код спишет ячейку нужного круга и передаст применение Мастеру. "
    "Заговоры ячеек не тратят."
)

PREP_MENU_TEXT = (
    "⚡ ПОДГОТОВКА ЗАКЛИНАНИЙ\n"
    "Нажми заклинание, чтобы заготовить или снять его: ✅ — готово к применению, "
    "❌ — не заготовлено."
)

REST_MENU_TEXT = (
    "🌙 ОТДЫХ\n"
    "Продолжительный отдых (8 часов) восстанавливает ВСЕ ячейки заклинаний.\n"
    "Короткий отдых (1 час) восстанавливает «магию пакты» Колдуна."
)

REST_NOT_CASTERTEXT = (
    "У этого героя нет магии, поэтому ячейки заклинаний восстанавливать нечего.\n"
    "Отдохнуть всё равно можно — просто опиши отдых словами."
)

# Фазы общения с игроком (см. creation_stage).
CREATION_STAGE_HERO = "creation"
CREATION_STAGE_CONFIRM = "confirmation"
CREATION_STAGE_PLAY = "play"

# Признаки того, что игрок просит сгенерировать героя вместо описания своего.
RANDOM_HERO_MARKERS: tuple[str, ...] = (
    "случайн", "наугад", "рандом", "random", "любой герой", "выбери за меня",
    "сгенерируй геро", "сгенерируй персон", "придумай за меня", "составь за меня",
)

# Согласие игрока подтвердить героя: короткое сообщение вида «да», «подтверждаю», «начинаем».
HERO_CONFIRM_PATTERN = re.compile(
    r"^\s*(?:я\s+)?(?:да|ага|угу|верно|всё верно|все верно|всё правильно|все правильно|"
    r"подтверждаю(?:\s+героя)?|согласен|согласна|ок|окей|окэй|хорошо|принято|готов|готова|"
    r"начинаем|начинай|поехали|играем|давай|yes|yep|ok|go)\s*,?\s*"
    r"(?:начинаем|начинай|поехали|играем|играть|давай|в путь|герой|героем|этим героем)?"
    r"\s*[!.,…]*\s*$",
    re.IGNORECASE,
)

# Имя героя из явного представления: «Меня зовут Боб», «зови меня Грим».
HERO_NAME_PATTERN = re.compile(
    r"(?:меня зовут|моё имя|мое имя|зови меня|зовут меня)\s+"
    r"(?P<name>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'\-]{1,31})",
    re.IGNORECASE,
)

# Имена и заготовки характера для случайного героя: (имя, описание внешности/характера).
RANDOM_HERO_PROFILES: tuple[tuple[str, str], ...] = (
    ("Каэлен", "Сухощавый, с обветренным лицом и внимательным взглядом; молчалив, но упрям."),
    ("Брунхильда", "Крепкая, с косой цвета соломы; смеётся громко, а в драке не отступает."),
    ("Тэм", "Невысокий ловкач с быстрыми глазами и вечной ухмылкой; ценит тишину и монету."),
    ("Селена", "Стройная, с серебристой прядью в волосах; говорит спокойно и смотрит прямо."),
    ("Грим", "Широкоплечий бородач со шрамами на руках; грубоват, но верен слову."),
    ("Нисса", "Хрупкая на вид, с цепкими пальцами и живым умом; любопытна до неприличия."),
    ("Ортега", "Загорелый, с кольцами в бороде; в пути поёт, чтобы не думать о прошлом."),
    ("Виллемина", "Молодая, с медной кожей и упрямым подбородком; верит, что удача — это умение."),
    ("Драган", "Высокий, с тяжёлым взглядом и спокойными движениями; сперва думает, потом бьёт."),
    ("Лиора", "Светловолосая, в дорожном плаще; собирает чужие истории, свою пока не рассказала."),
)

# ---------------------------------------------------------------------------
# 6. КУБИКИ (парсинг и броски считает код, а не нейросеть)
# ---------------------------------------------------------------------------

# Поддерживаем латинскую «d» и кириллическую «д», а также необязательные пробелы.
DICE_PATTERN = re.compile(
    r"^\s*(?P<count>\d{0,3})\s*[dDдД]\s*(?P<sides>\d{1,4})\s*(?P<modifier>[+-]\s*\d{1,4})?\s*$"
)

# Кости урона оружием: callback-действие -> число граней.
DAMAGE_DIE_SIDES: dict[str, int] = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12}

# Варианты броска d20: callback-действие -> режим ("" — обычный бросок).
D20_BUTTON_MODES: dict[str, str] = {"d20": "", "adv": "advantage", "dis": "disadvantage"}
D20_BUTTON_PURPOSES: dict[str, str] = {
    "d20": "бросок d20",
    "adv": "бросок d20 с преимуществом",
    "dis": "бросок d20 с помехой",
}

# Как оружие наносит урон: дальнобойное (ЛОВ), фехтовальное (лучшая из СИЛ/ЛОВ)
# или обычное рукопашное (СИЛ).
DAMAGE_ABILITY_RANGED = "ranged"
DAMAGE_ABILITY_FINESSE = "finesse"
DAMAGE_ABILITY_MELEE = "melee"

# Урон импровизированной атаки, если оружия в снаряжении нет.
IMPROVISED_DAMAGE_DICE = "1d4"
IMPROVISED_DAMAGE_TYPE = "дробящий"


@dataclass(slots=True)
class DiceRoll:
    """Результат броска кубиков в конкретной нотации.

    ``mode`` («advantage»/«disadvantage») используется только для d20:
    бросаются два кубика, а в зачёт идёт лучший/худший (см. ``kept``).
    Пустой список ``rolls`` означает фиксированное значение без броска кубиков.
    """

    count: int
    sides: int
    rolls: list[int] = field(default_factory=list)
    modifier: int = 0
    mode: str = ""

    @classmethod
    def flat(cls, amount: int) -> "DiceRoll":
        """Бросок без кубиков: фиксированное значение (например, урон «1»)."""
        return cls(count=0, sides=0, rolls=[], modifier=_as_int(amount))

    @property
    def is_flat(self) -> bool:
        """True, если кубики не бросались (только фиксированный модификатор)."""
        return not self.rolls

    @property
    def kept(self) -> list[int]:
        """Кубики, которые идут в зачёт (для преимущества/помехи — лучший/худший)."""
        if not self.rolls:
            return []
        if self.mode == "advantage":
            return [max(self.rolls)]
        if self.mode == "disadvantage":
            return [min(self.rolls)]
        return list(self.rolls)

    @property
    def total(self) -> int:
        """Итог броска: зачётные кубики + модификатор."""
        return sum(self.kept) + self.modifier

    @property
    def notation(self) -> str:
        """Нотация броска, например «2d6+3»."""
        if self.is_flat:
            return format_modifier(self.modifier)
        notation = f"{self.count}d{self.sides}"
        if self.modifier:
            notation += f"{self.modifier:+d}"
        return notation

    @property
    def mode_label(self) -> str:
        """Подпись варианта броска d20: преимущество, помеха или пусто."""
        if self.mode == "advantage":
            return " (преимущество)"
        if self.mode == "disadvantage":
            return " (помеха)"
        return ""

    def describe(self) -> str:
        """Человекочитаемая «математика» броска для игрока."""
        if self.is_flat:
            return f"{self.notation}{self.mode_label}: {self.total}"
        if self.mode and len(self.rolls) > 1:
            body = f"из {self.rolls[0]}/{self.rolls[1]} в зачёт {self.kept[0]}"
        else:
            body = " + ".join(str(value) for value in self.kept)
        if self.modifier > 0:
            body += f" + {self.modifier}"
        elif self.modifier < 0:
            body += f" - {abs(self.modifier)}"
        return f"{self.notation}{self.mode_label}: {body} = {self.total}"

    def context_message(self, purpose: str = "") -> str:
        """Служебное сообщение о броске для контекста Мастера.

        Математика броска уже посчитана кодом, поэтому Мастеру передаётся неоспоримое
        равенство «кубик + мод = ИТОГ»: он не должен пересчитывать модификаторы, метать
        кубик заново или подменять итог.
        """
        dice_total = sum(self.kept)
        if self.is_flat:
            detail = " (без броска кубиков, фиксированное значение)"
        elif self.mode and len(self.rolls) > 1:
            detail = f" (d20: {self.rolls[0]}/{self.rolls[1]}, в зачёт {self.kept[0]})"
        elif len(self.rolls) > 1:
            detail = f" ({self.count}d{self.sides}: {', '.join(map(str, self.rolls))})"
        else:
            detail = ""
        reason = f" Повод: {purpose}." if purpose else ""
        return (
            f"[СИСТЕМА]: Игрок выбросил на кубике {dice_total} + мод {self.modifier:+d} "
            f"= ИТОГ {self.total}{detail}.{reason} {DM_ROLL_INSTRUCTION}"
        )


def make_roll(count: int, sides: int, modifier: int = 0, mode: str = "") -> DiceRoll:
    """Бросает кубики кодом бота (кнопки, проверки, урон). Значения защищены от «мусора»."""
    count = max(1, min(_as_int(count), MAX_DICE_COUNT))
    sides = max(2, min(_as_int(sides), MAX_DICE_SIDES))
    rolls = [_rng.randint(1, sides) for _ in range(count)]
    return DiceRoll(count=count, sides=sides, rolls=rolls, modifier=_as_int(modifier), mode=mode)


def parse_and_roll(expression: str) -> DiceRoll:
    """Разбирает нотацию кубиков и выполняет бросок.

    :raise ValueError: если нотация некорректна или выходит за допустимые пределы.
    """
    match = DICE_PATTERN.match(expression)
    if match is None:
        raise ValueError("Неверный формат броска.")

    count = int(match.group("count") or 1)
    sides = int(match.group("sides"))
    raw_modifier = match.group("modifier")
    modifier = int(raw_modifier.replace(" ", "")) if raw_modifier else 0

    if not 1 <= count <= MAX_DICE_COUNT:
        raise ValueError(f"Число кубиков должно быть от 1 до {MAX_DICE_COUNT}.")
    if not 2 <= sides <= MAX_DICE_SIDES:
        raise ValueError(f"Число граней кубика должно быть от 2 до {MAX_DICE_SIDES}.")

    return make_roll(count=count, sides=sides, modifier=modifier)


# ---------------------------------------------------------------------------
# 7. НИЗКОУРОВНЕВЫЕ ХЕЛПЕРЫ (терпимы к «мусору» из JSON модели)
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int:
    """Аккуратно приводит значение из «сырого» JSON модели к int (0 при неудаче)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _as_change_int(value: Any) -> int:
    """Приводит изменение листа (XP/HP/gp) к int, терпимо к строкам «+50 XP» / «-10 hp»."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = _SIGNED_INT_IN_TEXT.search(value)
        if match is not None:
            return int(match.group())
    return 0


def _as_text_list(value: Any) -> list[str]:
    """Приводит значение к списку непустых строк (для add_items / remove_items)."""
    if isinstance(value, str):
        candidates: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        return []

    items: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            candidate = str(candidate)
        if isinstance(candidate, str) and candidate.strip():
            items.append(candidate.strip()[:80])
    return items


def _unique_spell_list(value: Any) -> list[str]:
    """Приводит значение к списку названий без пустых строк и дублей."""
    items: list[str] = []
    seen: set[str] = set()
    for candidate in _as_text_list(value):
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(candidate)
    return items


def _normalize_ability_codes(value: Any) -> list[str]:
    """Приводит значение к списку кодов характеристик ('str', 'dex', ...) без дублей."""
    codes: list[str] = []
    seen: set[str] = set()
    for candidate in _as_text_list(value):
        code = normalize_ability_key(candidate)
        if code is None and candidate.lower() in ABILITIES:
            code = candidate.lower()
        if code is not None and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _remove_first(items: list[str], target: str) -> bool:
    """Удаляет первое совпадение по названию (без учёта регистра). True при успехе."""
    needle = target.strip().lower()
    for index, item in enumerate(items):
        if item.lower() == needle:
            del items[index]
            return True
    return False


def _as_optional_text(value: Any) -> Optional[str]:
    """Возвращает непустую строку или None (для загрузки полей из базы)."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _as_optional_label(value: Any) -> Optional[str]:
    """Возвращает непустую текстовую метку (локация, цель) или None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# Извлекает первое число из строки вида «+50 XP», «-10 hp», «15».
_SIGNED_INT_IN_TEXT = re.compile(r"[-+]?\d+")

# Значения характеристик по умолчанию: 10 — «средний» герой без распределённых очков.
DEFAULT_ABILITIES: dict[str, int] = {code: 10 for code in ABILITIES}

# Заглушки для ещё не созданного героя.
DEFAULT_NAME = "Безымянный герой"
DEFAULT_RACE = "Не определена"
DEFAULT_CLASS = "Не определён"

# Краткая сводка локации (HUD): где отряд и какова его цель, пока Мастер не сменил их.
DEFAULT_LOCATION = "Неизвестно"
DEFAULT_QUEST = "Исследовать местность"

# Границы значений характеристик по правилам D&D.
MIN_ABILITY, MAX_ABILITY = 1, 30

# Особое значение current_hp «ещё не задано»: при создании HP = максимум.
UNSET_HP = -1


# ---------------------------------------------------------------------------
# 8. ЛИСТ ПЕРСОНАЖА (Character Sheet)
# ---------------------------------------------------------------------------


@dataclass
class Character:
    """Лист одного героя партии (D&D 2024).

    Партия в этом чате держит по одному такому листу на игрока (см. PartySession).
    Развивается автоматически по опыту через стандартные пороги уровней D&D 2024.
    Сводка локации и цели — общая для отряда и живёт в PartySession (не здесь).
    """

    name: str = DEFAULT_NAME
    race: str = DEFAULT_RACE
    class_name: str = DEFAULT_CLASS
    level: int = 1
    current_hp: int = UNSET_HP
    max_hp: int = 0
    xp: int = 0
    gp: int = 0
    abilities: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ABILITIES))
    inventory: list[str] = field(default_factory=list)
    heroic_inspiration: bool = False
    description: str = ""
    # Признак «игрок подтвердил героя». Пока он False, идёт этап создания персонажа.
    hero_confirmed: bool = False
    # --- Магия (spellcasting) ---
    is_spellcaster: bool = False
    spellcasting_ability: str = ""
    spell_slots: dict[str, dict[str, int]] = field(default_factory=dict)
    cantrips: list[str] = field(default_factory=list)
    spells_known: list[str] = field(default_factory=list)
    spells_prepared: list[str] = field(default_factory=list)
    # --- Владения (proficiencies) ---
    save_proficiencies: list[str] = field(default_factory=list)
    skill_proficiencies: list[str] = field(default_factory=list)
    weapon_proficiencies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Нормализует «сырые» данные: обрезает строки и приводит числа к правилам."""
        self.name = (self.name or "").strip()[:64] or DEFAULT_NAME
        self.race = (self.race or "").strip()[:64] or DEFAULT_RACE
        self.class_name = (self.class_name or "").strip()[:64] or DEFAULT_CLASS
        self.level = max(1, min(_as_int(self.level) or 1, MAX_LEVEL))

        normalized: dict[str, int] = {}
        for code in ABILITIES:
            value = _as_int(self.abilities.get(code, 10))
            normalized[code] = max(MIN_ABILITY, min(value, MAX_ABILITY))
        self.abilities = normalized

        self.xp = max(0, _as_int(self.xp))
        self.gp = max(0, _as_int(self.gp))

        if self.max_hp <= 0:
            # HP 1-го уровня = максимум кости хитов + модификатор ТЕЛ.
            self.max_hp = max(1, self.hit_die + self.ability_mod("con"))
        if self.current_hp < 0:
            self.current_hp = self.max_hp
        else:
            self.current_hp = max(0, min(_as_int(self.current_hp), self.max_hp))
        self.inventory = _as_text_list(self.inventory)
        self.description = (self.description or "").strip()[:400]
        # Владения: чистим списки и подтягиваем владения спасбросками из класса (PHB 2024).
        self.save_proficiencies = _normalize_ability_codes(self.save_proficiencies)
        self.skill_proficiencies = _unique_spell_list(self.skill_proficiencies)
        self.weapon_proficiencies = _unique_spell_list(self.weapon_proficiencies)
        self._sync_save_proficiencies()
        # Магия: ячейки, заговоры и заклинания приводим к правилам класса и уровня.
        self._normalize_spellcasting()

    def _normalize_spellcasting(self) -> None:
        """Инициализирует поля магии по классу и уровню (безопасно для записей прежних версий).

        Если у класса есть магия, а списков ещё нет, они заполняются классовыми заговорами
        и стартовыми заклинаниями 1-го уровня. Текущий остаток ячеек из базы сохраняется.
        """
        ability = spellcasting_ability_for_class(self.class_name)
        if ability is None:
            self.is_spellcaster = False
            self.spellcasting_ability = ""
            self.spell_slots = {}
            self.cantrips = []
            self.spells_known = []
            self.spells_prepared = []
            return

        self.is_spellcaster = True
        self.spellcasting_ability = ability

        # Количество ячеек берём из таблиц правил, остаток (current) — из базы, если он есть.
        totals = spell_slots_for_level(self.class_name, self.level)
        raw_slots = self.spell_slots if isinstance(self.spell_slots, Mapping) else {}
        normalized_slots: dict[str, dict[str, int]] = {}
        for circle, total in totals.items():
            stored = raw_slots.get(circle)
            current = total
            if isinstance(stored, Mapping):
                current = max(0, min(_as_int(stored.get("current", total)), total))
            normalized_slots[circle] = {"total": total, "current": current}
        self.spell_slots = normalized_slots

        self.cantrips = _unique_spell_list(self.cantrips)
        self.spells_known = _unique_spell_list(self.spells_known)
        self.spells_prepared = _unique_spell_list(self.spells_prepared)

        # Заговоры и заклинания 1+ круга — непересекающиеся списки.
        cantrip_set = set(self.cantrips)
        self.spells_known = [name for name in self.spells_known if name not in cantrip_set]
        self.spells_prepared = [name for name in self.spells_prepared if name not in cantrip_set]

        if not self.cantrips:
            self.cantrips = list(default_cantrips_for_class(self.class_name))
        if not self.spells_known:
            self.spells_known = list(default_known_spells_for_class(self.class_name))

        if is_spontaneous_caster(self.class_name):
            self.spells_prepared = list(self.spells_known)
            return

        prepared = [name for name in self.spells_prepared if name in self.spells_known]
        limit = self.max_prepared
        if not prepared:
            prepared = list(self.spells_known[:limit])
        self.spells_prepared = prepared[:limit]

    def init_default_spellcasting(self) -> None:
        """Заполняет магию «по умолчанию» для нового героя (класс и характеристики заданы)."""
        if not self.is_spellcaster:
            return
        self.cantrips = list(default_cantrips_for_class(self.class_name))
        self.spells_known = list(default_known_spells_for_class(self.class_name))
        if is_spontaneous_caster(self.class_name):
            self.spells_prepared = list(self.spells_known)
        else:
            self.spells_prepared = list(self.spells_known[: self.max_prepared])

    # --- Проверки состояния персонажа ---

    @property
    def is_created(self) -> bool:
        """True, если игрок уже описал героя: заданы имя, вид (раса) и класс."""
        return (
            self.name != DEFAULT_NAME
            and self.race != DEFAULT_RACE
            and self.class_name != DEFAULT_CLASS
        )

    # --- Производные характеристики (считаются по правилам) ---

    def ability_mod(self, code: str) -> int:
        """Модификатор характеристики по её коду ('str', 'dex', ...)."""
        return ability_modifier(self.abilities.get(code, 10))

    @property
    def hit_die(self) -> int:
        """Число граней кости хитов класса (d6/d8/d10/d12)."""
        return hit_die_sides(self.class_name)

    @property
    def proficiency_bonus(self) -> int:
        """Бонус мастерства по текущему уровню."""
        return proficiency_bonus(self.level)

    def roll_modifier(self, code: str, *, proficient: bool = False) -> int:
        """Модификатор броска d20: характеристика (+ бонус мастерства при владении)."""
        modifier = self.ability_mod(code)
        if proficient:
            modifier += self.proficiency_bonus
        return modifier

    def save_modifier(self, code: str) -> int:
        """Модификатор спасброска: характеристика + бонус мастерства при владении классом."""
        return self.roll_modifier(code, proficient=code in self.save_proficiencies)

    def skill_modifier(self, skill: str) -> int:
        """Модификатор проверки навыка: характеристика навыка + бонус мастерства при владении."""
        ability = SKILLS.get(skill)
        if ability is None:
            return 0
        owned = {name.strip().lower() for name in self.skill_proficiencies}
        return self.roll_modifier(ability, proficient=skill.strip().lower() in owned)

    def _sync_save_proficiencies(self, force: bool = False) -> None:
        """Заполняет владения спасбросками по классу (PHB 2024)."""
        if self.save_proficiencies and not force:
            return
        key = resolve_class_key(self.class_name)
        if key is None:
            return
        self.save_proficiencies = [
            code for code in CLASSES[key].get("saves", ()) if code in ABILITIES
        ]

    @property
    def initiative(self) -> int:
        """Модификатор инициативы (равен модификатору ЛОВ)."""
        return self.ability_mod("dex")

    @property
    def armor_class(self) -> int:
        """КД по правилам PHB 2024: доспех из снаряжения + модификатор ЛОВ (+2 за щит)."""
        armor = self.best_armor
        dex_mod = self.ability_mod("dex")

        if armor is None:
            armor_class = 10 + dex_mod
        else:
            _name, base_ac, max_dex = armor
            if max_dex == 0:            # тяжёлый доспех — модификатор ЛОВ не добавляется
                armor_class = base_ac
            elif max_dex is None:       # лёгкий доспех — модификатор ЛОВ без ограничения
                armor_class = base_ac + dex_mod
            else:                       # средний доспех — модификатор ЛОВ не выше предела
                armor_class = base_ac + min(dex_mod, max_dex)

        if self.has_shield:
            armor_class += SHIELD_BONUS
        return armor_class

    @property
    def best_armor(self) -> Optional[tuple[str, int, Optional[int]]]:
        """Лучший доспех из снаряжения: (название, база КД, предел модификатора ЛОВ)."""
        best: Optional[tuple[str, int, Optional[int]]] = None
        for item in self.inventory:
            lowered = item.strip().lower()
            for name, _category, base_ac, max_dex, _strength, _stealth in ARMOR:
                if name.lower() in lowered and (best is None or base_ac > best[1]):
                    best = (name, base_ac, max_dex)
        return best

    @property
    def has_shield(self) -> bool:
        """Есть ли щит в снаряжении героя."""
        return any("щит" in item.lower() for item in self.inventory)

    @property
    def armor_label(self) -> str:
        """Подпись доспеха для листа: «Кольчуга, щит» или «без доспехов»."""
        names: list[str] = []
        if self.best_armor is not None:
            names.append(self.best_armor[0])
        if self.has_shield:
            names.append("щит")
        return ", ".join(names) if names else "без доспехов"

    @property
    def passive_perception(self) -> int:
        """Пассивная Внимательность: 10 + модификатор МУД."""
        return 10 + self.ability_mod("wis")

    @property
    def is_alive(self) -> bool:
        """Жив ли персонаж (HP выше нуля)."""
        return self.current_hp > 0

    # --- Магия: производные величины и операции с ячейками ---

    @property
    def spell_save_dc(self) -> int:
        """КС спасброска от заклинаний: 8 + бонус мастерства + модификатор характеристики."""
        if not self.is_spellcaster:
            return 0
        return 8 + self.proficiency_bonus + self.ability_mod(self.spellcasting_ability)

    @property
    def spell_attack_bonus(self) -> int:
        """Модификатор атаки заклинанием: бонус мастерства + модификатор характеристики."""
        if not self.is_spellcaster:
            return 0
        return self.proficiency_bonus + self.ability_mod(self.spellcasting_ability)

    @property
    def max_prepared(self) -> int:
        """Лимит заготовленных заклинаний (уровень + мод. характеристики, мин. 1).

        Для спонтанных заклинателей лимита нет — возвращается число изученных заклинаний.
        Для класса без магии возвращается 0.
        """
        if not self.is_spellcaster:
            return 0
        limit = max_prepared_spells(
            self.class_name, self.level, self.ability_mod(self.spellcasting_ability or "int")
        )
        if limit is None:
            return len(self.spells_known)
        return max(1, limit)

    @property
    def total_slots(self) -> int:
        """Суммарное число ячеек заклинаний всех кругов."""
        return sum(slot["total"] for slot in self.spell_slots.values())

    @property
    def available_slots(self) -> int:
        """Сколько ячеек заклинаний сейчас свободно."""
        return sum(slot["current"] for slot in self.spell_slots.values())

    def spell_circle(self, spell_name: str) -> int:
        """Круг заклинания: 0 — заговор, иначе круг из справочника (по умолчанию 1)."""
        if spell_name in self.cantrips:
            return 0
        level = spell_level(spell_name)
        return 1 if level is None else level

    def castable_spells(self) -> list[tuple[str, int]]:
        """Доступные к применению заклинания: (название, круг); заговоры идут первыми."""
        entries: list[tuple[str, int]] = [(name, 0) for name in self.cantrips]
        entries += [(name, self.spell_circle(name)) for name in self.spells_prepared]
        return entries

    def can_prepare_more(self) -> bool:
        """Есть ли ещё место в лимите подготовки заклинаний."""
        return len(self.spells_prepared) < self.max_prepared

    def toggle_prepared(self, spell_name: str) -> bool:
        """Переключает заготовку заклинания. True — заготовлено, False — снято."""
        if spell_name in self.spells_prepared:
            self.spells_prepared = [name for name in self.spells_prepared if name != spell_name]
            return False
        self.spells_prepared.append(spell_name)
        return True

    def cast_spell(self, spell_name: str) -> "SpellCastResult":
        """Списывает ячейку круга при применении заклинания (заговоры ячеек не тратят).

        Применить можно только заговор или заготовленное заклинание.
        """
        if spell_name not in self.cantrips and spell_name not in self.spells_prepared:
            return SpellCastResult(ok=False, alert=SPELL_NOT_AVAILABLE_ALERT)

        circle = self.spell_circle(spell_name)
        circle_label = SPELL_LEVEL_RU.get(circle, f"{circle} круг")

        if circle == 0:
            return SpellCastResult(
                ok=True,
                note=f"🪄 Ты применяешь заговор «{spell_name}» (ячейки не тратятся).",
                context=(
                    f"[СИСТЕМА] Игрок применяет заклинание '{spell_name}' "
                    f"(заговор, ячейки не требуются)."
                ),
            )

        slot = self.spell_slots.get(str(circle))
        if not slot or slot["total"] <= 0:
            return SpellCastResult(
                ok=False,
                alert=f"Заклинания {circle_label} тебе пока недоступны.",
            )
        if slot["current"] <= 0:
            return SpellCastResult(
                ok=False,
                alert=f"🔒 Нет свободных ячеек {circle_label}! Отдохни, чтобы восстановить их.",
            )

        slot["current"] -= 1
        left, total = slot["current"], slot["total"]
        return SpellCastResult(
            ok=True,
            note=(
                f"🪄 Ты применяешь «{spell_name}» ({circle_label}). "
                f"Осталось ячеек {circle_label}: {left}/{total}."
            ),
            context=(
                f"[СИСТЕМА] Игрок применяет заклинание '{spell_name}' ({circle_label}). "
                f"Осталось ячеек {circle_label}: {left}/{total}."
            ),
        )

    def restore_spell_slots(self, long_rest: bool) -> list[str]:
        """Восстанавливает ячейки при отдыхе. Возвращает заметки для игрока.

        Продолжительный отдых возвращает все ячейки; короткий — только «магию пакта» Колдуна.
        """
        if not self.spell_slots:
            return []
        if not (long_rest or is_pact_caster(self.class_name)):
            return []

        notes: list[str] = []
        for circle, slot in self.spell_slots.items():
            slot["current"] = slot["total"]
            label = SPELL_LEVEL_RU.get(_as_int(circle), f"{circle} круг")
            notes.append(f"🔋 {label}: {slot['current']}/{slot['total']}.")
        return notes

    # --- Изменение листа персонажа ---

    def _level_up(self) -> str:
        """Повышает уровень: увеличивает максимум HP и лечит на ту же величину."""
        self.level += 1
        gained = average_hit_points(self.hit_die, self.ability_mod("con"))
        self.max_hp += gained
        self.current_hp = min(self.max_hp, self.current_hp + gained)
        self._normalize_spellcasting()
        return (
            f"⬆️ НОВЫЙ УРОВЕНЬ: {self.level}! Максимум HP: {self.max_hp} (+{gained}). "
            f"Бонус мастерства: {format_modifier(self.proficiency_bonus)}."
        )

    def _apply_level_ups(self) -> list[str]:
        """Повышает уровень столько раз, сколько позволяет накопленный опыт."""
        notes: list[str] = []
        while self.level < MAX_LEVEL:
            threshold = next_xp_threshold(self.level)
            if threshold is None or self.xp < threshold:
                break
            notes.append(self._level_up())
        return notes

    def _apply_optional_sheet(self, data: dict[str, Any]) -> list[str]:
        """Применяет необязательные поля листа (создание или правка персонажа Мастером)."""
        notes: list[str] = []

        name = data.get("name")
        if isinstance(name, str) and name.strip() and name.strip() != self.name:
            self.name = name.strip()[:64]
            notes.append(f"📛 Имя персонажа: {self.name}.")

        race = data.get("race")
        if isinstance(race, str) and race.strip() and race.strip() != self.race:
            self.race = race.strip()[:64]
            notes.append(f"🧬 Раса: {self.race}.")

        class_name = data.get("class_name")
        if isinstance(class_name, str) and class_name.strip():
            cleaned = class_name.strip()[:64]
            if cleaned != self.class_name:
                self.class_name = cleaned
                notes.append(f"⚔️ Класс: {self.class_name} (кость хитов d{self.hit_die}).")
                # Смена класса полностью пересобирает магию: списки заклинаний и ячейки.
                self.cantrips = []
                self.spells_known = []
                self.spells_prepared = []
                self.spell_slots = {}
                self._normalize_spellcasting()
                self._sync_save_proficiencies(force=True)

        description = data.get("description")
        if isinstance(description, str) and description.strip():
            cleaned_description = description.strip()[:400]
            if cleaned_description != self.description:
                self.description = cleaned_description
                notes.append("🖋️ Описание героя записано.")

        level = _as_int(data.get("level"))
        if 1 <= level <= MAX_LEVEL and level != self.level:
            self.level = level
            notes.append(f"🎖️ Уровень: {self.level}.")
            self._normalize_spellcasting()

        abilities = data.get("abilities")
        if isinstance(abilities, dict):
            changed: list[str] = []
            for raw_key, raw_value in abilities.items():
                code = normalize_ability_key(raw_key)
                if code is None:
                    continue
                value = max(MIN_ABILITY, min(_as_int(raw_value), MAX_ABILITY))
                if value != self.abilities[code]:
                    self.abilities[code] = value
                    changed.append(f"{ABILITIES[code]} {value}")
            if changed:
                notes.append("🧠 Характеристики: " + ", ".join(changed) + ".")
                if self.is_spellcaster and not is_spontaneous_caster(self.class_name):
                    self.spells_prepared = self.spells_prepared[: self.max_prepared]

        max_hp = _as_int(data.get("max_hp"))
        if max_hp > 0:
            self.max_hp = max_hp
            self.current_hp = min(self.current_hp, self.max_hp)

        if data.get("current_hp") is not None:
            self.current_hp = max(0, min(_as_int(data.get("current_hp")), self.max_hp))

        inspiration = data.get("heroic_inspiration")
        if isinstance(inspiration, bool):
            self.heroic_inspiration = inspiration
            if inspiration:
                notes.append("🌟 Вдохновение героя получено!")

        return notes

    def apply_control(self, data: dict[str, Any]) -> list[str]:
        """Применяет служебный JSON-блок Мастера к листу ЭТОГО персонажа.

        Возвращает список уведомлений для игрока (опыт, урон, предметы, золото, уровень).
        Сводка локации и цели сюда не входит — она общая для партии (см. PartySession).
        """
        notes: list[str] = list(self._apply_optional_sheet(data))

        xp_gained = _as_change_int(data.get("xp_gained"))
        if xp_gained > 0:
            self.xp += xp_gained
            notes.append(f"✨ Получено {xp_gained} XP (всего {self.xp}).")
            notes.extend(self._apply_level_ups())

        hp_change = _as_change_int(data.get("hp_change"))
        if hp_change:
            before = self.current_hp
            self.current_hp = max(0, min(self.current_hp + hp_change, self.max_hp))
            delta = self.current_hp - before
            if delta < 0:
                notes.append(f"💔 Потеряно {-delta} HP (осталось {self.current_hp}/{self.max_hp}).")
            elif delta > 0:
                notes.append(f"💚 Восстановлено {delta} HP (теперь {self.current_hp}/{self.max_hp}).")
            if self.current_hp == 0:
                notes.append("☠️ Персонаж без сознания (0 HP) — нужны спасброски от смерти.")

        for item in _as_text_list(data.get("add_items")):
            self.inventory.append(item)
            notes.append(f"🎁 Получено: {item}.")

        for item in _as_text_list(data.get("remove_items")):
            if _remove_first(self.inventory, item):
                notes.append(f"➖ Потеряно: {item}.")

        gp_change = _as_change_int(data.get("gp_change"))
        if gp_change:
            self.gp = max(0, self.gp + gp_change)
            if gp_change > 0:
                notes.append(f"💰 +{gp_change} золота (всего {self.gp} gp).")
            else:
                notes.append(f"💸 Потрачено {-gp_change} золота (осталось {self.gp} gp).")

        return notes

    # --- Сериализация для постоянного хранения (SQLite) ---

    def to_dict(self) -> dict[str, Any]:
        """Сериализует лист персонажа в JSON-совместимый словарь."""
        return {
            "name": self.name,
            "race": self.race,
            "class_name": self.class_name,
            "level": self.level,
            "current_hp": self.current_hp,
            "max_hp": self.max_hp,
            "xp": self.xp,
            "gp": self.gp,
            "abilities": {code: int(self.abilities[code]) for code in ABILITIES},
            "inventory": list(self.inventory),
            "heroic_inspiration": bool(self.heroic_inspiration),
            "description": self.description,
            "hero_confirmed": bool(self.hero_confirmed),
            "is_spellcaster": bool(self.is_spellcaster),
            "spellcasting_ability": self.spellcasting_ability,
            "spell_slots": {
                str(circle): {"total": int(slot["total"]), "current": int(slot["current"])}
                for circle, slot in self.spell_slots.items()
            },
            "cantrips": list(self.cantrips),
            "spells_known": list(self.spells_known),
            "spells_prepared": list(self.spells_prepared),
            "save_proficiencies": list(self.save_proficiencies),
            "skill_proficiencies": list(self.skill_proficiencies),
            "weapon_proficiencies": list(self.weapon_proficiencies),
        }

    def to_json(self) -> str:
        """Сериализует лист персонажа в строку JSON (с русскими буквами как есть)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Any) -> "Character":
        """Восстанавливает лист персонажа из словаря.

        Терпима к «мусору»: отсутствующие поля берутся по умолчанию, лишние
        игнорируются, а испорченные значения приводятся к допустимым.
        """
        if not isinstance(data, Mapping):
            return cls()

        abilities: dict[str, int] = {}
        raw_abilities = data.get("abilities")
        if isinstance(raw_abilities, Mapping):
            for raw_key, raw_value in raw_abilities.items():
                code = normalize_ability_key(raw_key)
                if code is None and raw_key in ABILITIES:
                    code = raw_key
                if code is not None:
                    abilities[code] = _as_int(raw_value)

        inspiration = data.get("heroic_inspiration")
        # Отсутствие current_hp означает «не задано» (полное здоровье), а не 0 HP.
        current_hp = UNSET_HP if data.get("current_hp") is None else _as_int(data.get("current_hp"))

        name = _as_optional_text(data.get("name")) or DEFAULT_NAME
        race = _as_optional_text(data.get("race")) or DEFAULT_RACE
        class_name = _as_optional_text(data.get("class_name")) or DEFAULT_CLASS

        raw_confirmed = data.get("hero_confirmed")
        if isinstance(raw_confirmed, bool):
            hero_confirmed = raw_confirmed
        else:
            hero_confirmed = (
                name != DEFAULT_NAME and race != DEFAULT_RACE and class_name != DEFAULT_CLASS
            )

        raw_slots = data.get("spell_slots")
        spell_slots: dict[str, dict[str, int]] = {}
        if isinstance(raw_slots, Mapping):
            for raw_circle, raw_slot in raw_slots.items():
                circle = str(raw_circle).strip()
                if not circle or not isinstance(raw_slot, Mapping):
                    continue
                spell_slots[circle] = {
                    "total": _as_int(raw_slot.get("total")),
                    "current": _as_int(raw_slot.get("current")),
                }

        return cls(
            name=name,
            race=race,
            class_name=class_name,
            level=_as_int(data.get("level")) or 1,
            current_hp=current_hp,
            max_hp=_as_int(data.get("max_hp")),
            xp=_as_int(data.get("xp")),
            gp=_as_int(data.get("gp")),
            abilities=abilities or dict(DEFAULT_ABILITIES),
            inventory=_as_text_list(data.get("inventory")),
            heroic_inspiration=inspiration if isinstance(inspiration, bool) else False,
            description=_as_optional_text(data.get("description")) or "",
            hero_confirmed=hero_confirmed,
            is_spellcaster=bool(data.get("is_spellcaster")),
            spellcasting_ability=_as_optional_text(data.get("spellcasting_ability")) or "",
            spell_slots=spell_slots,
            cantrips=_unique_spell_list(data.get("cantrips")),
            spells_known=_unique_spell_list(data.get("spells_known")),
            spells_prepared=_unique_spell_list(data.get("spells_prepared")),
            save_proficiencies=_normalize_ability_codes(data.get("save_proficiencies")),
            skill_proficiencies=_unique_spell_list(data.get("skill_proficiencies")),
            weapon_proficiencies=_unique_spell_list(data.get("weapon_proficiencies")),
        )

    @classmethod
    def from_json(cls, raw: Any) -> "Character":
        """Восстанавливает лист персонажа из JSON-строки (при ошибке — новый герой)."""
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str):
            return cls()
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("Повреждённая запись листа персонажа в базе — создаю нового героя.")
            return cls()
        return cls.from_dict(data)


@dataclass
class SpellCastResult:
    """Результат применения заклинания кодом бота (см. Character.cast_spell)."""

    ok: bool
    note: str = ""
    context: str = ""
    alert: str = ""


# ---------------------------------------------------------------------------
# 9. ФОРМАТИРОВАНИЕ (лист персонажа, отряд, ячейки заклинаний)
# ---------------------------------------------------------------------------


def hp_progress_bar(current_hp: int, max_hp: int, width: int = 20) -> str:
    """Рисует прогресс-бар HP, например: ████████░░░░░░░░░░░░."""
    ratio = 0.0 if max_hp <= 0 else max(0.0, min(1.0, current_hp / max_hp))
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def format_hud(location: str, quest: str) -> str:
    """Краткая сводка локации и цели отряда (HUD) перед ответами Мастера."""
    return f"📍 Локация: {location or DEFAULT_LOCATION}\n🎯 Текущая цель: {quest or DEFAULT_QUEST}"


def _spell_circle_label(circle: int) -> str:
    """Русское название круга заклинания для карточек и кнопок."""
    return SPELL_LEVEL_RU.get(circle, f"{circle} круг")


def _spell_circle_locked(character: Character, circle: int) -> bool:
    """Закончились ли ячейки нужного круга (заговоры не «запираются» никогда)."""
    if circle <= 0:
        return False
    slot = character.spell_slots.get(str(circle))
    return slot is None or slot["current"] <= 0


def format_slots_tracker(spell_slots: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Строки трекера ячеек заклинаний: «🔋 1 круг: [████░░░░] 2/4»."""
    if not spell_slots:
        return ["🔋 Ячейки: нет — заговоры ячеек не тратят."]
    lines: list[str] = []
    for circle in sorted(spell_slots, key=lambda value: _as_int(value)):
        slot = spell_slots[circle]
        total = _as_int(slot.get("total"))
        current = _as_int(slot.get("current"))
        label = _spell_circle_label(_as_int(circle))
        lines.append(f"🔋 {label}: [{hp_progress_bar(current, total, 8)}] {current}/{total}")
    return lines


def format_character_sheet(character: Character) -> str:
    """Форматирует лист персонажа в аккуратное сообщение для игрока."""
    threshold = next_xp_threshold(character.level)
    if threshold is None:
        xp_line = f"Опыт: {character.xp} XP (достигнут максимальный уровень)"
    else:
        xp_line = f"Опыт: {character.xp} / {threshold} XP до {character.level + 1} ур."

    hp_bar = hp_progress_bar(character.current_hp, character.max_hp)
    abilities = [
        f"{ABILITIES[code]} {character.abilities[code]:>2} "
        f"({format_modifier(character.ability_mod(code))})"
        for code in ABILITIES
    ]
    ability_rows = ["   ".join(abilities[index:index + 3]) for index in range(0, 6, 3)]

    if character.inventory:
        inventory = "\n".join(f" • {item}" for item in character.inventory)
        inventory_title = f"🎒 Снаряжение ({len(character.inventory)}):"
    else:
        inventory = " • (пусто)"
        inventory_title = "🎒 Снаряжение:"

    status = "жив" if character.is_alive else "без сознания"
    inspiration = "да" if character.heroic_inspiration else "нет"

    description_lines = (
        [f"🖋️ Описание: {character.description}"] if character.description else []
    )

    race_traits = species_traits(character.race)
    traits_lines: list[str] = (
        ["", "🧬 Черты (расовые)", *(f"• {trait}" for trait in race_traits)]
        if race_traits
        else []
    )

    magic_lines: list[str] = []
    if character.is_spellcaster:
        ability = ABILITY_FULL_RU.get(
            character.spellcasting_ability, character.spellcasting_ability.upper()
        )
        magic_lines = [
            "",
            f"🪄 Магия класса ({ability}): КС спасброска {character.spell_save_dc}, "
            f"атака заклинанием {format_modifier(character.spell_attack_bonus)}",
            *format_slots_tracker(character.spell_slots),
            f"🌟 Заговоры: {', '.join(character.cantrips) or 'нет'}",
            f"📚 Готово ({len(character.spells_prepared)}/{character.max_prepared}): "
            f"{', '.join(character.spells_prepared) or 'нет'}",
        ]

    return "\n".join(
        [
            "📜 ЛИСТ ПЕРСОНАЖА",
            "",
            f"📛 Имя: {character.name}",
            f"🧬 Раса: {character.race}",
            f"⚔️ Класс: {character.class_name} (кость хитов d{character.hit_die})",
            *description_lines,
            f"🎖️ Уровень: {character.level} "
            f"(бонус мастерства {format_modifier(character.proficiency_bonus)})",
            f"🧭 Инициатива {format_modifier(character.initiative)} | "
            f"КД {character.armor_class} ({character.armor_label}) | "
            f"Пассивная Внимательность {character.passive_perception}",
            "",
            f"❤️ HP [{hp_bar}] {character.current_hp}/{character.max_hp} ({status})",
            f"🌟 Вдохновение героя: {inspiration}",
            f"💰 Золото: {character.gp} gp",
            "",
            "📊 Прогресс",
            xp_line,
            "",
            "🧠 Характеристики",
            *ability_rows,
            *magic_lines,
            *traits_lines,
            "",
            inventory_title,
            inventory,
        ]
    )


def format_inventory(character: Character) -> str:
    """Компактный список снаряжения и золота для кнопки «🎒 Инвентарь»."""
    if character.inventory:
        title = f"🎒 СНАРЯЖЕНИЕ ({len(character.inventory)}):"
        items = [f" • {item}" for item in character.inventory]
    else:
        title = "🎒 СНАРЯЖЕНИЕ:"
        items = [" • (пусто)"]

    return "\n".join(
        [
            title,
            *items,
            "",
            f"💰 Золото: {character.gp} gp",
            f"⚔️ Класс: {character.class_name} | ❤️ HP: "
            f"{character.current_hp}/{character.max_hp}",
        ]
    )


def format_spell_card(character: Character) -> str:
    """Карточка «🪄 Книга заклинаний»: ячейки, заговоры и готовые заклинания."""
    lines = [
        SPELLS_MENU_TEXT,
        "",
        f"🔮 КС спасброска: {character.spell_save_dc} | "
        f"атака заклинанием {format_modifier(character.spell_attack_bonus)}",
        "",
        "📖 Ячейки",
        *format_slots_tracker(character.spell_slots),
        "",
        f"🌟 Заговоры ({len(character.cantrips)}): {', '.join(character.cantrips) or 'нет'}",
        f"📚 Готово ({len(character.spells_prepared)}/{character.max_prepared}): "
        f"{', '.join(character.spells_prepared) or 'нет'}",
    ]
    return "\n".join(lines)


def format_cast_menu(character: Character) -> str:
    """Карточка «🔥 Что применить»: список доступных заклинаний с кругами."""
    lines = [CAST_MENU_TEXT, ""]
    for name, circle in character.castable_spells():
        if circle == 0:
            mark = "🪄"
        elif _spell_circle_locked(character, circle):
            mark = "🔒"
        else:
            mark = "🔥"
        lines.append(f"{mark} {name} ({_spell_circle_label(circle)})")
    return "\n".join(lines)


def format_prep_menu(character: Character) -> str:
    """Карточка «⚡ Подготовка»: список изученных заклинаний с метками ✅/❌."""
    lines = [
        PREP_MENU_TEXT,
        "",
        f"Заготовлено {len(character.spells_prepared)} из {character.max_prepared}.",
        "",
    ]
    for name in character.spells_known:
        mark = "✅" if name in character.spells_prepared else "❌"
        lines.append(f"{mark} {name} ({_spell_circle_label(character.spell_circle(name))})")
    return "\n".join(lines)


def format_rest_menu(character: Character) -> str:
    """Карточка «🌙 Отдых»: что восстановит короткий и продолжительный отдых."""
    lines = [
        REST_MENU_TEXT,
        "",
        f"❤️ HP: {character.current_hp}/{character.max_hp} "
        f"[{hp_progress_bar(character.current_hp, character.max_hp)}]",
        "",
    ]
    if character.is_spellcaster:
        lines.append("📖 Ячейки сейчас")
        lines.extend(format_slots_tracker(character.spell_slots))
        if is_pact_caster(character.class_name):
            lines.append("☝️ Колдун: короткий отдых тоже восстанавливает магию пакта.")
        else:
            lines.append("☝️ На коротком отдыхе ячейки не восстанавливаются.")
    else:
        lines.append(REST_NOT_CASTERTEXT)
    return "\n".join(lines)


def format_party_status(party: "PartySession") -> str:
    """Краткий статус отряда: кто в группе, HP и уровень каждого героя."""
    confirmed = [ch for ch in party.characters.values() if ch.hero_confirmed]
    if not confirmed:
        return NOBODY_CONFIRMED_TEXT
    lines = [
        PARTY_STATUS_HEADER,
        format_hud(party.location, party.quest),
        "",
        f"Всего героев: {len(confirmed)}/{MAX_PARTY_SIZE}",
        "",
    ]
    for index, ch in enumerate(confirmed, start=1):
        status = "жив" if ch.is_alive else "без сознания"
        lines.append(
            f"{index}. {ch.name} — {ch.race} {ch.class_name}, {ch.level} ур. | "
            f"❤️ {ch.current_hp}/{ch.max_hp} ({status})"
        )
    return "\n".join(lines)


def character_label(character: Character) -> str:
    """Ярлык героя для реплик игрока в партии: «Имя (Вид Класс)»."""
    return f"{character.name} ({character.race} {character.class_name})"


def hero_summary(character: Character) -> str:
    """Однострочная выжимка о герое для служебных сообщений Мастеру."""
    if not character.is_created:
        missing = [
            title
            for title, value, default in (
                ("имя", character.name, DEFAULT_NAME),
                ("вид (раса)", character.race, DEFAULT_RACE),
                ("класс", character.class_name, DEFAULT_CLASS),
            )
            if value == default
        ]
        return "Лист персонажа ещё не заполнен: не заданы " + ", ".join(missing) + "."

    abilities = ", ".join(
        f"{ABILITIES[code]} {character.abilities[code]}" for code in ABILITIES
    )
    parts = [
        f"имя: {character.name}",
        f"вид (раса): {character.race}",
        f"класс: {character.class_name}",
        f"уровень: {character.level}",
        f"HP: {character.current_hp}/{character.max_hp}",
        f"КД: {character.armor_class}",
        f"характеристики: {abilities}",
    ]
    if character.description:
        parts.append(f"описание: {character.description}")
    if character.inventory:
        parts.append("снаряжение: " + ", ".join(character.inventory))
    if character.is_spellcaster:
        slots = ", ".join(
            f"{_spell_circle_label(int(circle))} {slot['current']}/{slot['total']}"
            for circle, slot in sorted(character.spell_slots.items(), key=lambda pair: int(pair[0]))
        )
        parts.append(
            f"магия: КС спасброска {character.spell_save_dc}, атака заклинанием "
            f"{format_modifier(character.spell_attack_bonus)}, ячейки: {slots or 'нет'}; "
            f"заговоры: {', '.join(character.cantrips) or 'нет'}; "
            f"готовые заклинания: {', '.join(character.spells_prepared) or 'нет'}"
        )
    return "Данные героя — " + "; ".join(parts) + "."


def build_party_sheet(party: "PartySession") -> str:
    """Сводный лист ВСЕЙ партии для системного контекста Мастера.

    Мастер ведёт общий учёт: ему нужно видеть всех героев сразу (кто ранен, чей ход,
    у кого какие ресурсы). Служебный JSON-блок при этом относится к действующему герою.
    """
    heroes = [ch for ch in party.characters.values() if ch.hero_confirmed]
    if not heroes:
        return ""
    lines = [
        "СОСТОЯНИЕ ПАРТИИ (общий лист отряда):",
        f"Локация отряда: {party.location}",
        f"Текущая цель: {party.quest}",
        f"Идёт бой: {'да' if party.in_combat else 'нет'}",
        "",
    ]
    for ch in heroes:
        lines.append(
            f"- {hero_summary(ch)}"
        )
    return "\n".join(lines)


def creation_stage(character: Character) -> str:
    """Фаза общения с игроком: создание героя, подтверждение героя или сама игра."""
    if not character.is_created:
        return CREATION_STAGE_HERO
    if not character.hero_confirmed:
        return CREATION_STAGE_CONFIRM
    return CREATION_STAGE_PLAY


def is_random_hero_request(text: str) -> bool:
    """Просит ли игрок сгенерировать героя вместо того, чтобы описывать своего."""
    lowered = text.strip().lower()
    if lowered.startswith(("/hero", "/randomhero")):
        return True
    return any(marker in lowered for marker in RANDOM_HERO_MARKERS)


def is_hero_confirmation(text: str) -> bool:
    """Короткое согласие игрока с созданным героем («да», «подтверждаю», «начинаем»)."""
    if len(text) > 40:
        return False
    return bool(HERO_CONFIRM_PATTERN.match(text.strip()))


def parse_join_deep_link(payload: Optional[str]) -> Optional[int]:
    """Разбирает deep-link аргумент команды /start вида «join_-1001234567890».

    Возвращает chat_id целевой группы (обычно отрицательный) или None, если аргумент
    не похож на «join_<chat_id>».
    """
    if not payload:
        return None
    match = JOIN_DEEP_LINK_PATTERN.match(payload.strip())
    if match is None:
        return None
    try:
        chat_id = int(match.group(1))
    except ValueError:
        return None
    if chat_id == 0:
        return None
    return chat_id


def creation_prompt(*, first_game_message: bool) -> str:
    """Инструкция Мастеру в фазе создания: первое сообщение игры или продолжение."""
    return HERO_CREATION_START_PROMPT if first_game_message else HERO_CREATION_PROMPT


def hero_confirmation_prompt(character: Character) -> str:
    """Инструкция Мастеру: герой создан, но игрок его ещё не подтвердил."""
    return (
        "[СИСТЕМА] Этап создания персонажа: герой записан в лист, но игрок его ещё НЕ подтвердил. "
        "Приключение и пролог начинать ЗАПРЕЩЕНО.\n"
        f"{hero_summary(character)}\n"
        "Задание: коротко (1–2 абзаца) отреагируй на сообщение игрока. Просит изменить героя — "
        "исправь нужные поля (\"name\", \"race\", \"class_name\", \"description\") в служебном "
        "JSON-блоке; описывает действия вместо героя — вежливо напомни, что сначала нужно "
        "подтвердить героя. В конце спроси, всё ли верно с героем: подтвердить можно словом «да» "
        "или кнопкой «✅ Подтвердить героя». Пролог не начинай."
    )


def prologue_prompt(character: Character, party_members: list[str]) -> str:
    """Инструкция Мастеру начать вводную сцену после подтверждения героя.

    :param party_members: имена уже подтверждённых героев отряда (без текущего), чтобы
        Мастер мог сразу собрать группу в одной сцене.
    """
    others = ""
    if party_members:
        others = (
            " В отряде уже есть герои: " + ", ".join(party_members) +
            ". Начни пролог так, чтобы пути героев естественно сходились в одну группу."
        )
    return (
        "[СИСТЕМА] Игрок подтвердил героя. Этап создания персонажа завершён — начинается игра.\n"
        f"{hero_summary(character)}\n"
        "Задание: начни вводную сцену пролога по правилам D&D 2024: атмосферно опиши, где и как "
        "начинается путь героя, придумай название мира и короткую завязку в духе тёмного "
        f"героического фэнтези, дай одну-две зацепки и остановись в точке выбора.{others} "
        f"Назови героя по имени («{character.name}, твоя история начинается…»). "
        "НЕ описывай действия, слова и мысли героя игрока. Закончи вопросом «Что ты делаешь?»"
    )


def join_scene_prompt(character: Character, party_members: list[str], location: str) -> str:
    """Инструкция Мастеру ввести нового героя в УЖЕ идущее приключение («Вариант Б»).

    Используется, когда герой подтверждён в личных сообщениях, а кампания в группе уже
    начата: Мастер не запускает пролог заново, а органично вплетает новичка в текущую сцену.
    """
    others = ", ".join(party_members) if party_members else "остальные герои отряда"
    return (
        "[СИСТЕМА] В уже идущее приключение вступает новый герой игрока — этап создания "
        "персонажа завершён.\n"
        f"{hero_summary(character)}\n"
        f"Отряд сейчас: {others}. Текущая локация отряда: {location}.\n"
        "Задание: органично введи нового героя в ТЕКУЩУЮ сцену (1–3 абзаца): где и как отряд "
        "встречает его, почему он присоединяется к группе. НЕ начинай приключение заново и не "
        f"устраивай новый пролог. Назови героя по имени («{character.name} присоединяется…») и "
        "закончи вопросом «Что ты делаешь?»"
    )


def new_hero_announcement(character: Character, mention: str) -> str:
    """Торжественное объявление в группе о прибытии нового героя отряда."""
    return NEW_HERO_ANNOUNCEMENT.format(
        name=character.name,
        race=character.race,
        class_name=character.class_name,
        level=character.level,
        mention=mention or NEW_HERO_ANNOUNCEMENT_EMPTY_MENTION,
    )


def random_hero_prompt(character: Character) -> str:
    """Инструкция Мастеру представить уже сгенерированного кодом случайного героя."""
    return (
        "[СИСТЕМА] Игрок выбрал случайного героя: система уже сгенерировала его строго по "
        "правилам PHB 2024 (характеристики, HP, стартовое снаряжение и золото) и записала в лист "
        "персонажа.\n"
        f"{hero_summary(character)}\n"
        "Задание: представь игроку этого героя (1–2 абзаца) — имя, вид, класс, внешность и "
        "характер. Приключение НЕ начинай и пролог не описывай: попроси игрока подтвердить героя "
        "(«да» или кнопка «✅ Подтвердить героя»)."
    )


def detect_hero_details(text: str) -> dict[str, str]:
    """Достаёт из сообщения игрока имя, вид и класс героя (страховка для служебного блока)."""
    details: dict[str, str] = {}

    name_match = HERO_NAME_PATTERN.search(text)
    if name_match is not None:
        raw_name = name_match.group("name").strip()
        details["name"] = raw_name[:1].upper() + raw_name[1:]

    species = species_name(text)
    if species is not None:
        details["race"] = species

    class_name = class_name_from_text(text)
    if class_name is not None:
        details["class_name"] = class_name

    return details


def auto_fill_hero_details(character: Character, text: str) -> list[str]:
    """Страховка на случай, если Мастер забудет служебный блок.

    Дополняются ТОЛЬКО пустые поля листа, причём вид и класс — лишь когда игрок назвал
    оба (случайное упоминание «мага» в рассказе не должно сделать героя волшебником).
    """
    if character.is_created:
        return []

    details = detect_hero_details(text)
    notes: list[str] = []

    if "name" in details and character.name == DEFAULT_NAME:
        character.name = details["name"][:64]
        notes.append(f"📛 Имя персонажа: {character.name}.")

    if "race" in details and "class_name" in details:
        if character.race == DEFAULT_RACE:
            character.race = details["race"][:64]
            notes.append(f"🧬 Раса: {character.race}.")
        if character.class_name == DEFAULT_CLASS:
            character.class_name = details["class_name"][:64]
            notes.append(f"⚔️ Класс: {character.class_name} (кость хитов d{character.hit_die}).")

    return notes


def _roll_ability_score() -> int:
    """Характеристика методом «4d6 без наименьшего кубика» (официальный метод PHB 2024)."""
    dice = sorted((_rng.randint(1, 6) for _ in range(4)), reverse=True)
    return sum(dice[:3])


def build_random_hero() -> Character:
    """Собирает случайного героя 1-го уровня строго по правилам PHB 2024 — без участия модели.

    Имя и описание берутся из заготовок, вид и класс — из официальных списков, характеристики
    бросаются методом 4d6 без наименьшего кубика, снаряжение и золото — стартовый набор.
    """
    name, description = _rng.choice(RANDOM_HERO_PROFILES)
    species = _rng.choice(tuple(SPECIES.values()))["name"]
    class_key = _rng.choice(tuple(CLASSES))
    class_name = CLASSES[class_key]["name"]

    scores = sorted((_roll_ability_score() for _ in ABILITIES), reverse=True)
    abilities = dict(zip(ability_priority_for_class(class_name), scores))

    return Character(
        name=name,
        race=species,
        class_name=class_name,
        level=1,
        abilities=abilities,
        inventory=list(starter_equipment_for_class(class_name)),
        gp=starting_gold_for_class(class_name),
        description=description,
    )


def apply_starter_loadout(character: Character, recalc_hp: bool = False) -> list[str]:
    """Доводит только что созданного героя до правил PHB 2024 и возвращает заметки для игрока.

    :param recalc_hp: пересчитать максимум HP (кость хитов + модификатор ТЕЛ), когда Мастер
        сам не задавал HP в служебном блоке.
    """
    notes: list[str] = []

    if all(character.abilities[code] == DEFAULT_ABILITIES[code] for code in ABILITIES):
        character.abilities = standard_array_for_class(character.class_name)
        spread = ", ".join(
            f"{ABILITIES[code]} {character.abilities[code]}" for code in ABILITIES
        )
        notes.append(f"🧠 Характеристики по стандартному набору PHB 2024: {spread}.")

    if not character.inventory:
        character.inventory = list(starter_equipment_for_class(character.class_name))
        notes.append(
            f"🎒 Стартовое снаряжение 1-го уровня: {', '.join(character.inventory)}."
        )
        if character.gp == 0:
            character.gp = starting_gold_for_class(character.class_name)
            notes.append(f"💰 Стартовое золото: {character.gp} gp.")

    if recalc_hp:
        new_max_hp = max(1, character.hit_die + character.ability_mod("con"))
        if new_max_hp != character.max_hp:
            character.max_hp = new_max_hp
            character.current_hp = new_max_hp
            notes.append(
                f"❤️ Здоровье 1-го уровня: d{character.hit_die} + модификатор ТЕЛ = "
                f"{character.max_hp} HP."
            )

    if character.is_spellcaster:
        character.init_default_spellcasting()
        notes.append(
            f"🪄 Магия класса готова: КС спасброска {character.spell_save_dc}, "
            f"заговоры: {', '.join(character.cantrips) or 'нет'}; "
            f"заготовлено: {', '.join(character.spells_prepared) or 'нет'}."
        )

    return notes


def find_inventory_weapon(character: Character) -> Optional[tuple[str, str, str, str]]:
    """Ищет в снаряжении игрока оружие из официального справочника PHB 2024.

    :return: (название, кость урона, тип урона, способ нанесения) или None.
        Способ нанесения — одна из констант DAMAGE_ABILITY_*.
    """
    for item in character.inventory:
        lowered = item.strip().lower()
        if not lowered:
            continue
        for name, damage, damage_type, properties, _mastery in WEAPONS:
            if name.lower() in lowered:
                props = properties.lower()
                if "боеприпас" in props:
                    kind = DAMAGE_ABILITY_RANGED
                elif "фехтовальное" in props:
                    kind = DAMAGE_ABILITY_FINESSE
                else:
                    kind = DAMAGE_ABILITY_MELEE
                return name, damage, damage_type, kind
    return None


def roll_weapon_damage(character: Character) -> tuple[DiceRoll, str]:
    """Бросает урон оружием из снаряжения игрока (с модификатором характеристики).

    Если оружия нет — считает импровизированную атаку (1d4 + СИЛ).
    """
    weapon = find_inventory_weapon(character)

    if weapon is None:
        damage_dice = IMPROVISED_DAMAGE_DICE
        damage_type = IMPROVISED_DAMAGE_TYPE
        ability_code = "str"
        label = f"импровизированная атака без оружия ({damage_dice})"
    else:
        name, damage_dice, damage_type, kind = weapon
        if kind == DAMAGE_ABILITY_RANGED:
            ability_code = "dex"
        elif kind == DAMAGE_ABILITY_FINESSE:
            ability_code = (
                "dex" if character.ability_mod("dex") > character.ability_mod("str") else "str"
            )
        else:
            ability_code = "str"
        label = f"{name} ({damage_dice}, {damage_type} урон)"

    modifier = character.ability_mod(ability_code)

    dice_match = DICE_PATTERN.match(damage_dice)
    if dice_match is None:
        # Фиксированный урон (например, «1» у духовой трубки) — кубики не бросаем.
        roll = DiceRoll.flat(_as_int(damage_dice) + modifier)
    else:
        roll = make_roll(
            count=int(dice_match.group("count") or 1),
            sides=int(dice_match.group("sides")),
            modifier=modifier,
        )

    return roll, f"{label}, модификатор {ABILITIES[ability_code]} {format_modifier(modifier)}"


def weapon_ability_code(character: Character) -> str:
    """Код характеристики, модификатор которой идёт в урон текущим оружием.

    Дальнобойное оружие — ЛОВ, фехтовальное — лучшая из СИЛ/ЛОВ, остальное — СИЛ.
    Если оружия в снаряжении нет — импровизированная атака (СИЛ).
    """
    weapon = find_inventory_weapon(character)
    if weapon is None:
        return "str"

    _name, _dice, _type, kind = weapon
    if kind == DAMAGE_ABILITY_RANGED:
        return "dex"
    if kind == DAMAGE_ABILITY_FINESSE:
        return "dex" if character.ability_mod("dex") > character.ability_mod("str") else "str"
    return "str"


def roll_damage_die(character: Character, sides: int) -> tuple[DiceRoll, str]:
    """Бросает 1d<sides> как урон оружием игрока (с модификатором характеристики).

    Нужен кнопкам урона d4/d6/d8/d10/d12: кость выбирает игрок по просьбе Мастера,
    а модификатор характеристики берётся из текущего оружия в листе персонажа.
    """
    sides = max(2, min(_as_int(sides), MAX_DICE_SIDES))
    ability_code = weapon_ability_code(character)
    modifier = character.ability_mod(ability_code)
    roll = make_roll(count=1, sides=sides, modifier=modifier)

    weapon = find_inventory_weapon(character)
    if weapon is None:
        weapon_label = "без оружия (импровизированная атака)"
    else:
        weapon_label = f"{weapon[0]} ({weapon[1]}, {weapon[2]} урон)"
    label = f"{weapon_label}, модификатор {ABILITIES[ability_code]} {format_modifier(modifier)}"
    return roll, label


def _damage_die_sides(damage_dice: str) -> Optional[int]:
    """Число граней единственной кости урона («1d8» -> 8) или None."""
    match = DICE_PATTERN.match(damage_dice.strip())
    if match is None or int(match.group("count") or 1) != 1:
        return None
    return int(match.group("sides"))


# ---------------------------------------------------------------------------
# 10. РАЗБОР СЛУЖЕБНОГО JSON-БЛОКА МАСТЕРА
# ---------------------------------------------------------------------------

CONTROL_BLOCK_KEYS = frozenset(
    {
        "xp_gained", "hp_change", "add_items", "remove_items", "gp_change",
        "name", "race", "class_name", "level", "abilities", "description",
        "max_hp", "current_hp", "heroic_inspiration",
        "location", "quest", "in_combat", "battle_summary",
    }
)

# Синонимы ключей, которыми модель иногда называет поля: приводим их к каноническим именам.
CONTROL_KEY_ALIASES: dict[str, str] = {
    "xp": "xp_gained",
    "exp": "xp_gained",
    "experience": "xp_gained",
    "опыт": "xp_gained",
    "опыта": "xp_gained",
    "hp": "hp_change",
    "health": "hp_change",
    "hp_delta": "hp_change",
    "damage": "hp_change",
    "dmg": "hp_change",
    "урон": "hp_change",
    "хп": "hp_change",
    "gold": "gp_change",
    "gp": "gp_change",
    "золото": "gp_change",
    "items": "add_items",
    "add_item": "add_items",
    "предметы": "add_items",
    "remove_item": "remove_items",
    "class": "class_name",
    "species": "race",
    "локация": "location",
    "цель": "quest",
    "combat": "in_combat",
    "in_battle": "in_combat",
    "combat_active": "in_combat",
    "бой": "in_combat",
    "battle": "in_combat",
    "battle_result": "battle_summary",
    "battle_log": "battle_summary",
    "итог_битвы": "battle_summary",
}


def _normalize_control_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Приводит ключи служебного блока к каноническим (терпимо к синонимам модели)."""
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        canonical = key.strip()
        if canonical not in CONTROL_BLOCK_KEYS:
            canonical = CONTROL_KEY_ALIASES.get(canonical.lower(), canonical)
        if canonical not in normalized:
            normalized[canonical] = value
    return normalized


class ControlBlock(BaseModel):
    """Строгая схема служебного JSON-блока Мастера.

    Неизвестные ключи отбрасываются (``extra="ignore"``), а известные приводятся
    к ожидаемому типу. Отсутствующие поля остаются ``None`` и в словарь не попадают.
    """

    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    race: Optional[str] = None
    class_name: Optional[str] = None
    description: Optional[str] = None
    level: Optional[int] = None
    abilities: Optional[dict[str, int]] = None
    max_hp: Optional[int] = None
    current_hp: Optional[int] = None
    heroic_inspiration: Optional[bool] = None
    location: Optional[str] = None
    quest: Optional[str] = None

    xp_gained: Optional[int] = None
    hp_change: Optional[int] = None
    add_items: Optional[list[str]] = None
    remove_items: Optional[list[str]] = None
    gp_change: Optional[int] = None

    in_combat: Optional[bool] = None
    battle_summary: Optional[str] = None


def validate_control_block(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Приводит служебный блок Мастера к схеме :class:`ControlBlock`.

    Возвращает нормализованный словарь без пустых полей. Если блок не проходит
    проверку целиком, возвращаем его как есть (apply_control сам защищается от «мусора»).
    """
    try:
        model = ControlBlock.model_validate(data)
    except ValidationError as error:
        logger.warning("Служебный блок Мастера не прошёл валидацию, беру как есть: %s", error)
        return data

    cleaned = model.model_dump(exclude_none=True)
    return cleaned or None


# Служебный блок в тройных обратных кавычках (``` или ```json).
FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _iter_balanced_objects(text: str):
    """Итератор по сбалансированным подстрокам {...} (с учётом строк JSON)."""
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield start, index + 1, text[start:index + 1]
                start = None


def _try_parse_json(raw: str) -> Optional[dict]:
    """Пытается разобрать JSON-объект; при неудаче возвращает None."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Удаляет из текста указанные диапазоны (с объединением пересечений)."""
    if not spans:
        return text

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def extract_control_block(text: str) -> tuple[str, Optional[dict]]:
    """Вырезает служебный JSON-блок из ответа Мастера.

    Возвращает пару (текст без блока, данные блока или None).
    """
    control: Optional[dict] = None
    spans: list[tuple[int, int]] = []

    # 1) Блоки внутри тройных обратных кавычек (вырезаем всегда, применяем — если валиден).
    for match in FENCED_JSON_PATTERN.finditer(text):
        data = _try_parse_json(match.group(1))
        if data is not None:
            control = _normalize_control_keys(data)
        spans.append(match.span())

    # 2) «Голые» JSON-объекты (если модель забыла про обратные кавычки).
    for start, end, raw in _iter_balanced_objects(text):
        if any(begin <= start and end <= finish for begin, finish in spans):
            continue
        data = _try_parse_json(raw)
        if data is None:
            continue
        data = _normalize_control_keys(data)
        if CONTROL_BLOCK_KEYS.intersection(data.keys()):
            control = data
            spans.append((start, end))

    clean = _strip_spans(text, spans)
    # Убираем «осиротевшие» пустые блоки кода и лишние пустые строки.
    clean = re.sub(r"```(?:json)?\s*```", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    if control is not None:
        control = validate_control_block(control)

    return clean.strip(), control


# ---------------------------------------------------------------------------
# 11. БАЗА ДАННЫХ SQLITE (партии, герои партии и общая история чата)
# ---------------------------------------------------------------------------

# Схема: одна партия на chat_id. Герои ключом (chat_id, user_id); история — по chat_id.
DB_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS campaigns (
    chat_id    INTEGER PRIMARY KEY,
    location   TEXT    NOT NULL DEFAULT '{DEFAULT_LOCATION}',
    quest      TEXT    NOT NULL DEFAULT '{DEFAULT_QUEST}',
    in_combat  INTEGER NOT NULL DEFAULT 0,
    started    INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS characters (
    chat_id        INTEGER NOT NULL,
    user_id        INTEGER NOT NULL,
    data           TEXT    NOT NULL,
    name           TEXT    NOT NULL DEFAULT '{DEFAULT_NAME}',
    race           TEXT    NOT NULL DEFAULT '{DEFAULT_RACE}',
    class_name     TEXT    NOT NULL DEFAULT '{DEFAULT_CLASS}',
    level          INTEGER NOT NULL DEFAULT 1,
    current_hp     INTEGER NOT NULL DEFAULT 0,
    max_hp         INTEGER NOT NULL DEFAULT 0,
    hero_confirmed INTEGER NOT NULL DEFAULT 0,
    updated_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chat_id, user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    user_id    INTEGER,
    char_name  TEXT    NOT NULL DEFAULT '',
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages (chat_id, id);
CREATE INDEX IF NOT EXISTS idx_characters_chat ON characters (chat_id, user_id);
"""


@dataclass
class Campaign:
    """Общее состояние партии (сводка HUD, бой и признак начатой игры)."""

    chat_id: int
    location: str = DEFAULT_LOCATION
    quest: str = DEFAULT_QUEST
    in_combat: bool = False
    started: bool = False


class PartyDatabase:
    """Постоянное хранилище партий в локальном файле SQLite (party_database.db).

    Таблицы:
        * campaigns  — по одной записи на чат: локация, цель, бой, «игра началась»;
        * characters — лист каждого героя (JSON) + дублирующие колонки для чтения/отладки;
        * messages   — общая история диалога чата (роли user / assistant).

    Соединение открывается лениво, переиспользуется и защищено блокировкой.
    """

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        """Открывает соединение (один раз) и применяет схему. Вызывать под self._lock."""
        if self._conn is not None:
            return self._conn

        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(DB_SCHEMA)
        conn.commit()
        self._conn = conn
        return conn

    def init(self) -> None:
        """Создаёт файл базы, таблицы и индексы (идемпотентно)."""
        with self._lock:
            self._connect()
        logger.info("База данных готова: %s", self.path)

    def close(self) -> None:
        """Закрывает соединение с базой (вызывается при остановке бота)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # --- таблица campaigns ---

    def save_campaign(self, campaign: Campaign) -> None:
        """Сохраняет (или обновляет) общее состояние партии чата."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO campaigns (chat_id, location, quest, in_combat, started, updated_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(chat_id) DO UPDATE SET "
                "location = excluded.location, quest = excluded.quest, "
                "in_combat = excluded.in_combat, started = excluded.started, "
                "updated_at = CURRENT_TIMESTAMP",
                (
                    int(campaign.chat_id),
                    campaign.location,
                    campaign.quest,
                    1 if campaign.in_combat else 0,
                    1 if campaign.started else 0,
                ),
            )
            conn.commit()

    def load_campaign(self, chat_id: int) -> Optional[Campaign]:
        """Возвращает сохранённую партию чата или None, если записи ещё нет."""
        with self._lock:
            row = self._connect().execute(
                "SELECT location, quest, in_combat, started FROM campaigns WHERE chat_id = ?",
                (int(chat_id),),
            ).fetchone()
        if row is None:
            return None
        return Campaign(
            chat_id=int(chat_id),
            location=(row["location"] or "").strip()[:120] or DEFAULT_LOCATION,
            quest=(row["quest"] or "").strip()[:120] or DEFAULT_QUEST,
            in_combat=bool(row["in_combat"]),
            started=bool(row["started"]),
        )

    # --- таблица characters ---

    def save_character(self, chat_id: int, user_id: int, character: Character) -> None:
        """Сохраняет (или обновляет) лист героя игрока в партии чата.

        Весь лист лежит в JSON-колонке data, а ключевые поля (имя, вид, класс, уровень,
        HP, подтверждение) дублируются в колонки для удобного чтения SQL-запросами.
        """
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO characters "
                "(chat_id, user_id, data, name, race, class_name, level, current_hp, max_hp, "
                " hero_confirmed, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(chat_id, user_id) DO UPDATE SET "
                "data = excluded.data, name = excluded.name, race = excluded.race, "
                "class_name = excluded.class_name, level = excluded.level, "
                "current_hp = excluded.current_hp, max_hp = excluded.max_hp, "
                "hero_confirmed = excluded.hero_confirmed, updated_at = CURRENT_TIMESTAMP",
                (
                    int(chat_id),
                    int(user_id),
                    character.to_json(),
                    character.name,
                    character.race,
                    character.class_name,
                    character.level,
                    character.current_hp,
                    character.max_hp,
                    1 if character.hero_confirmed else 0,
                ),
            )
            conn.commit()

    def load_character(self, chat_id: int, user_id: int) -> Optional[Character]:
        """Возвращает сохранённый лист героя или None, если игрок ещё не в отряде."""
        with self._lock:
            row = self._connect().execute(
                "SELECT data FROM characters WHERE chat_id = ? AND user_id = ?",
                (int(chat_id), int(user_id)),
            ).fetchone()
        if row is None:
            return None
        return Character.from_json(row["data"])

    def load_party(self, chat_id: int) -> dict[int, Character]:
        """Возвращает всех героев партии чата: {user_id: Character}."""
        with self._lock:
            rows = self._connect().execute(
                "SELECT user_id, data FROM characters WHERE chat_id = ? ORDER BY user_id",
                (int(chat_id),),
            ).fetchall()
        party: dict[int, Character] = {}
        for row in rows:
            party[int(row["user_id"])] = Character.from_json(row["data"])
        return party

    def delete_character(self, chat_id: int, user_id: int) -> None:
        """Удаляет лист героя игрока (выход игрока из отряда или сброс партии)."""
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute(
                    "DELETE FROM characters WHERE chat_id = ? AND user_id = ?",
                    (int(chat_id), int(user_id)),
                )

    # --- таблица messages ---

    def append_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        *,
        user_id: Optional[int] = None,
        char_name: str = "",
    ) -> int:
        """Добавляет одно сообщение (user/assistant) в общую историю чата.

        Возвращает id вставленной строки: он нужен, чтобы при свёртке боя удалить именно
        боевые сообщения и заменить их кратким резюме (см. PartySession.end_combat).
        """
        with self._lock:
            conn = self._connect()
            cursor = conn.execute(
                "INSERT INTO messages (chat_id, user_id, char_name, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    int(chat_id),
                    int(user_id) if user_id is not None else None,
                    str(char_name or ""),
                    str(role),
                    str(content),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def load_history(self, chat_id: int, limit: int = MAX_HISTORY_MESSAGES) -> list[dict[str, Any]]:
        """Возвращает последние `limit` сообщений чата в хронологическом порядке."""
        with self._lock:
            rows = self._connect().execute(
                "SELECT user_id, char_name, role, content FROM messages "
                "WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (int(chat_id), int(limit)),
            ).fetchall()
        # Запрос шёл от свежих к старым — разворачиваем в хронологию.
        return [
            {
                "user_id": row["user_id"],
                "char_name": row["char_name"],
                "role": row["role"],
                "content": row["content"],
            }
            for row in reversed(rows)
        ]

    def last_message_id(self, chat_id: int) -> Optional[int]:
        """Возвращает id последнего сообщения чата (или None, если история пуста)."""
        with self._lock:
            row = self._connect().execute(
                "SELECT MAX(id) AS last_id FROM messages WHERE chat_id = ?",
                (int(chat_id),),
            ).fetchone()
        if row is None or row["last_id"] is None:
            return None
        return int(row["last_id"])

    def delete_messages_after(self, chat_id: int, first_id: int) -> int:
        """Удаляет сообщения чата начиная с id >= first_id. Возвращает число удалённых."""
        with self._lock:
            conn = self._connect()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE chat_id = ? AND id >= ?",
                    (int(chat_id), int(first_id)),
                )
        return int(cursor.rowcount or 0)

    def reset_party(self, chat_id: int) -> None:
        """Полностью очищает партию чата: сбрасывает кампанию, героев и историю."""
        with self._lock:
            conn = self._connect()
            with conn:  # атомарно: либо удалилось всё, либо ничего
                conn.execute("DELETE FROM messages WHERE chat_id = ?", (int(chat_id),))
                conn.execute("DELETE FROM characters WHERE chat_id = ?", (int(chat_id),))
                conn.execute("DELETE FROM campaigns WHERE chat_id = ?", (int(chat_id),))


# Единственный на процесс экземпляр базы (файл party_database.db в корне проекта).
db = PartyDatabase()


# ---------------------------------------------------------------------------
# 12. ПАРТИЯ И ПАМЯТЬ ДИАЛОГА (по одной партии на chat_id)
# ---------------------------------------------------------------------------


class PartySession:
    """Партия одного чата: герои всех игроков, общая история и общее состояние.

    В памяти держим всю историю, но в модель вне боя уходит скользящее окно последних
    MAX_HISTORY_MESSAGES сообщений. Во время боя окно НЕ обрезается, а после боя подробные
    раунды сворачиваются в краткое резюме (см. begin_combat/end_combat). Изменения сразу
    дублируются в SQLite, поэтому прогресс поднимается из базы после перезапуска.
    """

    __slots__ = (
        "chat_id",
        "characters",
        "location",
        "quest",
        "in_combat",
        "started",
        "history",
        "combat_start",
        "combat_first_db_id",
        "_last_db_id",
    )

    def __init__(self, chat_id: int) -> None:
        self.chat_id: int = int(chat_id)
        # user_id -> Character: свой герой у каждого игрока отряда.
        self.characters: dict[int, Character] = {}
        self.location: str = DEFAULT_LOCATION
        self.quest: str = DEFAULT_QUEST
        self.in_combat: bool = False
        # True, когда пролог уже начат (первый герой подтверждён).
        self.started: bool = False
        self.history: list[dict[str, Any]] = []
        self.combat_start: Optional[int] = None
        self.combat_first_db_id: Optional[int] = None
        self._last_db_id: Optional[int] = None

    @classmethod
    def restore(cls, chat_id: int) -> "PartySession":
        """Поднимает партию из базы: состояние кампании, героев и последние сообщения."""
        session = cls(chat_id)
        campaign = db.load_campaign(chat_id)
        if campaign is not None:
            session.location = campaign.location
            session.quest = campaign.quest
            session.in_combat = campaign.in_combat
            session.started = campaign.started
        session.characters = db.load_party(chat_id)
        for message in db.load_history(chat_id):
            session.history.append(message)
        session._last_db_id = db.last_message_id(chat_id)
        return session

    # --- герои партии ---

    def character_for(self, user_id: int) -> Optional[Character]:
        """Возвращает героя игрока или None, если он ещё не в отряде."""
        return self.characters.get(int(user_id))

    def ensure_character(self, user_id: int) -> Character:
        """Возвращает героя игрока, создавая пустой лист, если он ещё не в отряде."""
        user_id = int(user_id)
        character = self.characters.get(user_id)
        if character is None:
            character = Character()
            self.characters[user_id] = character
            db.save_character(self.chat_id, user_id, character)
        return character

    def confirmed_heroes(self) -> dict[int, Character]:
        """Только подтверждённые герои (участники текущего приключения)."""
        return {uid: ch for uid, ch in self.characters.items() if ch.hero_confirmed}

    def is_full(self) -> bool:
        """Достигнут ли предел размера отряда."""
        return len(self.characters) >= MAX_PARTY_SIZE

    def save_character(self, user_id: int) -> None:
        """Сохраняет лист конкретного героя в базу."""
        character = self.characters.get(int(user_id))
        if character is not None:
            db.save_character(self.chat_id, int(user_id), character)

    def save_campaign(self) -> None:
        """Сохраняет общее состояние партии (локация, цель, бой, started) в базу."""
        db.save_campaign(
            Campaign(
                chat_id=self.chat_id,
                location=self.location,
                quest=self.quest,
                in_combat=self.in_combat,
                started=self.started,
            )
        )

    # --- история диалога ---

    def add(
        self,
        role: str,
        content: str,
        *,
        user_id: Optional[int] = None,
        char_name: str = "",
    ) -> None:
        """Добавляет сообщение в общую историю партии (память + база)."""
        entry = {
            "role": role,
            "content": content,
            "user_id": int(user_id) if user_id is not None else None,
            "char_name": char_name or "",
        }
        self.history.append(entry)
        self._last_db_id = db.append_message(
            self.chat_id, role, content, user_id=user_id, char_name=char_name
        )

    def _format_history_entry(self, entry: dict[str, Any]) -> dict[str, str]:
        """Готовит запись истории к отправке в API (ярлык игрока для реплик партии)."""
        content = entry.get("content", "")
        if entry.get("role") == "user" and entry.get("char_name"):
            content = f"[Игрок {entry['char_name']}]: {content}"
        return {"role": entry.get("role", "user"), "content": content}

    def messages(self) -> list[dict[str, str]]:
        """Снимок истории для API (без системного промпта).

        Вне боя — скользящее окно последних MAX_HISTORY_MESSAGES сообщений. В бою окно
        НЕ обрезается: к последним мирным сообщениям добавляется вся боевая цепочка целиком.
        """
        if self.in_combat:
            start = self.combat_start if self.combat_start is not None else len(self.history)
            start = max(0, min(start, len(self.history)))
            head = self.history[:start]
            if len(head) > MAX_HISTORY_MESSAGES:
                head = head[-MAX_HISTORY_MESSAGES:]
            battle = self.history[start:]
            if len(battle) > COMBAT_HISTORY_SAFETY_LIMIT:
                battle = battle[-COMBAT_HISTORY_SAFETY_LIMIT:]
            window = [*head, *battle]
        elif len(self.history) > MAX_HISTORY_MESSAGES:
            window = list(self.history[-MAX_HISTORY_MESSAGES:])
        else:
            window = list(self.history)
        return [self._format_history_entry(entry) for entry in window]

    def begin_combat(self) -> None:
        """Помечает начало боя: с этого момента messages() не обрезает историю."""
        self.in_combat = True
        self.combat_start = len(self.history) - 1 if self.history else 0
        self.combat_first_db_id = self._last_db_id

    def end_combat(self, summary: str) -> None:
        """Завершает бой: сворачивает подробную боевую цепочку в одну строку-резюме."""
        start = self.combat_start if self.combat_start is not None else len(self.history)
        start = max(0, min(start, len(self.history)))
        note = {
            "role": "user",
            "content": f"[СИСТЕМА] {BATTLE_SUMMARY_PREFIX} {summary.strip() or 'сражение завершено.'}",
            "user_id": None,
            "char_name": "",
        }
        self.history = self.history[:start] + [note]
        self.in_combat = False
        self.combat_start = None
        if self.combat_first_db_id:
            db.delete_messages_after(self.chat_id, self.combat_first_db_id)
        self._last_db_id = db.append_message(self.chat_id, note["role"], note["content"])
        self.combat_first_db_id = None

    def reset(self) -> None:
        """Полностью очищает партию чата (память + база)."""
        self.characters.clear()
        self.location = DEFAULT_LOCATION
        self.quest = DEFAULT_QUEST
        self.in_combat = False
        self.started = False
        self.history.clear()
        self.combat_start = None
        self.combat_first_db_id = None
        self._last_db_id = None
        db.reset_party(self.chat_id)


# chat_id -> PartySession
_parties: dict[int, PartySession] = {}

# «Вариант Б»: user_id -> chat_id группы, для которой игрок создаёт героя в личных сообщениях.
# Заполняется при переходе по deep-link «/start join_<chat_id>» и очищается после доставки героя.
_join_targets: dict[int, int] = {}

# user_id -> chat_id группы, куда герой уже доставлен (нужно для вежливых подсказок в ЛС).
_delivered: dict[int, int] = {}


def get_party(chat_id: int) -> PartySession:
    """Возвращает партию чата, при первом обращении поднимая её из базы."""
    party = _parties.get(chat_id)
    if party is None:
        party = PartySession.restore(chat_id)
        _parties[chat_id] = party
        logger.info(
            "Партия чата %s загружена из базы (героев: %d, сообщений: %d)",
            chat_id,
            len(party.characters),
            len(party.history),
        )
    return party


# ---------------------------------------------------------------------------
# 13. КЛИЕНТ LLM (официальный API DeepSeek, OpenAI-совместимый эндпоинт)
# ---------------------------------------------------------------------------

llm_client: Optional[AsyncOpenAI] = (
    AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=0,
    )
    if LLM_API_KEY
    else None
)

# Ошибки, которые имеет смысл повторить: таймауты, обрывы связи, лимит (429) и 5xx.
_RETRYABLE_LLM_ERRORS = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


# Допустимые роли сообщений в API DeepSeek (OpenAI-совместимый формат).
_LLM_ROLES = ("system", "user", "assistant")


def _coerce_content(content: Any) -> str:
    """Приводит content сообщения к простой строке.

    DeepSeek ждёт, что content каждого сообщения — обычная строка (str). Если туда
    случайно попадёт список/кортеж строк (например, из-за опечатки в константе-промпте,
    превратившей строку в кортеж), API отвечает 422 «invalid type: string …,
    expected …ContentBlock». Здесь любой вход безопасно склеивается в строку.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)


def _normalize_messages(messages: Iterable[Any]) -> list[dict[str, str]]:
    """Гарантирует валидный для DeepSeek список сообщений.

    Каждый элемент приводится к виду {"role": system|user|assistant, "content": str}:
    нестандартные роли заменяются (первое сообщение — «system», остальные — «user»),
    а content всегда становится обычной строкой. Это защищает от 422, возникающих,
    если content оказался списком/кортежем или роль была не из допустимого набора.
    """
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(messages):
        if isinstance(raw, dict):
            raw_role = str(raw.get("role") or "").strip().lower()
            content = _coerce_content(raw.get("content"))
        else:
            raw_role = ""
            content = _coerce_content(raw)
        role = raw_role if raw_role in _LLM_ROLES else ("system" if index == 0 else "user")
        normalized.append({"role": role, "content": content})
    return normalized


async def _create_chat_completion(messages: list[dict[str, str]]):
    """Отправляет запрос к LLM, повторяя его при временных сбоях.

    Перед отправкой сообщения нормализуются (см. _normalize_messages): роли и типы
    content приводятся к формату, который принимает DeepSeek. Повторы выполняются с
    экспоненциальной задержкой до LLM_MAX_ATTEMPTS попыток.
    """
    if llm_client is None:
        raise RuntimeError("LLM_API_KEY не задан — клиент LLM недоступен.")

    messages = _normalize_messages(messages)

    last_error: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            return await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=DM_TEMPERATURE,
                max_tokens=DM_MAX_TOKENS,
                stream=False,
            )
        except _RETRYABLE_LLM_ERRORS as error:
            last_error = error
            if attempt >= LLM_MAX_ATTEMPTS:
                break
            delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Временный сбой LLM (%s): попытка %d/%d, повтор через %.1f с",
                type(error).__name__,
                attempt,
                LLM_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


async def ask_dungeon_master(
    history: Iterable[dict[str, str]],
    *,
    species_name: Optional[str] = None,
    hero_ready: bool = False,
    party_sheet: str = "",
) -> str:
    """Отправляет историю диалога Мастеру и возвращает текст ответа.

    Системный промпт собирается под фазу игры (см. build_system_prompt). Сводный лист
    партии (party_sheet) добавляется отдельным системным сообщением, чтобы Мастер видел
    всех героев сразу. Сама история уже ограничена по длине (см. PartySession).
    """
    if llm_client is None:
        raise RuntimeError("LLM_API_KEY не задан — клиент LLM недоступен.")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(species_name=species_name, hero_ready=hero_ready),
        },
    ]
    if party_sheet:
        messages.append({"role": "system", "content": party_sheet})
    messages.extend(history)

    response = await _create_chat_completion(messages)

    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info(
            "📊 [TOKENS] Вход: %s, Выход: %s | ИТОГО: %s",
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
            getattr(usage, "total_tokens", 0),
        )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Мастер вернул пустой ответ.")

    return content.strip()


# ---------------------------------------------------------------------------
# 14. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ВЫВОДА
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Разбивает длинный текст на части по лимиту Telegram (с переносом по абзацам)."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip("\n ")
    return chunks


async def send_long_message(
    message: Message,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """Отправляет (возможно, длинный) текст, разбивая его на несколько сообщений.

    Инлайн-клавиатура (если задана) прикрепляется к последнему сообщению.
    """
    chunks = [chunk for chunk in split_message(text) if chunk]
    for index, chunk in enumerate(chunks):
        markup = reply_markup if index == len(chunks) - 1 else None
        await message.answer(chunk, reply_markup=markup)


async def _send_chat_message(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """Отправляет (возможно, длинный) текст в произвольный чат, разбивая его на части.

    Нужно для «Варианта Б»: Мастер вводит нового героя прямо в групповом чате, хотя событие
    (подтверждение героя) пришло из личных сообщений. Клавиатура крепится к последней части.
    """
    chunks = [chunk for chunk in split_message(text) if chunk]
    for index, chunk in enumerate(chunks):
        markup = reply_markup if index == len(chunks) - 1 else None
        await bot.send_message(chat_id, chunk, reply_markup=markup)


# ---------------------------------------------------------------------------
# 15. МАРШРУТИЗАЦИЯ, ДОСТУП И ИНЛАЙН-КЛАВИАТУРЫ
# ---------------------------------------------------------------------------

router = Router()


class RateLimiter:
    """Ограничитель частоты обращений на пользователя (скользящее окно)."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max(1, int(max_requests))
        self._window = max(1.0, float(window_seconds))
        self._hits: dict[int, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, user_id: int) -> bool:
        """True, если запрос можно обработать; False — лимит превышен."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(user_id, deque())
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._max:
                return False
            hits.append(now)
            return True


_rate_limiter = RateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


async def _reject_event(event: Any, text: str) -> None:
    """Сообщает пользователю об отказе (текстом или всплывающим окном кнопки)."""
    try:
        if isinstance(event, CallbackQuery):
            await event.answer(text, show_alert=True)
        else:
            await event.answer(text)
    except Exception:  # noqa: BLE001 — отказ не должен ломать обработку апдейтов
        logger.exception("Не удалось отправить сообщение об отказе")


class AccessAndRateLimitMiddleware(BaseMiddleware):
    """Отсекает посторонних (белый список) и слишком частые обращения."""

    async def __call__(self, handler, event, data):
        user_id = getattr(getattr(event, "from_user", None), "id", None)
        if user_id is None:
            return await handler(event, data)

        if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
            logger.warning("Отказ в доступе пользователю %s (нет в белом списке)", user_id)
            await _reject_event(event, ACCESS_DENIED_TEXT)
            return None

        if user_id not in ADMIN_USER_IDS and not _rate_limiter.allow(user_id):
            logger.warning("Пользователь %s превысил лимит обращений", user_id)
            await _reject_event(event, RATE_LIMIT_TEXT)
            return None

        return await handler(event, data)


def _is_addressed_to_bot(message: Message) -> bool:
    """Адресовано ли сообщение боту (в группе отвечаем только когда позвали).

    Срабатывает на: упоминание @бота в тексте; ответ (reply) на сообщение бота;
    объект-упоминание в entities. В личном чате бот отвечает всегда.
    """
    chat = message.chat
    if chat is not None and chat.type == "private":
        return True

    text = message.text or message.caption or ""
    if BOT_USERNAME and f"@{BOT_USERNAME.lower()}" in text.lower():
        return True

    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.id == BOT_ID:
        return True

    for entity in (message.entities or message.caption_entities or []):
        if entity.type == "mention":
            mention = text[entity.offset: entity.offset + entity.length].lstrip("@").lower()
            if BOT_USERNAME and mention == BOT_USERNAME.lower():
                return True
        elif entity.type == "text_mention" and getattr(entity, "user", None) is not None:
            if entity.user.id == BOT_ID:
                return True

    return False


def _cb(action: str, uid: int, param: str = "") -> str:
    """Собирает callback_data с вшитым id владельца: «action[:param]:uid»."""
    if param:
        return f"{action}:{param}:{int(uid)}"
    return f"{action}:{int(uid)}"


def _parse_callback(data: str) -> tuple[str, str, Optional[int]]:
    """Разбирает callback_data на (действие, параметр, id владельца)."""
    parts = (data or "").split(":")
    if not parts:
        return "", "", None
    action = parts[0]
    owner: Optional[int] = None
    if parts and parts[-1].isdigit():
        owner = int(parts[-1])
        parts = parts[:-1]
    param = parts[1] if len(parts) > 1 else ""
    return action, param, owner


def _weapon_damage_button(character: Character, uid: int) -> Optional[InlineKeyboardButton]:
    """Кнопка урона текущим оружием героя или None, если её показывать не нужно."""
    weapon = find_inventory_weapon(character)
    if weapon is None:
        return InlineKeyboardButton(
            text=f"🗡 {IMPROVISED_DAMAGE_DICE} (без оружия)",
            callback_data=_cb("roll", uid, "d4"),
        )

    _name, damage_dice, damage_type, _kind = weapon
    sides = _damage_die_sides(damage_dice)
    action = f"d{sides}" if sides is not None else ""
    if action not in DAMAGE_DIE_SIDES:
        return None
    return InlineKeyboardButton(
        text=f"🗡 {damage_dice} {damage_type}",
        callback_data=_cb("roll", uid, action),
    )


def build_action_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Игровая сетка кнопок с учётом экипированного оружия героя.

    ``callback_data`` несут id владельца (``uid``): нажатие принимает только его герой.
    """
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(text="🎲 d20", callback_data=_cb("roll", uid, "d20")),
            InlineKeyboardButton(text="🎲 d20 с преим.", callback_data=_cb("roll", uid, "adv")),
            InlineKeyboardButton(text="🎲 d20 с помех.", callback_data=_cb("roll", uid, "dis")),
        ],
    ]

    damage_button = _weapon_damage_button(character, uid)
    if damage_button is not None:
        rows.append([damage_button])

    rows.append(
        [
            InlineKeyboardButton(text="📜 Лист", callback_data=_cb("sheet", uid)),
            InlineKeyboardButton(text="🎒 Инвентарь", callback_data=_cb("inventory", uid)),
            InlineKeyboardButton(text="🎲 Бросок урона", callback_data=_cb("roll", uid, "damage")),
        ]
    )
    rows.append(
        [InlineKeyboardButton(text="🧠 Проверки по статам", callback_data=_cb("checks", uid))]
    )
    if character.is_spellcaster:
        rows.append(
            [
                InlineKeyboardButton(text="📜 Заклинания", callback_data=_cb("spells", uid)),
                InlineKeyboardButton(text="🌙 Отдых", callback_data=_cb("rest", uid)),
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_checks_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Сетка проверок характеристик и спасбросков с модификаторами из листа персонажа."""
    check_buttons = [
        InlineKeyboardButton(
            text=f"{ABILITIES[code]} {format_modifier(character.roll_modifier(code))}",
            callback_data=_cb("check", uid, code),
        )
        for code in ABILITIES
    ]
    save_buttons = [
        InlineKeyboardButton(
            text=f"🛡 {ABILITIES[code]} {format_modifier(character.save_modifier(code))}",
            callback_data=_cb("save", uid, code),
        )
        for code in ABILITIES
    ]
    rows = [check_buttons[index:index + 3] for index in range(0, len(check_buttons), 3)]
    rows += [save_buttons[index:index + 3] for index in range(0, len(save_buttons), 3)]
    rows.append([InlineKeyboardButton(text="⬅️ Быстрые действия", callback_data=_cb("menu", uid))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _spell_button(
    character: Character, prefix: str, index: int, name: str, circle: int, uid: int
) -> InlineKeyboardButton:
    """Кнопка заклинания: ``prefix`` — «cast» (применить) или «prep» (подготовить)."""
    if prefix == "cast":
        mark = "🔒" if _spell_circle_locked(character, circle) else ("🪄" if circle == 0 else "🔥")
    else:
        mark = "✅" if name in character.spells_prepared else "❌"
    return InlineKeyboardButton(
        text=f"{mark} {name} ({_spell_circle_label(circle)})",
        callback_data=_cb(prefix, uid, str(index)),
    )


def build_spells_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Сетка раздела магии: применение, подготовка (если нужно) и отдых."""
    rows: list[list[InlineKeyboardButton]] = []
    if character.is_spellcaster:
        rows.append(
            [InlineKeyboardButton(text="🔥 Применить заклинание", callback_data=_cb("cast", uid))]
        )
        if not is_spontaneous_caster(character.class_name):
            rows.append(
                [
                    InlineKeyboardButton(
                        text="⚡ Подготовка "
                        f"({len(character.spells_prepared)}/{character.max_prepared})",
                        callback_data=_cb("prep", uid),
                    )
                ]
            )
        rows.append([InlineKeyboardButton(text="🌙 Отдохнуть", callback_data=_cb("rest", uid))])
    rows.append([InlineKeyboardButton(text="⬅️ Быстрые действия", callback_data=_cb("menu", uid))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_cast_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Кнопки применения: по одной на каждый заговор и готовое заклинание."""
    rows = [
        [_spell_button(character, "cast", index, name, circle, uid)]
        for index, (name, circle) in enumerate(character.castable_spells())
    ]
    rows.append([InlineKeyboardButton(text="🔄 Обновить", callback_data=_cb("cast", uid))])
    rows.append([InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data=_cb("spells", uid))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_prep_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Кнопки подготовки: ✅/❌ на каждое изученное заклинание (кроме спонтанных)."""
    rows: list[list[InlineKeyboardButton]] = []
    if character.is_spellcaster and not is_spontaneous_caster(character.class_name):
        for index, name in enumerate(character.spells_known):
            rows.append(
                [_spell_button(character, "prep", index, name, character.spell_circle(name), uid)]
            )
    rows.append([InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data=_cb("spells", uid))])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_rest_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Кнопки отдыха: короткий (1 час) и продолжительный (8 часов)."""
    rows = [
        [InlineKeyboardButton(text="☕ Короткий отдых (1 час)", callback_data=_cb("rest", uid, "short"))],
        [InlineKeyboardButton(text="🌙 Продолжительный отдых (8 часов)", callback_data=_cb("rest", uid, "long"))],
        [InlineKeyboardButton(text="⬅️ Книга заклинаний", callback_data=_cb("spells", uid))],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_creation_keyboard(uid: int) -> InlineKeyboardMarkup:
    """Кнопки этапа создания персонажа: пока герой не подтверждён, показываем именно их."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🎲 Случайный герой", callback_data=_cb("hero", uid, "random")),
                InlineKeyboardButton(text="✅ Подтвердить героя", callback_data=_cb("hero", uid, "confirm")),
            ],
            [
                InlineKeyboardButton(text="📜 Лист", callback_data=_cb("sheet", uid)),
            ],
        ]
    )


def build_join_dm_button(target_chat_id: int) -> InlineKeyboardMarkup:
    """Кнопка-ссылка (deep-link) на личку бота для создания героя под конкретную группу.

    Ссылка вида https://t.me/<bot>?start=join_<chat_id> открывает ЛС бота и отправляет
    /start с аргументом; обработчик /start запускает создание героя именно для этой группы.
    """
    url = f"https://t.me/{BOT_USERNAME}?start=join_{int(target_chat_id)}"
    return InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=JOIN_DM_BUTTON_TEXT, url=url)]]
    )


def phase_keyboard(character: Character, uid: int) -> InlineKeyboardMarkup:
    """Клавиатура по фазе игры: создание героя или игровые кнопки героя."""
    return build_action_keyboard(character, uid) if character.hero_confirmed else build_creation_keyboard(uid)


# ---------------------------------------------------------------------------
# 16. БОЕВОЙ РЕЖИМ И ОТВЕТ МАСТЕРА
# ---------------------------------------------------------------------------

# Напоминание Мастеру во время боя (не хранится в истории).
COMBAT_CONTINUATION_NOTE = (
    "[СИСТЕМА] Продолжается бой. Веди полный учёт всех врагов, их ранений и позиций, "
    "заканчивай ход фразой «Твой ход» и указывай в служебном блоке \"in_combat\": true. "
    "Если бой в этом ходе ЗАВЕРШИЛСЯ — поставь \"in_combat\": false и добавь "
    "\"battle_summary\" с кратким итогом схватки."
)


def _control_flag_to_bool(value: Any) -> Optional[bool]:
    """Приводит значение флага из служебного блока к bool/None (терпимо к строкам)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "да", "y", "on", "бой", "бою"}:
            return True
        if token in {"false", "0", "no", "нет", "n", "off", "мир"}:
            return False
    return None


def _fallback_battle_summary(party: PartySession) -> str:
    """Запасной итог боя, если Мастер не прислал поле "battle_summary"."""
    location = (party.location or "").strip()
    return f"сражение завершено (локация: {location})" if location else "сражение завершено."


def _update_combat_state(
    party: PartySession,
    control: Optional[dict],
    reply: str,
    was_in_combat: bool,
) -> None:
    """Обновляет боевой режим партии: начало боя и свёртку боя после его окончания."""
    if not party.started:
        return  # до начала игры боя быть не может

    signal = _control_flag_to_bool(control.get("in_combat")) if control else None

    if signal is None:
        # Мастер не прислал флаг: ориентируемся на каноническую фразу конца боевого хода.
        ends_in_combat = "твой ход" in reply.lower()
        if was_in_combat and not ends_in_combat:
            party.end_combat(_fallback_battle_summary(party))
            logger.info("Чат %s: бой завершён (без флага) — история свёрнута", party.chat_id)
        elif not was_in_combat and ends_in_combat:
            party.begin_combat()
            logger.info("Чат %s: начался бой — история не будет обрезаться", party.chat_id)
        return

    if signal and not was_in_combat:
        party.begin_combat()
        logger.info("Чат %s: начался бой — история не будет обрезаться", party.chat_id)
    elif not signal and was_in_combat:
        summary = ""
        if control:
            raw = control.get("battle_summary")
            if isinstance(raw, str) and raw.strip():
                summary = " ".join(raw.split())
        if len(summary) > 400:
            summary = summary[:400].rstrip() + "…"
        party.end_combat(summary or _fallback_battle_summary(party))
        logger.info("Чат %s: бой завершён — раунды сжаты в резюме", party.chat_id)


def _apply_party_control(party: PartySession, control: Optional[dict]) -> list[str]:
    """Применяет общую сводку Мастера (локация, цель отряда) и сохраняет кампанию."""
    if not control:
        return []
    notes: list[str] = []
    changed = False

    location = control.get("location")
    if isinstance(location, str) and location.strip():
        cleaned = location.strip()[:120]
        if cleaned != party.location:
            party.location = cleaned
            notes.append(f"📍 Новая локация отряда: {party.location}.")
            changed = True

    quest = control.get("quest")
    if isinstance(quest, str) and quest.strip():
        cleaned = quest.strip()[:120]
        if cleaned != party.quest:
            party.quest = cleaned
            notes.append(f"🎯 Новая цель отряда: {party.quest}.")
            changed = True

    if changed:
        party.save_campaign()
    return notes


def _character_change_status(character: Character, xp_delta: int, hp_delta: int) -> str:
    """Короткая сводка об изменениях XP/HP для уведомления игроку."""
    parts: list[str] = []
    if xp_delta > 0:
        parts.append(f"⚡ Получено опыта: +{xp_delta} XP (всего {character.xp})")
    elif xp_delta < 0:
        parts.append(f"⚡ Опыт: {xp_delta} XP (всего {character.xp})")
    if hp_delta:
        parts.append(f"❤️ HP: {character.current_hp}/{character.max_hp}")
    return " | ".join(parts)


async def _answer_with_dungeon_master(
    message: Message,
    party: PartySession,
    actor: Character,
    uid: int,
    *,
    hero_card_title: Optional[str] = None,
    output_chat_id: Optional[int] = None,
) -> None:
    """Запрашивает ответ Мастера, обновляет лист действующего героя и отправляет ответ в чат.

    :param actor: герой того игрока, чья реплика только что пришла — служебный блок
        Мастера относится ИМЕННО к нему.
    :param hero_card_title: заголовок карточки героя, пока он не подтверждён.
    :param output_chat_id: куда отправить ответ Мастера. По умолчанию — туда, откуда пришло
        сообщение. Для «Варианта Б» задаём групповой чат: история и партия — групповые, а
        событие (подтверждение героя) пришло из личных сообщений.
    """
    out_chat = int(output_chat_id) if output_chat_id is not None else message.chat.id
    await message.bot.send_chat_action(out_chat, ChatAction.TYPING)

    was_in_combat = party.in_combat
    request_messages = party.messages()
    if was_in_combat:
        request_messages = [
            *request_messages,
            {"role": "user", "content": COMBAT_CONTINUATION_NOTE},
        ]

    try:
        raw_reply = await ask_dungeon_master(
            request_messages,
            species_name=actor.race,
            hero_ready=actor.is_created,
            party_sheet=build_party_sheet(party),
        )
    except APIError as error:
        logger.error("Ошибка LLM / DeepSeek API: %s", error)
        await _send_chat_message(message.bot, out_chat, API_ERROR_TEXT)
        return
    except Exception:  # noqa: BLE001 — на верхнем уровне бота логируем всё непредвиденное
        logger.exception("Непредвиденная ошибка при обращении к Мастеру")
        await _send_chat_message(message.bot, out_chat, GENERIC_ERROR_TEXT)
        return

    reply, control = extract_control_block(raw_reply)

    xp_before = actor.xp
    hp_before = actor.current_hp

    notes = actor.apply_control(control) if control else []
    notes.extend(_apply_party_control(party, control))

    starter_notes: list[str] = []
    if actor.is_created and not actor.hero_confirmed:
        hp_explicit = bool(
            control and (control.get("max_hp") is not None or control.get("current_hp") is not None)
        )
        starter_notes = apply_starter_loadout(actor, recalc_hp=not hp_explicit)
        notes.extend(starter_notes)

    xp_delta = actor.xp - xp_before
    hp_delta = actor.current_hp - hp_before

    if control is not None:
        logger.info("Служебный блок Мастера применён (чат %s): %s", party.chat_id, control)
    if control is not None or starter_notes:
        party.save_character(uid)

    if not reply:
        reply = "Мастер молчаливо следит за происходящим.\n\nЧто вы делаете?"

    _update_combat_state(party, control, reply, was_in_combat)

    party.add("assistant", reply)

    outgoing = reply
    if actor.hero_confirmed:
        outgoing = f"{format_hud(party.location, party.quest)}\n\n{reply}"
    if out_chat == message.chat.id:
        await send_long_message(message, outgoing, reply_markup=phase_keyboard(actor, uid))
    else:
        await _send_chat_message(
            message.bot, out_chat, outgoing, reply_markup=phase_keyboard(actor, uid)
        )

    if notes:
        status = _character_change_status(actor, xp_delta, hp_delta)
        body = "\n".join(f"• {note}" for note in notes)
        prefix = f"{status}\n" if status else ""
        await _send_chat_message(
            message.bot, out_chat, f"{prefix}📈 Обновление листа персонажа ({actor.name}):\n{body}"
        )

    if actor.is_created and not actor.hero_confirmed:
        if hero_card_title is not None:
            await _send_hero_card(message, party, actor, uid, hero_card_title)
        elif starter_notes:
            await _send_hero_card(message, party, actor, uid, HERO_CARD_TITLE_CREATED)
        else:
            await _send_hero_card(message, party, actor, uid)


async def _resolve_roll(
    message: Message,
    party: PartySession,
    actor: Character,
    uid: int,
    roll: DiceRoll,
    context: Optional[str] = None,
) -> None:
    """Обрабатывает бросок кодом бота (команда /roll или кнопка).

    1) показывает игроку «математику» броска; 2) пишет бросок в историю как ход игрока
    (в память и в SQLite, с ярлыком героя); 3) просит Мастера описать исход.
    """
    await message.answer(f"🎲 {actor.name}: {roll.describe()}")
    logger.info("Чат %s: %s бросил %s = %s", party.chat_id, actor.name, roll.notation, roll.total)

    party.add("user", context or roll.context_message(), user_id=uid, char_name=character_label(actor))
    await _answer_with_dungeon_master(message, party, actor, uid)


def _callback_context(callback: CallbackQuery) -> Optional[tuple[Message, int]]:
    """Достаёт сообщение и id игрока из нажатия кнопки: (Message, user_id) или None."""
    if callback.from_user is None or not isinstance(callback.message, Message):
        return None
    return callback.message, callback.from_user.id


async def _send_hero_card(
    message: Message,
    party: PartySession,
    actor: Character,
    uid: int,
    title: str = HERO_CARD_TITLE_PENDING,
) -> None:
    """Показывает лист созданного героя и просит игрока подтвердить его."""
    await send_long_message(
        message,
        f"{title}\n\n{format_character_sheet(actor)}\n\n{HERO_CARD_QUESTION}",
        reply_markup=build_creation_keyboard(uid),
    )


async def _begin_hero_creation(
    message: Message, party: PartySession, actor: Character, uid: int
) -> None:
    """Этап 1: просит Мастера поприветствовать игрока и помочь создать героя.

    Инструкция «первое сообщение» используется, только если в истории партии ещё нет
    реплик (самый первый игрок); иначе берётся вариант «продолжение».
    """
    first_message = not party.history
    party.add(
        "user",
        creation_prompt(first_game_message=first_message),
        user_id=uid,
        char_name=character_label(actor) if actor.is_created else "",
    )
    await _answer_with_dungeon_master(message, party, actor, uid)


async def _create_random_hero(
    message: Message, party: PartySession, actor: Character, uid: int
) -> Character:
    """Генерирует случайного героя кодом по правилам PHB 2024 и просит Мастера его представить."""
    hero = build_random_hero()
    # Имя и описание, которые игрок успел назвать сам, не теряем.
    if actor.name != DEFAULT_NAME:
        hero.name = actor.name
    if actor.description:
        hero.description = actor.description

    party.characters[int(uid)] = hero
    party.save_character(uid)
    logger.info(
        "Чат %s: игрок %s получил случайного героя: %s, %s %s",
        party.chat_id,
        uid,
        hero.name,
        hero.race,
        hero.class_name,
    )

    party.add("user", random_hero_prompt(hero), user_id=uid, char_name=character_label(hero))
    await _answer_with_dungeon_master(
        message, party, hero, uid, hero_card_title=HERO_CARD_TITLE_RANDOM
    )
    return hero


async def _confirm_hero_and_start_prologue(
    message: Message, party: PartySession, actor: Character, uid: int
) -> None:
    """Игрок подтвердил героя: фиксируем это и начинаем вводную сцену.

    Первый подтверждённый герой запускает пролог и помечает партию «начатой»; остальные
    вводятся в уже идущую историю (Мастер получает список товарищей по отряду).
    """
    actor.hero_confirmed = True
    party.save_character(uid)

    others = [ch.name for ch in party.confirmed_heroes().values() if ch is not actor]
    if not party.started:
        party.started = True
        party.save_campaign()

    party.add(
        "user",
        prologue_prompt(actor, others),
        user_id=uid,
        char_name=character_label(actor),
    )
    logger.info("Чат %s: игрок %s подтвердил героя «%s»", party.chat_id, uid, actor.name)
    await _answer_with_dungeon_master(message, party, actor, uid)


async def _start_private_hero_creation(
    message: Message, uid: int, target_chat_id: int
) -> None:
    """Запускает пошаговое создание героя в ЛС для указанной группы («Вариант Б»).

    Deep-link «/start join_<chat_id>» фиксирует target_chat_id у игрока и открывает привычный
    диалог создания героя в личных сообщениях бота (случайный или кастомный).
    """
    uid = int(uid)
    target_chat_id = int(target_chat_id)
    target_party = get_party(target_chat_id)

    existing = target_party.character_for(uid)
    if existing is not None and existing.hero_confirmed:
        _join_targets.pop(uid, None)
        await message.answer(PRIVATE_HERO_ALREADY_TEXT)
        return
    if existing is None and target_party.is_full():
        _join_targets.pop(uid, None)
        await message.answer(PARTY_FULL_TEXT.format(max_size=MAX_PARTY_SIZE))
        return

    _join_targets[uid] = target_chat_id
    _delivered.pop(uid, None)

    # Черновая партия в ЛС: Мастер ведёт создание героя здесь, чтобы не засорять группу.
    private_party = get_party(message.chat.id)
    actor = private_party.ensure_character(uid)

    logger.info("Игрок %s начал создание героя в ЛС для чата %s", uid, target_chat_id)
    await message.answer(JOIN_DM_START_TEXT)

    if actor.is_created and not actor.hero_confirmed:
        await _send_hero_card(message, private_party, actor, uid)
        return
    await _begin_hero_creation(message, private_party, actor, uid)


async def _deliver_hero_to_group(
    message: Message,
    private_party: PartySession,
    actor: Character,
    uid: int,
    *,
    username: Optional[str],
    full_name: str,
) -> None:
    """Переносит подтверждённого в ЛС героя в целевую группу и запускает вступление.

    1) подтверждает героя и сохраняет его в БД под (target_chat_id, user_id);
    2) убирает черновую партию из ЛС;
    3) пишет игроку в ЛС «Персонаж готов»;
    4) объявляет нового героя в группе и прикрепляет карточку;
    5) первый герой запускает пролог, остальные — вводятся в уже идущую сцену.
    """
    uid = int(uid)
    target_chat_id = _join_targets.get(uid)
    if target_chat_id is None:
        await message.answer(PRIVATE_HERO_NO_TARGET_TEXT)
        return

    target_party = get_party(target_chat_id)

    # Повторная проверка на случай гонки: герой уже есть или отряд полон.
    existing = target_party.character_for(uid)
    if existing is not None and existing.hero_confirmed:
        _join_targets.pop(uid, None)
        await message.answer(PRIVATE_HERO_ALREADY_TEXT)
        return
    if existing is None and target_party.is_full():
        await message.answer(PARTY_FULL_TEXT.format(max_size=MAX_PARTY_SIZE))
        return

    actor.hero_confirmed = True
    target_party.characters[uid] = actor
    target_party.save_character(uid)

    others = [ch.name for ch in target_party.confirmed_heroes().values() if ch is not actor]

    # Прибираем черновую партию в ЛС: герой теперь живёт в группе.
    private_party.characters.pop(uid, None)
    private_party.reset()
    _parties.pop(message.chat.id, None)
    _join_targets.pop(uid, None)
    _delivered[uid] = target_chat_id

    logger.info(
        "Чат %s: игрок %s подтвердил героя «%s» в ЛС и передал его в группу %s",
        message.chat.id,
        uid,
        actor.name,
        target_chat_id,
    )

    await message.answer(PRIVATE_HERO_DONE_TEXT)

    # Торжественное объявление в группе + карточка героя.
    mention = f"@{username}" if username else (full_name or "")
    try:
        await message.bot.send_message(target_chat_id, new_hero_announcement(actor, mention))
        await _send_chat_message(message.bot, target_chat_id, format_character_sheet(actor))
    except Exception:  # noqa: BLE001 — группу могли удалить или бот в ней заблокирован
        logger.exception("Не удалось объявить нового героя в чате %s", target_chat_id)

    # Пролог (первый герой) или ввод в уже идущую сцену.
    if not target_party.started:
        target_party.started = True
        target_party.save_campaign()
        prompt = prologue_prompt(actor, others)
    else:
        prompt = join_scene_prompt(actor, others, target_party.location)

    target_party.add("user", prompt, user_id=uid, char_name=character_label(actor))
    await _answer_with_dungeon_master(
        message, target_party, actor, uid, output_chat_id=target_chat_id
    )


# ---------------------------------------------------------------------------
# 17. ОБРАБОТЧИКИ (aiogram router)
# ---------------------------------------------------------------------------


def _strip_bot_mention(text: str) -> str:
    """Убирает из реплики игрока упоминание @бота (чтобы оно не уходило Мастеру)."""
    if BOT_USERNAME:
        text = re.sub(rf"@{re.escape(BOT_USERNAME)}\b", "", text, flags=re.IGNORECASE)
    return text.strip()


async def _require_actor(
    callback: CallbackQuery,
) -> Optional[tuple[Message, int, PartySession, Character]]:
    """Проверяет владельца кнопки и наличие героя. Возвращает контекст или None.

    Если кнопку нажал не тот, кому она принадлежит (в callback_data вшит id владельца),
    показываем подсказку «Это действие другого персонажа!».
    """
    target = _callback_context(callback)
    if target is None:
        await callback.answer(STALE_CALLBACK_TEXT, show_alert=True)
        return None

    message, uid = target
    _, _, owner = _parse_callback(callback.data or "")
    if owner is not None and owner != uid:
        await callback.answer(NOT_YOUR_BUTTON_TEXT, show_alert=True)
        return None

    party = get_party(message.chat.id)
    actor = party.character_for(uid)
    if actor is None:
        await callback.answer(NOT_IN_PARTY_TEXT, show_alert=True)
        return None
    return message, uid, party, actor


def _is_group_chat(chat_type: str) -> bool:
    """True для групповых чатов Telegram (обычная группа или супергруппа)."""
    return chat_type in {"group", "supergroup"}


def _private_hint_for(uid: int) -> str:
    """Подсказка в ЛС: где создаётся герой (создание идёт из группового чата)."""
    if int(uid) in _delivered:
        return PRIVATE_HERO_DONE_HINT
    return JOIN_PRIVATE_HINT


@router.message(CommandStart())
async def handle_start(message: Message, command: CommandObject) -> None:
    """/start — в группе справка; в ЛС с аргументом «join_<chat_id>» — создание героя.

    «Вариант Б»: из группы игрок приходит по deep-link /start join_<chat_id>, и здесь
    запускается пошаговое создание героя в личных сообщениях для указанной группы.
    """
    if message.from_user is None:
        return
    uid = message.from_user.id
    target_chat_id = parse_join_deep_link(command.args)

    if target_chat_id is not None and message.chat.type == "private":
        await _start_private_hero_creation(message, uid, target_chat_id)
        return

    if message.chat.type == "private":
        await message.answer(_private_hint_for(uid))
        return
    await message.answer(WELCOME_TEXT)


@router.message(Command("join"))
async def handle_join(message: Message) -> None:
    """/join — в группе присылает кнопку для создания героя в личных сообщениях.

    «Вариант Б»: чтобы не загромождать общий чат, пошаговое создание героя проходит в ЛС.
    Игрок переходит по deep-link и создаёт героя; готовый герой автоматически появляется здесь.
    """
    if message.from_user is None:
        return

    # В ЛС самостоятельное вступление не начинаем: герой создаётся из группового чата.
    if message.chat.type == "private":
        await message.answer(_private_hint_for(message.from_user.id))
        return

    if not _is_group_chat(message.chat.type):
        await message.answer(JOIN_PRIVATE_HINT)
        return

    party = get_party(message.chat.id)
    uid = message.from_user.id

    actor = party.character_for(uid)
    if actor is not None and actor.hero_confirmed:
        await send_long_message(
            message,
            JOIN_ALREADY_TEXT.format(sheet=format_character_sheet(actor)),
            reply_markup=build_action_keyboard(actor, uid),
        )
        return

    if actor is None and party.is_full():
        await message.answer(PARTY_FULL_TEXT.format(max_size=MAX_PARTY_SIZE))
        return

    if not BOT_USERNAME:
        await message.answer(JOIN_DM_NO_USERNAME_TEXT)
        return

    mention = (
        f"@{message.from_user.username}"
        if message.from_user.username
        else message.from_user.full_name
    )
    logger.info("Чат %s: игрок %s получил кнопку создания героя в ЛС", message.chat.id, uid)
    await message.answer(
        JOIN_GROUP_DM_TEXT.format(mention=mention),
        reply_markup=build_join_dm_button(message.chat.id),
    )


@router.message(Command("hero"))
async def handle_random_hero(message: Message) -> None:
    """/hero — сгенерировать случайного героя 1-го уровня (в рамках вступления в отряд)."""
    if message.from_user is None:
        return
    # В ЛС команда имеет смысл только во время создания героя (после перехода из группы).
    if message.chat.type == "private" and int(message.from_user.id) not in _join_targets:
        await message.answer(_private_hint_for(message.from_user.id))
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)

    if actor is not None and actor.hero_confirmed:
        await message.answer(
            HERO_ALREADY_CONFIRMED_TEXT,
            reply_markup=build_action_keyboard(actor, uid),
        )
        return

    if actor is None and party.is_full():
        await message.answer(PARTY_FULL_TEXT.format(max_size=MAX_PARTY_SIZE))
        return

    actor = party.ensure_character(uid)
    await _create_random_hero(message, party, actor, uid)


@router.message(Command("party"))
async def handle_party(message: Message) -> None:
    """/party — статус отряда: кто в группе, HP и уровень каждого героя."""
    party = get_party(message.chat.id)
    await message.answer(format_party_status(party))


@router.message(Command("sheet"))
async def handle_sheet(message: Message) -> None:
    """/sheet — лист СВОЕГО персонажа."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return
    await send_long_message(
        message,
        format_character_sheet(actor),
        reply_markup=phase_keyboard(actor, uid),
    )


@router.message(Command("inventory"))
async def handle_inventory(message: Message) -> None:
    """/inventory — снаряжение и золото СВОЕГО персонажа."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return
    await message.answer(format_inventory(actor), reply_markup=phase_keyboard(actor, uid))


@router.message(Command("check"))
async def handle_check(message: Message) -> None:
    """/check — меню проверок характеристик."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return
    await message.answer(CHECKS_MENU_TEXT, reply_markup=build_checks_keyboard(actor, uid))


@router.message(Command("spells"))
async def handle_spells(message: Message) -> None:
    """/spells — книга заклинаний своего персонажа."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return
    if not actor.is_spellcaster:
        await message.answer(NOT_SPELLCASTER_TEXT, reply_markup=phase_keyboard(actor, uid))
        return
    await send_long_message(
        message,
        format_spell_card(actor),
        reply_markup=build_spells_keyboard(actor, uid),
    )


@router.message(Command("rest"))
async def handle_rest(message: Message) -> None:
    """/rest — меню отдыха героя (короткий/продолжительный)."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return
    await message.answer(format_rest_menu(actor), reply_markup=build_rest_keyboard(actor, uid))


@router.message(Command("roll"))
async def handle_roll(message: Message, command: CommandObject) -> None:
    """/roll <кубик> — бросок кубиков кодом с описанием исхода Мастером."""
    if message.from_user is None:
        return
    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        await message.answer(NOT_IN_PARTY_TEXT)
        return

    expression = (command.args or "").strip()
    if not expression:
        await message.answer(ROLL_USAGE_TEXT)
        return
    try:
        roll = parse_and_roll(expression)
    except ValueError as error:
        await message.answer(f"⚠️ {error}\n\n{ROLL_USAGE_TEXT}")
        return

    await _resolve_roll(message, party, actor, uid, roll)


@router.message(Command("reset_party"))
async def handle_reset_party(message: Message) -> None:
    """/reset_party — полный сброс партии чата (только администраторы)."""
    if message.from_user is None:
        return
    if not _is_admin(message.from_user.id):
        await message.answer(RESET_PARTY_NO_RIGHTS_TEXT)
        return
    party = get_party(message.chat.id)
    party.reset()
    _parties.pop(message.chat.id, None)
    await message.answer(RESET_PARTY_DONE_TEXT)
    logger.info(
        "Чат %s: партия сброшена администратором %s", message.chat.id, message.from_user.id
    )


@router.message(F.text & ~F.text.startswith("/"))
async def handle_player_action(message: Message) -> None:
    """Текстовое сообщение: создание героя, подтверждение или действие персонажа.

    В группе реагируем только на сообщения, адресованные боту (упоминание/ответ/команда).
    """
    if message.from_user is None:
        return
    if not _is_addressed_to_bot(message):
        return

    text = _strip_bot_mention((message.text or "").strip())
    if not text:
        return
    if len(text) > MAX_PLAYER_INPUT:
        text = text[:MAX_PLAYER_INPUT]

    party = get_party(message.chat.id)
    uid = message.from_user.id
    actor = party.character_for(uid)
    if actor is None:
        hint = _private_hint_for(uid) if message.chat.type == "private" else NOT_IN_PARTY_TEXT
        await message.answer(hint)
        return

    # Герой подтверждён — это обычное действие персонажа.
    if actor.hero_confirmed:
        party.add("user", text, user_id=uid, char_name=character_label(actor))
        await _answer_with_dungeon_master(message, party, actor, uid)
        return

    stage = creation_stage(actor)

    if stage == CREATION_STAGE_HERO:
        # Страховка: если Мастер забудет служебный блок, имя, вид и класс распознает код.
        fallback_notes = auto_fill_hero_details(actor, text)
        if fallback_notes:
            party.save_character(uid)
            await message.answer(
                "📈 Обновление листа персонажа:\n"
                + "\n".join(f"• {note}" for note in fallback_notes)
            )

        if not actor.is_created and is_random_hero_request(text):
            await _create_random_hero(message, party, actor, uid)
            return

        if actor.is_created and is_hero_confirmation(text):
            if message.chat.type == "private" and uid in _join_targets:
                # «Вариант Б»: игрок подтвердил героя словом — отправляем его в группу.
                await _deliver_hero_to_group(
                    message,
                    party,
                    actor,
                    uid,
                    username=message.from_user.username,
                    full_name=message.from_user.full_name,
                )
            else:
                await _confirm_hero_and_start_prologue(message, party, actor, uid)
            return

        char_name = character_label(actor) if actor.is_created else ""
        party.add("user", text, user_id=uid, char_name=char_name)
        party.add("user", creation_prompt(first_game_message=False), user_id=uid)
        await _answer_with_dungeon_master(message, party, actor, uid)
        return

    if stage == CREATION_STAGE_CONFIRM:
        if is_hero_confirmation(text):
            if message.chat.type == "private" and uid in _join_targets:
                # «Вариант Б»: игрок подтвердил героя словом — отправляем его в группу.
                await _deliver_hero_to_group(
                    message,
                    party,
                    actor,
                    uid,
                    username=message.from_user.username,
                    full_name=message.from_user.full_name,
                )
            else:
                await _confirm_hero_and_start_prologue(message, party, actor, uid)
            return
        party.add("user", text, user_id=uid, char_name=character_label(actor))
        party.add("user", hero_confirmation_prompt(actor), user_id=uid)
        await _answer_with_dungeon_master(message, party, actor, uid)
        return

    # CREATION_STAGE_PLAY не должен сюда попадать (обработан выше).
    party.add("user", text, user_id=uid, char_name=character_label(actor))
    await _answer_with_dungeon_master(message, party, actor, uid)


@router.message()
async def handle_unsupported(message: Message) -> None:
    """Заглушка для нетекстовых сообщений (фото, стикеры и т.п.)."""
    if message.from_user is None or not _is_addressed_to_bot(message):
        return
    await message.answer(
        "Я понимаю только текст и кнопки. Опиши действие словами, нажми кнопку "
        "быстрого броска (🎲 d20, 🎲 Бросок урона, 🧠 Проверки по статам, 📜 Заклинания) "
        "или используй команды /join, /party, /roll, /sheet, /inventory, /check, /spells, /rest."
    )


@router.callback_query()
async def handle_callback(callback: CallbackQuery) -> None:
    """Единый диспетчер инлайн-кнопок: проверяет владельца и разводит по действиям."""
    action, param, _owner = _parse_callback(callback.data or "")

    # --- этап создания персонажа ---
    if action == "hero":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context
        if actor.hero_confirmed:
            await callback.answer(HERO_ALREADY_CONFIRMED_ALERT, show_alert=True)
            return
        if param == "random":
            await callback.answer()
            await _create_random_hero(message, party, actor, uid)
            return
        if param == "confirm":
            if not actor.is_created:
                await callback.answer(HERO_NOT_CREATED_ALERT, show_alert=True)
                return
            await callback.answer()
            if message.chat.type == "private" and uid in _join_targets:
                # «Вариант Б»: герой подтверждён в ЛС — отправляем его в целевую группу.
                await _deliver_hero_to_group(
                    message,
                    party,
                    actor,
                    uid,
                    username=callback.from_user.username if callback.from_user else None,
                    full_name=callback.from_user.full_name if callback.from_user else "",
                )
            else:
                await _confirm_hero_and_start_prologue(message, party, actor, uid)
            return
        await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
        return

    # --- простые меню листа персонажа ---
    if action in {"sheet", "inventory", "checks", "menu", "spells"}:
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context

        if action == "spells" and not actor.is_spellcaster:
            await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
            return
        await callback.answer()

        if action == "sheet":
            await send_long_message(
                message, format_character_sheet(actor), reply_markup=phase_keyboard(actor, uid)
            )
        elif action == "inventory":
            await message.answer(format_inventory(actor), reply_markup=phase_keyboard(actor, uid))
        elif action == "checks":
            await message.answer(CHECKS_MENU_TEXT, reply_markup=build_checks_keyboard(actor, uid))
        elif action == "menu":
            await message.answer(ACTION_MENU_TEXT, reply_markup=build_action_keyboard(actor, uid))
        elif action == "spells":
            await send_long_message(
                message, format_spell_card(actor), reply_markup=build_spells_keyboard(actor, uid)
            )
        return

    # --- магия: применение и подготовка ---
    if action == "cast":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context
        if not actor.is_spellcaster:
            await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
            return

        if not param:
            await callback.answer()
            await send_long_message(
                message, format_cast_menu(actor), reply_markup=build_cast_keyboard(actor, uid)
            )
            return

        spells = actor.castable_spells()
        index = int(param) if param.isdigit() else -1
        if not 0 <= index < len(spells):
            await callback.answer(SPELL_MENU_STALE_TEXT, show_alert=True)
            return
        name, _circle = spells[index]
        result = actor.cast_spell(name)
        if not result.ok:
            await callback.answer(result.alert, show_alert=True)
            return
        await callback.answer()
        party.save_character(uid)
        await message.answer(result.note)
        party.add("user", result.context, user_id=uid, char_name=character_label(actor))
        await _answer_with_dungeon_master(message, party, actor, uid)
        return

    if action == "prep":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context
        if not actor.is_spellcaster:
            await callback.answer(NOT_SPELLCASTER_TEXT, show_alert=True)
            return
        if is_spontaneous_caster(actor.class_name):
            await callback.answer(SPONTANEOUS_PREP_ALERT, show_alert=True)
            return

        if not param:
            await callback.answer()
            await message.answer(
                format_prep_menu(actor), reply_markup=build_prep_keyboard(actor, uid)
            )
            return

        index = int(param) if param.isdigit() else -1
        if not 0 <= index < len(actor.spells_known):
            await callback.answer(SPELL_MENU_STALE_TEXT, show_alert=True)
            return
        name = actor.spells_known[index]
        prepared = actor.toggle_prepared(name)
        party.save_character(uid)
        await callback.answer(f"{'✅ Заготовлено' if prepared else '❌ Снято'}: {name}")
        await message.answer(
            format_prep_menu(actor), reply_markup=build_prep_keyboard(actor, uid)
        )
        return

    # --- отдых ---
    if action == "rest":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context

        if not param:
            await callback.answer()
            await message.answer(
                format_rest_menu(actor), reply_markup=build_rest_keyboard(actor, uid)
            )
            return

        long_rest = param == "long"
        notes = actor.restore_spell_slots(long_rest=long_rest)
        if long_rest:
            actor.current_hp = actor.max_hp
            notes.insert(0, f"❤️ Продолжительный отдых: HP восстановлены до {actor.max_hp}.")

        party.save_character(uid)
        await callback.answer()

        title = "🌙 Продолжительный отдых (8 часов)" if long_rest else "☕ Короткий отдых (1 час)"
        body = "\n".join(f"• {note}" for note in notes) or "• Отдых прошёл спокойно."
        await message.answer(f"{title}\n{body}")

        kind = "продолжительный отдых (8 часов)" if long_rest else "короткий отдых (1 час)"
        party.add(
            "user",
            f"[СИСТЕМА] Герой {actor.name} отдыхает: {kind}.",
            user_id=uid,
            char_name=character_label(actor),
        )
        await _answer_with_dungeon_master(message, party, actor, uid)
        return

    # --- броски кубиков кнопками ---
    if action == "roll":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context

        if param in DAMAGE_DIE_SIDES:
            roll, label = roll_damage_die(actor, DAMAGE_DIE_SIDES[param])
            ctx = roll.context_message(f"бросок урона {roll.notation} ({label})")
        elif param == "damage":
            roll, label = roll_weapon_damage(actor)
            ctx = roll.context_message(f"бросок урона ({label})")
        elif param in D20_BUTTON_MODES:
            mode = D20_BUTTON_MODES[param]
            roll = make_roll(count=2 if mode else 1, sides=20, mode=mode)
            ctx = roll.context_message(D20_BUTTON_PURPOSES[param])
        else:
            await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
            return

        await callback.answer()
        await _resolve_roll(message, party, actor, uid, roll, context=ctx)
        return

    # --- проверка характеристики ---
    if action == "check":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context
        if param not in ABILITY_FULL_RU:
            await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
            return
        await callback.answer()
        modifier = actor.roll_modifier(param)
        roll = make_roll(count=1, sides=20, modifier=modifier)
        ctx = roll.context_message(
            f"проверка {ABILITY_GENITIVE_RU[param]} (мод. {format_modifier(modifier)})"
        )
        await _resolve_roll(message, party, actor, uid, roll, context=ctx)
        return

    # --- спасбросок ---
    if action == "save":
        context = await _require_actor(callback)
        if context is None:
            return
        message, uid, party, actor = context
        if param not in ABILITY_FULL_RU:
            await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)
            return
        await callback.answer()
        proficient = param in actor.save_proficiencies
        modifier = actor.save_modifier(param)
        roll = make_roll(count=1, sides=20, modifier=modifier)
        prof_note = "владение классом" if proficient else "без владения"
        ctx = roll.context_message(
            f"спасбросок {ABILITY_GENITIVE_RU[param]} (мод. {format_modifier(modifier)}, {prof_note})"
        )
        await _resolve_roll(message, party, actor, uid, roll, context=ctx)
        return

    await callback.answer(UNKNOWN_BUTTON_TEXT, show_alert=True)


# ---------------------------------------------------------------------------
# 18. ЗАПУСК БОТА
# ---------------------------------------------------------------------------

BOT_COMMANDS = [
    BotCommand(command="start", description="Справка и приглашение в отряд"),
    BotCommand(command="join", description="Создать героя: бот пришлёт кнопку перехода в ЛС"),
    BotCommand(command="party", description="Статус отряда: кто в группе и их HP"),
    BotCommand(command="sheet", description="Лист своего персонажа"),
    BotCommand(command="hero", description="Случайный герой 1-го уровня (PHB 2024)"),
    BotCommand(command="roll", description="Бросить кубик, например d20 или 2d6+3"),
    BotCommand(command="inventory", description="Снаряжение и золото"),
    BotCommand(command="check", description="Проверки характеристик с модификатором"),
    BotCommand(command="spells", description="Книга заклинаний: ячейки, применение, подготовка"),
    BotCommand(command="rest", description="Отдых: восстановить HP и ячейки заклинаний"),
    BotCommand(command="reset_party", description="Сбросить партию чата (администраторы)"),
]


async def main() -> None:
    """Точка входа: проверяет конфиг, поднимает polling и корректно всё закрывает."""
    global BOT_ID, BOT_USERNAME

    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "Не задан TELEGRAM_BOT_TOKEN.\n"
            "Создайте файл .env на основе .env.example и укажите токен от @BotFather."
        )
    if not LLM_API_KEY:
        raise SystemExit(
            "Не задан LLM_API_KEY.\n"
            "Создайте файл .env на основе .env.example и укажите ключ DeepSeek API."
        )

    # parse_mode=None: ответы Мастера — «сырой» текст, чтобы разметка модели не ломала отправку.
    bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))

    # Узнаём id и username бота: нужны фильтру адресации в группе и кнопкам-владельцам.
    me = await bot.get_me()
    BOT_ID = me.id
    BOT_USERNAME = me.username or ""
    logger.info("Бот: @%s (id %s)", BOT_USERNAME, BOT_ID)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    # Белый список и ограничение частоты: применяем к сообщениям и нажатиям кнопок.
    access_middleware = AccessAndRateLimitMiddleware()
    dispatcher.message.outer_middleware(access_middleware)
    dispatcher.callback_query.outer_middleware(access_middleware)
    logger.info(
        "Доступ: %s; лимит %d запросов за %.0f с (админов: %d).",
        "белый список" if ALLOWED_USER_IDS else "все пользователи",
        RATE_LIMIT_MAX_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS,
        len(ADMIN_USER_IDS),
    )

    # Готовим постоянное хранилище: создаём party_database.db, таблицы и индексы.
    db.init()

    try:
        await bot.set_my_commands(BOT_COMMANDS)
        logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        if llm_client is not None:
            await llm_client.close()
        await bot.session.close()
        db.close()
        logger.info("Соединения закрыты. До встречи в подземелье!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit) as exc:
        if str(exc):
            print(exc)


