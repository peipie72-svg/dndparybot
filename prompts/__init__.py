"""Пакет prompts: тексты промптов Мастера, вынесенные из main.py.

Бот ведёт ТОЛЬКО классический D&D 2024 («Забытые Королевства») в групповых чатах
(партия из 2–6 игроков). Сеттинг Warcraft и выбор мира удалены (Фаза B).
"""
from __future__ import annotations

from .control_block import CONTROL_BLOCK_INSTRUCTIONS
from .creation import (
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
from .dm_prompt import BASE_DM_PROMPT
from .settings import DM_ROLL_INSTRUCTION

__all__ = [
    "BASE_DM_PROMPT",
    "CONTROL_BLOCK_INSTRUCTIONS",
    "DM_ROLL_INSTRUCTION",
    "HERO_ALREADY_CONFIRMED_ALERT",
    "HERO_ALREADY_CONFIRMED_TEXT",
    "HERO_CARD_QUESTION",
    "HERO_CARD_TITLE_CREATED",
    "HERO_CARD_TITLE_PENDING",
    "HERO_CARD_TITLE_RANDOM",
    "HERO_CREATION_PROMPT",
    "HERO_CREATION_START_PROMPT",
    "HERO_NOT_CREATED_ALERT",
]

