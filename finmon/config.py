"""Налаштування: шляхи, моделі, ціни."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = ROOT / "evals"
RESULTS_DIR = EVAL_DIR / "results"
TRACES_DIR = ROOT / "traces"

EMBED_MODEL = os.getenv("FINMON_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

MODELS = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
    "gemini": "gemini-3.8-flash",
    "gemini_lite": "gemini-3.5-flash-lite",
}


def provider_of(model: str) -> str:
    return "google" if model.startswith("gemini-") else "anthropic"

# USD за 1M токенів: (input, output, cache_read). Запис у кеш Anthropic = 1.25 × input.
# Ціни Gemini — платний тариф, діють до 31.12.2026 (aistudio.google.com/docs/pricing, перевірено 10.11.2026).
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
}

# Моделі, для яких вмикаємо серверний fallback при відмові (refusal).
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Конфігурації агентів: яка модель працює в якій ролі.
AGENT_CONFIGS: dict[str, dict[str, str]] = {
    "opus": {"triage": "opus", "investigator": "opus", "reviewer": "opus"},
    "sonnet": {"triage": "sonnet", "investigator": "sonnet", "reviewer": "sonnet"},
    "haiku": {"triage": "haiku", "investigator": "haiku", "reviewer": "haiku"},
    # Маршрутизація за вартістю: дешева модель збирає факти, сильніша перевіряє висновок.
    "routed": {"triage": "haiku", "investigator": "haiku", "reviewer": "sonnet"},
    # Другий провайдер — для порівняння ціни і якості та як резерв.
    "gemini": {"triage": "gemini", "investigator": "gemini", "reviewer": "gemini"},
    "gemini_lite": {"triage": "gemini_lite", "investigator": "gemini_lite", "reviewer": "gemini_lite"},
}

# Резервна модель іншого провайдера, якщо основна недоступна після повторів (збій мережі, 429, 5xx).
PROVIDER_FALLBACK = {"opus": "gemini", "sonnet": "gemini", "haiku": "gemini", "gemini": "haiku", "gemini_lite": "haiku"}
FALLBACK_ENABLED = os.getenv("FINMON_FALLBACK", "1") != "0"
DEFAULT_CONFIG = os.getenv("FINMON_CONFIG", "routed")  # найкращий баланс ціни і якості за оцінкою

# Рівень effort для моделей, що його підтримують.
EFFORT = {"triage": "low", "investigator": "medium", "reviewer": "medium"}  # у Gemini — thinking_level

MAX_TOOL_TURNS = 8
MAX_REVISIONS = 1
THRESHOLD_UAH = 400_000  # порогова фінансова операція, ст. 20 Закону 361-IX
