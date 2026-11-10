"""Структуровані відповіді агентів (structured outputs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["ESCALATE", "REQUEST_INFO", "CLOSE"]
Level = Literal["low", "medium", "high"]


class Hypothesis(BaseModel):
    typology: str = Field(description="Коротка назва схеми або легального пояснення")
    why: str


class TriageResult(BaseModel):
    risk_level: Level
    hypotheses: list[Hypothesis] = Field(description="2–4 гіпотези: і підозрілі схеми, і легальні пояснення")
    plan: list[str] = Field(description="Що перевірити, по кроках")


class Indicator(BaseModel):
    description: str = Field(description="Що саме підозріло, з цифрами")
    evidence_tx_ids: list[str] = Field(description="ID транзакцій, які це підтверджують")
    legal_refs: list[str] = Field(description="chunk_id з search_regulations")


class InvestigationResult(BaseModel):
    indicators: list[Indicator]
    mitigating_factors: list[str] = Field(description="Факти, що пояснюють операції легально")
    missing_information: list[str] = Field(description="Яких документів чи даних бракує")
    proposed_decision: Decision
    confidence: Level
    rationale: str


class LegalBasis(BaseModel):
    chunk_id: str
    why: str


class ReviewResult(BaseModel):
    verdict: Literal["approve", "revise"]
    issues: list[str] = Field(description="Що не так з розслідуванням; порожньо, якщо approve")
    final_decision: Decision
    confidence: Level
    summary_ua: str = Field(description="3–5 речень для аналітика: що сталось, чому таке рішення")
    legal_basis: list[LegalBasis]
    next_steps: list[str]
