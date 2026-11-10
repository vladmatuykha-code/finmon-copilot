"""Спостережуваність: метрики Prometheus, структуровані JSON-логи і короткий підсумок для сторінки моніторингу.

Метрики віддає /metrics (формат Prometheus), підсумок за останні розбори — /stats і /monitor.
Правила алертів для Prometheus: deploy/prometheus/alerts.yml.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import statistics
import sys
import threading
import time
from collections import deque

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------- метрики
INVESTIGATIONS = Counter("finmon_investigations_total", "Завершені розбори спрацювань", ["config", "decision"])
INVESTIGATION_ERRORS = Counter("finmon_investigation_errors_total", "Розбори, що впали з помилкою", ["config", "error"])
INVESTIGATION_SECONDS = Histogram("finmon_investigation_seconds", "Час одного розбору", ["config"],
                                  buckets=(5, 10, 20, 30, 45, 60, 90, 120, 180, 300))
INVESTIGATION_COST = Histogram("finmon_investigation_cost_usd", "Вартість одного розбору, USD", ["config"],
                               buckets=(0.005, 0.01, 0.02, 0.035, 0.05, 0.1, 0.2, 0.3, 0.5))
GROUNDING_VIOLATIONS = Counter("finmon_grounding_violations_total", "Посилання, які не пройшли перевірку verify", ["kind"])
REVISIONS = Counter("finmon_revisions_total", "Висновки, повернуті рецензентом на доопрацювання", ["config"])

LLM_CALLS = Counter("finmon_llm_calls_total", "Виклики моделей", ["provider", "model", "node", "status"])
LLM_SECONDS = Histogram("finmon_llm_call_seconds", "Час одного виклику моделі", ["provider", "model"],
                        buckets=(0.5, 1, 2, 4, 8, 15, 30, 60, 120))
LLM_TOKENS = Counter("finmon_llm_tokens_total", "Токени", ["model", "kind"])
LLM_COST = Counter("finmon_llm_cost_usd_total", "Витрати на моделі, USD", ["model", "node"])
FALLBACKS = Counter("finmon_provider_fallbacks_total", "Переключення на резервну модель", ["from_model", "to_model", "reason"])

TOOL_CALLS = Counter("finmon_tool_calls_total", "Виклики інструментів агентом", ["tool", "status"])
TOOL_SECONDS = Histogram("finmon_tool_call_seconds", "Час виклику інструмента", ["tool"],
                         buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2))

# ---------------------------------------------------------------- логи
alert_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("alert_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)), "level": record.levelname,
                   "event": record.getMessage(), "alert_id": alert_ctx.get()}
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


log = logging.getLogger("finmon")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(JsonFormatter())
    log.addHandler(_h)
    log.setLevel(os.getenv("FINMON_LOG_LEVEL", "INFO").upper())
    log.propagate = False


def event(name: str, level: int = logging.INFO, **fields) -> None:
    log.log(level, name, extra={"fields": fields})


# ---------------------------------------------------------------- підсумок для /stats
class RecentStats:
    """Останні N розборів у пам'яті процесу: для сторінки моніторингу без Prometheus."""

    def __init__(self, size: int = 500):
        self.items: deque[dict] = deque(maxlen=size)
        self.fallbacks: deque[dict] = deque(maxlen=100)
        self.lock = threading.Lock()

    def add(self, item: dict) -> None:
        with self.lock:
            self.items.append(item)

    def add_fallback(self, item: dict) -> None:
        with self.lock:
            self.fallbacks.append(item)

    def summary(self) -> dict:
        with self.lock:
            items, fbs = list(self.items), list(self.fallbacks)
        ok = [i for i in items if i["status"] == "ok"]
        lat = sorted(i["latency_s"] for i in ok)

        def pct(p: float):
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None

        decisions: dict[str, int] = {}
        for i in ok:
            decisions[i["decision"]] = decisions.get(i["decision"], 0) + 1
        return {
            "investigations": len(items), "errors": len(items) - len(ok),
            "error_rate": round((len(items) - len(ok)) / len(items), 3) if items else 0.0,
            "latency_p50_s": pct(0.5), "latency_p95_s": pct(0.95),
            "cost_avg_usd": round(statistics.mean(i["cost_usd"] for i in ok), 4) if ok else None,
            "cost_total_usd": round(sum(i["cost_usd"] for i in ok), 3),
            "decisions": decisions,
            "grounding_violations": sum(i.get("grounding_violations", 0) for i in ok),
            "fallbacks": len(fbs), "last_fallbacks": fbs[-5:],
            "recent": items[-15:][::-1],
        }


RECENT = RecentStats()
