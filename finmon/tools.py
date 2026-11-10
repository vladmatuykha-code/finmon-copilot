"""Інструменти агента-розслідувача: схеми для Claude (strict tool use) і їх виконання."""
from __future__ import annotations

import json
import time

from . import obs
from .rag.index import get_retriever
from .store import Bank


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    {"name": "get_client_profile", "strict": True,
     "description": "Анкета клієнта (KYC): тип, вік, рід занять, задекларований дохід, очікувана активність, документи в справі, кредити, нотатки, статус публічного діяча.",
     "input_schema": _schema({"client_id": {"type": "string"}}, ["client_id"])},
    {"name": "get_transactions", "strict": True,
     "description": "Транзакції клієнта з фільтрами. direction: in|out|any. channel: any|card_p2p|cash_deposit|cash_withdrawal|cash_atm|transfer|swift|pos|acquiring|internal|loan_repayment. Повертає до limit найбільших операцій за сумою.",
     "input_schema": _schema({
         "client_id": {"type": "string"},
         "direction": {"type": "string", "enum": ["in", "out", "any"]},
         "channel": {"type": "string"},
         "min_amount": {"type": "number"},
         "limit": {"type": "integer"},
     }, ["client_id", "direction", "channel", "min_amount", "limit"])},
    {"name": "get_account_stats", "strict": True,
     "description": "Агреговані показники рахунку за 60 днів: обороти, кількість унікальних відправників і отримувачів, частка готівки і P2P, транзитність (частка коштів, що пішли за 24 год), кратність до задекларованого доходу, операції біля порогу 400 тис. грн, нічні операції, нові пристрої.",
     "input_schema": _schema({"client_id": {"type": "string"}}, ["client_id"])},
    {"name": "get_counterparty", "strict": True,
     "description": "Відомості про контрагента: тип, країна, дата реєстрації, КВЕД, скільки клієнтів банку з ним пов'язані.",
     "input_schema": _schema({"counterparty_id": {"type": "string"}}, ["counterparty_id"])},
    {"name": "check_watchlists", "strict": True,
     "description": "Перевірка особи за санкційними переліками: нечіткий збіг ПІБ і звірка дати народження та РНОКПП. Порожній рядок, якщо дата чи код невідомі.",
     "input_schema": _schema({"name": {"type": "string"}, "birth_date": {"type": "string"}, "rnokpp": {"type": "string"}}, ["name", "birth_date", "rnokpp"])},
    {"name": "get_online_banking_sessions", "strict": True,
     "description": "Сесії онлайн-банкінгу клієнта: пристрої, ОС, країна IP, нові пристрої.",
     "input_schema": _schema({"client_id": {"type": "string"}}, ["client_id"])},
    {"name": "get_atm_activity", "strict": True,
     "description": "Події в банкоматі: які картки знімали готівку, коли, звідки поповнені. date у форматі YYYY-MM-DD або порожній рядок.",
     "input_schema": _schema({"atm_id": {"type": "string"}, "date": {"type": "string"}}, ["atm_id", "date"])},
    {"name": "search_regulations", "strict": True,
     "description": "Пошук у Законі 361-IX і Положенні НБУ №65 (зокрема Додаток 19 «Критерії ризику» і Додаток 20 «Індикатори підозрілості»). Повертає фрагменти з chunk_id — посилайся на них у висновку. Формулюй запит описом ситуації.",
     "input_schema": _schema({"query": {"type": "string"}}, ["query"])},
]


class ToolBox:
    """Виконує виклики інструментів і запам'ятовує, що агент реально бачив (для перевірки обґрунтованості)."""

    def __init__(self, bank: Bank, client_id: str, retriever_strategy: str = "structural_short", embed_model: str = "minilm", k: int = 5):
        self.bank = bank
        self.client_id = client_id
        self.retriever = get_retriever(retriever_strategy, embed_model)
        self.k = k
        self.seen_chunks: dict[str, dict] = {}
        self.seen_tx: set[str] = set()
        self.log: list[dict] = []

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        t0 = time.perf_counter()
        try:
            out = self._dispatch(name, args)
            is_error = False
        except Exception as e:  # помилку повертаємо моделі як результат інструмента
            out, is_error = {"error": f"{type(e).__name__}: {e}"}, True
        dt = time.perf_counter() - t0
        self.log.append({"tool": name, "args": args, "ms": round(dt * 1000, 1), "error": is_error})
        obs.TOOL_CALLS.labels(name, "error" if is_error else "ok").inc()
        obs.TOOL_SECONDS.labels(name).observe(dt)
        return json.dumps(out, ensure_ascii=False), is_error

    def _dispatch(self, name: str, a: dict):
        b = self.bank
        if name == "get_client_profile":
            return b.profile(a["client_id"])
        if name == "get_transactions":
            res = b.transactions(a["client_id"], None if a["direction"] == "any" else a["direction"],
                                 None if a["channel"] in ("any", "") else a["channel"], a["min_amount"], limit=min(a["limit"] or 40, 60))
            self.seen_tx.update(t["tx_id"] for t in res["transactions"])
            return res
        if name == "get_account_stats":
            return b.features(a["client_id"])
        if name == "get_counterparty":
            return b.counterparty(a["counterparty_id"])
        if name == "check_watchlists":
            return b.watchlist_check(a["name"], a["birth_date"] or None, a["rnokpp"] or None)
        if name == "get_online_banking_sessions":
            return b.client_logins(a["client_id"])
        if name == "get_atm_activity":
            return b.atm(a["atm_id"], a["date"] or None)
        if name == "search_regulations":
            hits = self.retriever.search(a["query"], k=self.k, mode="hybrid")
            for h in hits:
                self.seen_chunks[h["chunk_id"]] = h
            return [{"chunk_id": h["chunk_id"], "source": h["source"], "text": h["text"][:1200]} for h in hits]
        raise ValueError(f"Невідомий інструмент {name}")
