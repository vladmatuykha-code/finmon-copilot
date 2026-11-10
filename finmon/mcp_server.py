"""MCP-сервер FinMon: інструменти розслідування для Claude Desktop, Claude Code та інших MCP-клієнтів.

Аналітик може розбирати спрацювання прямо в чаті з Claude: Claude сам викликає ті самі
інструменти, що й агент-розслідувач (анкета, транзакції, контрагенти, санкційні переліки,
пошук по нормативці), або запускає весь граф агентів одним інструментом.

Усі інструменти даних — лише на читання. Запуск:
    python -m finmon.mcp_server                       # stdio, набір eval
    FINMON_DATASET=eval_large python -m finmon.mcp_server

Підключення до Claude Code:
    claude mcp add finmon -- python -m finmon.mcp_server
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from .store import Bank
from .tools import ToolBox

DATASET = os.getenv("FINMON_DATASET", "eval")

mcp = FastMCP(
    "finmon",
    instructions=(
        "Інструменти помічника аналітика фінмоніторингу банку (синтетичні дані). "
        "Почни з list_alerts або get_alert, збери факти інструментами, знайди норму через "
        "search_regulations і посилайся лише на chunk_id, які реально отримав. "
        "Рішення: ESCALATE, REQUEST_INFO або CLOSE."
    ),
)


@lru_cache(maxsize=1)
def _bank() -> Bank:
    return Bank(DATASET)


def _call(name: str, args: dict) -> dict | list:
    """Той самий диспетчер, що в агента: однакова поведінка в графі й у MCP."""
    out, is_error = ToolBox(_bank(), client_id="").run(name, args)
    data = json.loads(out)
    if is_error:
        raise ValueError(data.get("error", "помилка інструмента"))
    return data


@mcp.tool()
def list_alerts() -> list[dict]:
    """Список спрацювань системи моніторингу: id, правило, клієнт."""
    b = _bank()
    return [{"alert_id": a["alert_id"], "rule_id": a["rule_id"], "rule_text": a["rule_text"],
             "client_id": a["client_id"], "client": b.clients[a["client_id"]]["name"]} for a in b.alerts.values()]


@mcp.tool()
def get_alert(alert_id: str) -> dict:
    """Повні дані одного спрацювання: правило, клієнт, ключові транзакції."""
    return _bank().alert(alert_id)


@mcp.tool()
def get_client_profile(client_id: str) -> dict:
    """Анкета клієнта (KYC): тип, професія, задекларований дохід, очікувана активність, документи в справі."""
    return _call("get_client_profile", {"client_id": client_id})


@mcp.tool()
def get_transactions(client_id: str, direction: str = "any", channel: str = "any", min_amount: float = 0, limit: int = 40) -> dict:
    """Транзакції клієнта. direction: in | out | any. channel: transfer, card_p2p, cash, swift, pos, any."""
    return _call("get_transactions", {"client_id": client_id, "direction": direction, "channel": channel,
                                      "min_amount": min_amount, "limit": limit})


@mcp.tool()
def get_account_stats(client_id: str) -> dict:
    """Агреговані показники рахунку: обороти, частка транзиту за 24 год, готівка, кількість контрагентів."""
    return _call("get_account_stats", {"client_id": client_id})


@mcp.tool()
def get_counterparty(counterparty_id: str) -> dict:
    """Дані контрагента: тип, країна, дата реєстрації, скільки клієнтів банку з ним працюють."""
    return _call("get_counterparty", {"counterparty_id": counterparty_id})


@mcp.tool()
def check_watchlists(name: str, birth_date: str = "", rnokpp: str = "") -> dict:
    """Перевірка за санкційними переліками і переліком публічних діячів."""
    return _call("check_watchlists", {"name": name, "birth_date": birth_date, "rnokpp": rnokpp})


@mcp.tool()
def get_online_banking_sessions(client_id: str) -> dict:
    """Входи в онлайн-банкінг: пристрої, IP, чи збігаються пристрої з іншими клієнтами."""
    return _call("get_online_banking_sessions", {"client_id": client_id})


@mcp.tool()
def get_atm_activity(atm_id: str, date: str = "") -> dict:
    """Зняття готівки в банкоматі: хто і скільки знімав (для пошуку кластерів)."""
    return _call("get_atm_activity", {"atm_id": atm_id, "date": date})


@mcp.tool()
def search_regulations(query: str) -> list[dict]:
    """Гібридний пошук (вектори + BM25 + RRF) по Закону 361-IX і Положенню НБУ №65. Повертає chunk_id, джерело, текст."""
    return _call("search_regulations", {"query": query})


@mcp.tool()
def investigate_alert(alert_id: str, config: str = "routed") -> dict:
    """Запускає весь граф агентів (triage → investigator → verify → reviewer) і повертає рішення зі звітом.
    Витрачає токени LLM: близько $0.04 на розбір у конфігурації routed."""
    from .agents.graph import Copilot  # важкий імпорт лише за потреби

    res = Copilot(DATASET).run(alert_id, config)
    return res["final"]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
