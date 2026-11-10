"""MCP-сервер: справжній stdio-клієнт піднімає сервер, бачить інструменти і отримує дані."""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"list_alerts", "get_alert", "get_client_profile", "get_transactions", "get_account_stats",
            "get_counterparty", "check_watchlists", "get_online_banking_sessions", "get_atm_activity",
            "search_regulations", "investigate_alert"}


async def _session_check() -> tuple[set[str], list, dict]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "finmon.mcp_server"], cwd=str(ROOT),
                                   env={**os.environ, "FINMON_DATASET": "eval", "FINMON_LOG_LEVEL": "WARNING"})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            alerts = json.loads((await session.call_tool("list_alerts", {})).content[0].text)
            alerts = alerts if isinstance(alerts, list) else [alerts]
            cid = alerts[0]["client_id"]
            profile = json.loads((await session.call_tool("get_client_profile", {"client_id": cid})).content[0].text)
            return tools, alerts, profile


def test_mcp_server_exposes_tools_and_data():
    tools, alerts, profile = asyncio.run(_session_check())
    assert EXPECTED <= tools
    assert len(alerts) >= 1 and "alert_id" in alerts[0]
    assert profile["client_id"] == alerts[0]["client_id"]
