"""Доступ до синтетичної «банківської бази» і розрахунок ознак клієнта.

Ознаки (features) використовуються і агентом (як стислий огляд), і ML-бейзлайном.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache

from .config import SYNTH_DIR, THRESHOLD_UAH

CASH_IN = {"cash_deposit"}
CASH_OUT = {"cash_withdrawal", "cash_atm"}


class Bank:
    def __init__(self, dataset: str = "eval"):
        data = json.loads((SYNTH_DIR / dataset / "bank.json").read_text(encoding="utf-8"))
        self.clients: dict = data["clients"]
        self.counterparties: dict = data["counterparties"]
        self.alerts = {a["alert_id"]: a for a in data["alerts"]}
        self.watchlist: list = data["watchlist"]
        self.logins: list = data["logins"]
        self.atm_events: list = data["atm_events"]
        self.tx_by_client: dict[str, list] = {}
        self.tx_by_id: dict[str, dict] = {}
        for t in data["transactions"]:
            self.tx_by_client.setdefault(t["client_id"], []).append(t)
            self.tx_by_id[t["tx_id"]] = t
        for lst in self.tx_by_client.values():
            lst.sort(key=lambda t: t["ts"])

    # ---------- запити ----------
    def alert(self, alert_id: str) -> dict:
        return self.alerts[alert_id]

    def profile(self, client_id: str) -> dict:
        c = dict(self.clients[client_id])
        if "birth_year" in c:
            c["age"] = 2026 - c["birth_year"]
        return c

    def transactions(self, client_id: str, direction: str | None = None, channel: str | None = None,
                     min_amount: float = 0, date_from: str | None = None, date_to: str | None = None, limit: int = 40) -> dict:
        txs = self.tx_by_client.get(client_id, [])
        sel = [t for t in txs
               if (not direction or t["direction"] == direction)
               and (not channel or t["channel"] == channel)
               and t["amount_uah"] >= min_amount
               and (not date_from or t["ts"][:10] >= date_from)
               and (not date_to or t["ts"][:10] <= date_to)]
        sel_sorted = sorted(sel, key=lambda t: -t["amount_uah"])[:limit]
        sel_sorted.sort(key=lambda t: t["ts"])
        return {"total_matching": len(sel), "returned": len(sel_sorted), "note": "Повертаються найбільші за сумою, впорядковані за часом" if len(sel) > limit else "",
                "transactions": [{k: v for k, v in t.items() if v is not None and k != "client_id"} for t in sel_sorted]}

    def counterparty(self, cp_id: str) -> dict:
        cp = self.counterparties.get(cp_id)
        if not cp:
            return {"error": f"Контрагента {cp_id} не знайдено"}
        n_clients = len({t["client_id"] for lst in self.tx_by_client.values() for t in lst if t["counterparty_id"] == cp_id})
        return {**cp, "clients_of_bank_linked": n_clients}

    def client_logins(self, client_id: str) -> dict:
        ls = [l for l in self.logins if l["client_id"] == client_id]
        return {"sessions": ls, "unique_devices": len({l["device"] for l in ls}), "ip_countries": sorted({l["ip_country"] for l in ls}),
                "note": "" if ls else "Нових пристроїв і нетипових сесій не зафіксовано"}

    def atm(self, atm_id: str, date: str | None = None) -> dict:
        ev = [e for e in self.atm_events if e["atm_id"] == atm_id and (not date or e["ts"][:10] == date)]
        return {"atm_id": atm_id, "events": ev[:40], "unique_cards": len({e["card"] for e in ev}),
                "funding_sources": Counter(e["funded_by"] for e in ev).most_common(3)}

    def watchlist_check(self, name: str, birth_date: str | None = None, rnokpp: str | None = None) -> dict:
        res = []
        for w in self.watchlist:
            score = SequenceMatcher(None, name.lower(), w["name"].lower()).ratio()
            if score < 0.8:
                continue
            res.append({"listed_name": w["name"], "list": w["list"], "name_similarity": round(score, 2),
                        "birth_date_match": None if not birth_date else birth_date == w["birth_date"],
                        "rnokpp_match": None if not rnokpp else rnokpp == w["rnokpp"],
                        "listed_birth_date": w["birth_date"], "reason": w["reason"]})
        return {"query": name, "matches": res, "note": "" if res else "Збігів не знайдено"}

    # ---------- ознаки ----------
    def features(self, client_id: str) -> dict:
        txs = self.tx_by_client.get(client_id, [])
        c = self.clients[client_id]
        ins = [t for t in txs if t["direction"] == "in"]
        outs = [t for t in txs if t["direction"] == "out"]
        in_sum = sum(t["amount_uah"] for t in ins)
        out_sum = sum(t["amount_uah"] for t in outs)
        declared = c.get("declared_monthly_income") or 0
        # частка надходжень, які пішли з рахунку протягом 24 год (транзит)
        passed = 0.0
        for t in ins:
            t0 = datetime.fromisoformat(t["ts"])
            nxt = [o for o in outs if t0 <= datetime.fromisoformat(o["ts"]) <= t0 + timedelta(hours=24)]
            passed += min(t["amount_uah"], sum(o["amount_uah"] for o in nxt))
        purposes = " ".join((t.get("purpose") or "").lower() for t in txs)
        return {
            "n_tx": len(txs),
            "n_in": len(ins),
            "n_out": len(outs),
            "in_sum_uah": round(in_sum),
            "out_sum_uah": round(out_sum),
            "max_single_in_uah": round(max((t["amount_uah"] for t in ins), default=0)),
            "unique_senders": len({t["counterparty_id"] for t in ins if t["counterparty_id"]}),
            "unique_receivers": len({t["counterparty_id"] for t in outs if t["counterparty_id"]}),
            "p2p_in_share": round(sum(t["amount_uah"] for t in ins if t["channel"] == "card_p2p") / in_sum, 2) if in_sum else 0,
            "cash_in_uah": round(sum(t["amount_uah"] for t in ins if t["channel"] in CASH_IN)),
            "cash_out_uah": round(sum(t["amount_uah"] for t in outs if t["channel"] in CASH_OUT)),
            "near_threshold_cash_ops": sum(1 for t in txs if t["channel"] in CASH_IN | CASH_OUT and 0.85 * THRESHOLD_UAH <= t["amount_uah"] < THRESHOLD_UAH),
            "threshold_ops": sum(1 for t in txs if t["amount_uah"] >= THRESHOLD_UAH),
            "pass_through_24h_share": round(passed / in_sum, 2) if in_sum else 0,
            "income_multiple": round(in_sum / (2 * declared), 1) if declared else None,
            "international_in_share": round(sum(t["amount_uah"] for t in ins if t["channel"] == "swift") / in_sum, 2) if in_sum else 0,
            "night_ops": sum(1 for t in txs if int(t["ts"][11:13]) >= 22 or int(t["ts"][11:13]) < 6),
            "kw_gambling": sum(purposes.count(k) for k in ("депозит", "деп ", "вивід", "виграш", "поповнення рахунку гри")),
            "kw_loan_aid": sum(purposes.count(k) for k in ("фінансова допомога", "фінансової допомоги", "loan agreement")),
            "new_devices": len({l["device"] for l in self.logins if l["client_id"] == client_id}),
            "documents_on_file": len(c.get("documents_on_file", [])),
            "is_pep": int(bool(c.get("is_pep"))),
            "client_type": c.get("type"),
            "declared_monthly_income": declared,
        }


@lru_cache(maxsize=2)
def get_bank(dataset: str = "eval") -> Bank:
    return Bank(dataset)
