"""Перевірка якості перед релізом: падає з кодом 1, якщо метрики гірші за пороги в evals/thresholds.json.

    python -m evals.gate retrieval          # після run_retrieval_eval --quick
    python -m evals.gate agent routed       # після run_agent_eval routed
"""
from __future__ import annotations

import json
import sys

from finmon.config import EVAL_DIR, RESULTS_DIR


def check(name: str, ok: bool, detail: str, failures: list[str]) -> None:
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        failures.append(name)


def main() -> int:
    th = json.loads((EVAL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    kind = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    failures: list[str] = []
    if kind == "retrieval":
        t = th["retrieval"]
        path = RESULTS_DIR / "retrieval_quick.json"
        rows = json.loads((path if path.exists() else RESULTS_DIR / "retrieval.json").read_text(encoding="utf-8"))
        row = next(r for r in rows if r["strategy"] == t["strategy"] and r.get("embed", "minilm") == t["embed"] and r["mode"] == t["mode"])
        check("Recall@5 пошуку по нормативці", row["recall@5"] >= t["min_recall@5"], f"{row['recall@5']:.0%} (поріг {t['min_recall@5']:.0%})", failures)
    else:
        tag = sys.argv[2] if len(sys.argv) > 2 else "routed"
        m = json.loads((RESULTS_DIR / f"agent_{tag}.json").read_text(encoding="utf-8"))["metrics"]
        t = th["agent"]
        check("Точність рішень", m["accuracy"] >= t["min_accuracy"], f"{m['accuracy']:.1%} (поріг {t['min_accuracy']:.0%})", failures)
        check("Пропущені схеми", m["missed_schemes"] <= t["max_missed_schemes"], f"{m['missed_schemes']} (максимум {t['max_missed_schemes']})", failures)
        check("Хибні ескалації", m["false_escalations"] <= t["max_false_escalations"], f"{m['false_escalations']} (максимум {t['max_false_escalations']})", failures)
        check("Обґрунтованість посилань на норми", m["legal_refs_grounded_rate"] >= t["min_legal_refs_grounded_rate"],
              f"{m['legal_refs_grounded_rate']:.0%} (поріг {t['min_legal_refs_grounded_rate']:.0%})", failures)
        check("Вартість розбору", m["cost_per_alert_usd"] <= t["max_cost_per_alert_usd"], f"${m['cost_per_alert_usd']} (максимум ${t['max_cost_per_alert_usd']})", failures)
        check("Розбори без збоїв", m["failed"] == 0, f"впало {m['failed']}", failures)
    print("Перевірку пройдено" if not failures else f"Не пройдено: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
