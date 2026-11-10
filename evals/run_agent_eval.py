"""Прогін мультиагентного копілота на всіх спрацюваннях і порівняння конфігурацій моделей.

    python -m evals.run_agent_eval haiku routed sonnet opus gemini [--workers 6] [--limit 10]
    python -m evals.run_agent_eval routed --dataset eval_large        # 200 спрацювань

Пише evals/results/agent_<config>[_<dataset>].json (по кожному спрацюванню) і agent_summary.json / agent.md.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("FINMON_LOG_LEVEL", "WARNING")  # у прогоні оцінки лишаємо тільки попередження і помилки

from finmon.agents.graph import Copilot
from finmon.config import AGENT_CONFIGS, MODELS, RESULTS_DIR, SYNTH_DIR, TRACES_DIR
from finmon.rag import index as rag_index

from .metrics import decision_metrics

_lock = threading.Lock()
_orig_search = rag_index.Retriever.search


def _locked_search(self, *a, **kw):  # локальний Qdrant і ONNX-ембединги — по одному запиту за раз
    with _lock:
        return _orig_search(self, *a, **kw)


rag_index.Retriever.search = _locked_search


def run_config(config: str, alert_ids: list[str], labels: dict, workers: int, prompt: str = "v2", dataset: str = "eval") -> dict:
    copilot = Copilot(dataset, prompt_version=prompt)
    tag = config if prompt == "v2" else f"{config}_{prompt}"
    if dataset != "eval":
        tag = f"{tag}_{dataset}"
    results, failures = {}, {}
    t0 = time.perf_counter()

    def one(aid: str):
        return aid, copilot.run(aid, config)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, a) for a in alert_ids]
        for f in as_completed(futs):
            try:
                aid, res = f.result()
            except Exception as e:  # падіння одного спрацювання не зупиняє оцінку
                failures[str(e)[:80]] = failures.get(str(e)[:80], 0) + 1
                continue
            results[aid] = res
            fin = res["final"]
            ok = "✓" if fin["decision"] == labels[aid]["label"] else "✗"
            print(f"[{tag}] {aid} {ok} {fin['decision']:12s} (істина {labels[aid]['label']:12s}) ${fin['cost_usd']:.4f} {fin['latency_s']}s tools={fin['tool_calls']}", flush=True)
    wall = time.perf_counter() - t0

    done = sorted(results)
    y_true = [labels[a]["label"] for a in done]
    y_pred = [results[a]["final"]["decision"] for a in done]
    finals = [results[a]["final"] for a in done]
    ver = [f["verification"] for f in finals]
    m = decision_metrics(y_true, y_pred)
    m.update({
        "config": tag, "dataset": dataset, "prompt_version": prompt, "roles": {k: MODELS[v] for k, v in AGENT_CONFIGS[config].items()},
        "failed": sum(failures.values()), "failures": failures,
        "cost_total_usd": round(sum(f["cost_usd"] for f in finals), 3),
        "cost_per_alert_usd": round(statistics.mean(f["cost_usd"] for f in finals), 4),
        "latency_p50_s": round(statistics.median(f["latency_s"] for f in finals), 1),
        "tool_calls_avg": round(statistics.mean(f["tool_calls"] for f in finals), 1),
        "revision_rate": round(sum(f["revisions"] > 0 for f in finals) / len(finals), 3),
        "provider_fallbacks": sum(f.get("provider_fallbacks", 0) for f in finals),
        "reviewer_changed_decision": sum(f["decision"] != f["investigator_decision"] for f in finals),
        "evidence_grounded_rate": round(1 - sum(len(v["nonexistent_or_foreign_tx"]) + len(v["tx_not_retrieved_by_agent"]) for v in ver) / max(1, sum(v["evidence_tx_total"] for v in ver)), 3),
        "legal_refs_grounded_rate": round(1 - sum(len(v["legal_refs_not_from_search"]) for v in ver) / max(1, sum(v["legal_refs_total"] for v in ver)), 3),
        "searched_regulations_rate": round(sum(v["regulations_searched"] for v in ver) / len(ver), 3),
        "errors": [{"alert_id": a, "typology": labels[a]["typology"], "true": labels[a]["label"], "pred": results[a]["final"]["decision"]}
                   for a in done if results[a]["final"]["decision"] != labels[a]["label"]],
        "wall_clock_s": round(wall, 1),
    })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"agent_{tag}.json").write_text(json.dumps({"metrics": m, "alerts": {a: results[a]["final"] for a in done}}, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(TRACES_DIR / f"{tag}.jsonl", "w", encoding="utf-8") as fh:
        for a in done:
            fh.write(json.dumps({"alert_id": a, "usage": results[a]["usage"], "tool_log": results[a]["tool_log"], "triage": results[a]["triage"]}, ensure_ascii=False) + "\n")
    return m


def write_summary() -> None:
    rows = []
    for cfg in ("haiku", "routed", "sonnet", "opus", "gemini", "gemini_lite"):
        for tag in (f"{cfg}_v1", cfg, f"{cfg}_eval_large", f"{cfg}_dev", f"{cfg}_v3_dev", f"{cfg}_v3_eval_large"):
            p = RESULTS_DIR / f"agent_{tag}.json"
            if p.exists():
                rows.append(json.loads(p.read_text(encoding="utf-8"))["metrics"])
    base = RESULTS_DIR / "baseline.json"
    md = ["| Конфігурація (промпт) | Точність | Macro-F1 | Recall ESCALATE | Пропущені схеми | Хибні ескалації | Обґрунтованість посилань | $ / спрацювання | Час p50, с |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in rows:
        name = m['config'].replace('_v1', '').replace('_v3', '').replace('_eval_large', '').replace('_dev', '')
        ds = m.get('dataset', 'eval')
        word = 'спрацювання' if m['n'] % 10 in (2, 3, 4) and m['n'] % 100 not in (12, 13, 14) else 'спрацювань'
        n = '' if ds == 'eval' else f", {m['n']} {word}" + (", набір для налаштування" if ds == 'dev' else '')
        md.append(f"| {name} ({m.get('prompt_version', 'v2')}{n}) | {m['accuracy']:.1%} | {m['macro_f1']:.2f} | {m['escalate_recall']:.0%} | {m['missed_schemes']} | {m['false_escalations']} | "
                  f"{m['legal_refs_grounded_rate']:.0%} | {m['cost_per_alert_usd']:.3f} | {m['latency_p50_s']:.0f} |")
    if base.exists():
        for m in json.loads(base.read_text(encoding="utf-8")):
            md.append(f"| ML: {m['model'].replace('logistic_regression', 'логістична регресія').replace('gradient_boosting', 'градієнтний бустинг')} | {m['accuracy']:.1%} | {m['macro_f1']:.2f} | {m['escalate_recall']:.0%} | {m['missed_schemes']} | {m['false_escalations']} | — | ≈0 | <0.01 |")
    (RESULTS_DIR / "agent_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "agent.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=["haiku"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2", "v3"])
    ap.add_argument("--dataset", default="eval", choices=["eval", "eval_large", "dev"])
    a = ap.parse_args()
    labels = json.loads((SYNTH_DIR / a.dataset / "labels.json").read_text(encoding="utf-8"))
    ids = sorted(labels)[: a.limit or None]
    for cfg in a.configs:
        m = run_config(cfg, ids, labels, a.workers, a.prompt, a.dataset)
        print(json.dumps({k: m[k] for k in ("config", "accuracy", "macro_f1", "escalate_recall", "missed_schemes", "false_escalations", "cost_total_usd", "cost_per_alert_usd", "latency_p50_s", "failed")}, ensure_ascii=False))
    write_summary()


if __name__ == "__main__":
    main()
