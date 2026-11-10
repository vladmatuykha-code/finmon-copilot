"""Оцінка пошуку: Recall@5 і MRR@10 для кожної стратегії нарізки та режиму пошуку.

    python -m evals.run_retrieval_eval            # усі стратегії і моделі
    python -m evals.run_retrieval_eval --quick    # лише робоча конфігурація, для CI (пише retrieval_quick.json)
"""
from __future__ import annotations

import json
import time

from finmon.config import EVAL_DIR, RESULTS_DIR
from finmon.rag.chunking import ref_matches
from finmon.rag.index import Retriever, _qdrant

import sys

CONFIGS = [  # (нарізка, модель ембедингів)
    ("fixed", "minilm"), ("structural_noctx", "minilm"), ("structural", "minilm"), ("structural_short", "minilm"),
    ("structural_noctx", "e5"), ("structural_short", "e5"),
]
MODES = ["bm25", "dense", "hybrid"]


def first_hit(results: list[dict], expected: list[str]) -> int | None:
    for rank, r in enumerate(results, 1):
        if any(ref_matches(e, r["refs"]) for e in expected):
            return rank
    return None


def main() -> None:
    qa = json.loads((EVAL_DIR / "retrieval_qa.json").read_text(encoding="utf-8"))
    quick = "--quick" in sys.argv
    only = {a for a in sys.argv[1:] if not a.startswith("--")}  # напр. "minilm" — лише конфігурації з цією моделлю
    clients: dict[str, object] = {}
    rows = []
    for strat, model in CONFIGS:
        if only and model not in only:
            continue
        if quick and (strat, model) != ("structural_short", "minilm"):
            continue
        if model not in clients:
            clients[model] = _qdrant(model)
        r = Retriever(strat, model, client=clients[model])
        for mode in MODES:
            if model != "minilm" and mode == "bm25":
                continue  # BM25 не залежить від моделі ембедингів
            t0 = time.perf_counter()
            hits5, rr = 0, 0.0
            for item in qa:
                rank = first_hit(r.search(item["q"], k=10, mode=mode), item["expected"])
                hits5 += rank is not None and rank <= 5
                rr += 1.0 / rank if rank else 0.0
            ms = (time.perf_counter() - t0) * 1000 / len(qa)
            row = {"strategy": strat, "embed": model, "mode": mode, "recall@5": round(hits5 / len(qa), 3), "mrr@10": round(rr / len(qa), 3), "ms_per_query": round(ms, 1)}
            rows.append(row)
            print(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if quick:
        (RESULTS_DIR / "retrieval_quick.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        for c in clients.values():
            c.close()
        return
    out = RESULTS_DIR / "retrieval.json"
    if only and out.exists():  # дописуємо до попередніх результатів
        prev = [r for r in json.loads(out.read_text(encoding="utf-8")) if r.get("embed") not in only]
        rows = prev + rows
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| Нарізка | Ембединги | Пошук | Recall@5 | MRR@10 | мс/запит |", "|---|---|---|---|---|---|"]
    md += [f"| {r['strategy']} | {r.get('embed', 'minilm')} | {r['mode']} | {r['recall@5']:.0%} | {r['mrr@10']:.2f} | {r['ms_per_query']} |" for r in rows]
    (RESULTS_DIR / "retrieval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for c in clients.values():
        c.close()


if __name__ == "__main__":
    main()
