"""Каскад: дотренована модель (LoRA) вирішує впевнені випадки локально, невпевнені йдуть у Claude Haiku.

Питання: скільки текстів можна не відправляти в LLM і яку точність це дає.
Потрібен адаптер models/purpose-lora (python -m evals.run_purpose_clf).

    python -m evals.run_purpose_cascade
"""
from __future__ import annotations

import json

import torch

from finmon.config import RESULTS_DIR
from finmon.nlp.purposes import LABELS, PurposeClassifier

from .run_purpose_clf import llm_zero_shot, load, scores


def main() -> None:
    test = load("test")
    X, y, sub = [t["text"] for t in test], [t["label"] for t in test], [t["subset"] for t in test]
    clf = PurposeClassifier()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            enc = clf.tok(X[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt").to(clf.device)
            probs += torch.softmax(clf.model(**enc).logits.float(), -1).cpu().tolist()
    local = [LABELS[max(range(len(p)), key=p.__getitem__)] for p in probs]
    conf = [max(p) for p in probs]
    llm_pred, usd_per_1000, _ = llm_zero_shot(X, "haiku")
    rows = []
    for thr in (0.0, 0.8, 0.9, 0.95, 0.98, 0.99, 1.01):
        pred = [l if c >= thr else m for l, c, m in zip(local, conf, llm_pred)]
        to_llm = sum(c < thr for c in conf) / len(conf)
        s = scores(y, pred, sub)
        rows.append({"threshold": thr, "share_to_llm": round(to_llm, 3), "usd_per_1000": round(usd_per_1000 * to_llm, 4),
                     **{k: s[k] for k in ("accuracy", "acc_standard", "acc_hard", "macro_f1", "risky_recall")}})
        print(rows[-1])
    per_item = [{"text": x, "true": t, "subset": s_, "lora": l, "lora_conf": round(c, 3), "haiku": m} for x, t, s_, l, c, m in zip(X, y, sub, local, conf, llm_pred)]
    (RESULTS_DIR / "purpose_cascade.json").write_text(json.dumps({"rows": rows, "items": per_item}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
