"""Метрики якості рішень (спільні для агентів і ML-бейзлайну)."""
from __future__ import annotations

LABELS = ["ESCALATE", "REQUEST_INFO", "CLOSE"]


def decision_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
    f1s = {}
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}
    confusion = {t: {p: sum(a == t and b == p for a, b in zip(y_true, y_pred)) for p in LABELS} for t in LABELS}
    # Найдорожча помилка — пропустити схему: справжній ESCALATE закрито як CLOSE.
    missed = sum(t == "ESCALATE" and p == "CLOSE" for t, p in zip(y_true, y_pred))
    false_esc = sum(t == "CLOSE" and p == "ESCALATE" for t, p in zip(y_true, y_pred))
    return {
        "n": n, "accuracy": round(acc, 3),
        "macro_f1": round(sum(v["f1"] for v in f1s.values()) / len(LABELS), 3),
        "escalate_recall": f1s["ESCALATE"]["recall"], "escalate_precision": f1s["ESCALATE"]["precision"],
        "missed_schemes": missed, "false_escalations": false_esc,
        "per_class": f1s, "confusion": confusion,
    }
