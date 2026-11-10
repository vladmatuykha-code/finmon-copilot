"""ML-бейзлайн: класичні моделі на числових ознаках рахунку замість LLM.

Навчання на окремому синтетичному наборі (240 спрацювань, інший seed), перевірка на тих самих 40,
що й агенти. Показує, що можна отримати без LLM і де LLM дає приріст.

    python -m evals.run_baseline
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finmon.config import RESULTS_DIR, SYNTH_DIR
from finmon.store import Bank

from .metrics import decision_metrics

NUMERIC = ["n_tx", "n_in", "n_out", "in_sum_uah", "out_sum_uah", "max_single_in_uah", "unique_senders", "unique_receivers",
           "p2p_in_share", "cash_in_uah", "cash_out_uah", "near_threshold_cash_ops", "threshold_ops", "pass_through_24h_share",
           "income_multiple", "international_in_share", "night_ops", "kw_gambling", "kw_loan_aid", "new_devices",
           "documents_on_file", "is_pep", "declared_monthly_income"]
RULES = ["R01", "R02", "R03", "R05", "R06", "R07", "R08", "R09", "R11", "R12", "R13", "R15"]
TYPES = ["individual", "fop", "legal_entity"]


def featurize(dataset: str):
    bank = Bank(dataset)
    labels = json.loads((SYNTH_DIR / dataset / "labels.json").read_text(encoding="utf-8"))
    X, y, ids, typ = [], [], [], []
    for aid in sorted(labels):
        a = bank.alert(aid)
        f = bank.features(a["client_id"])
        row = [float(f[k] if f[k] is not None else -1) for k in NUMERIC]
        row[NUMERIC.index("in_sum_uah")] = np.log1p(row[NUMERIC.index("in_sum_uah")])
        row[NUMERIC.index("out_sum_uah")] = np.log1p(row[NUMERIC.index("out_sum_uah")])
        row += [float(a["rule_id"] == r) for r in RULES] + [float(f["client_type"] == t) for t in TYPES]
        X.append(row)
        y.append(labels[aid]["label"])
        ids.append(aid)
        typ.append(labels[aid]["typology"])
    return np.array(X), np.array(y), ids, np.array(typ)


def main() -> None:
    Xtr, ytr, _, ttr = featurize("train")
    Xte, yte, ids, tte = featurize("eval")
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    rows = []
    for name, model in models.items():
        cv = cross_val_score(model, Xtr, ytr, cv=5, scoring="accuracy")
        model.fit(Xtr, ytr)
        pred = list(model.predict(Xte))
        m = decision_metrics(list(yte), pred)
        m.update({"model": name, "train_size": len(ytr), "cv_accuracy_train": round(float(cv.mean()), 3),
                  "errors": [{"alert_id": i, "true": t, "pred": p} for i, t, p in zip(ids, yte, pred) if t != p]})
        rows.append(m)
        print(name, {k: m[k] for k in ("accuracy", "macro_f1", "escalate_recall", "missed_schemes", "false_escalations", "cv_accuracy_train")})
        # Нова схема: модель не бачила цю типологію під час навчання (leave-one-typology-out).
        loto = list(yte)
        for t in sorted(set(tte)):
            mask_tr, mask_te = ttr != t, tte == t
            model.fit(Xtr[mask_tr], ytr[mask_tr])
            for i, p in zip(np.where(mask_te)[0], model.predict(Xte[mask_te])):
                loto[i] = p
        m2 = decision_metrics(list(yte), loto)
        m2.update({"model": f"{name} (нові схеми)", "train_size": len(ytr),
                   "errors": [{"alert_id": i, "typology": t, "true": a, "pred": b} for i, t, a, b in zip(ids, tte, yte, loto) if a != b]})
        rows.append(m2)
        print(name, "unseen typologies", {k: m2[k] for k in ("accuracy", "macro_f1", "escalate_recall", "missed_schemes", "false_escalations")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
