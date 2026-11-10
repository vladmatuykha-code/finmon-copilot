"""Класифікатор призначень платежів: ключові слова vs класичний ML vs дотренована мовна модель vs LLM.

* ключові слова — як прості правила в банку;
* TF-IDF + логістична регресія;
* XLM-RoBERTa base: повне дотренування (усі 278 млн параметрів) і LoRA (≈1% параметрів);
* Claude Haiku без навчання (zero-shot), для порівняння ціни.

Навчання на data/purposes/train.jsonl (Claude Haiku), 10% — валідація для вибору епохи.
Тест — data/purposes/test.jsonl (Gemini): «standard» і «hard» без очевидних ключових слів.

    python -m evals.run_purpose_clf [--skip-llm]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline, make_union
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from finmon.config import DATA_DIR, RESULTS_DIR, ROOT
from finmon.nlp.purposes import ADAPTER_DIR, BASE_MODEL, CLASSES, LABELS, RISKY, keyword_predict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
L2I = {l: i for i, l in enumerate(LABELS)}


def load(name: str) -> list[dict]:
    with open(DATA_DIR / "purposes" / f"{name}.jsonl", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def seed_all(s: int = 0) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def scores(y_true: list[str], y_pred: list[str | None], subset: list[str]) -> dict:
    y_pred = [p or "none" for p in y_pred]
    out = {"accuracy": round(accuracy_score(y_true, y_pred), 3), "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 3)}
    for part in ("standard", "hard"):
        idx = [i for i, s in enumerate(subset) if s == part]
        out[f"acc_{part}"] = round(accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx]), 3)
    out["risky_recall"] = round(recall_score(y_true, y_pred, labels=RISKY, average="macro", zero_division=0), 3)
    out["per_class_recall"] = {l: round(r, 3) for l, r in zip(LABELS, recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0))}
    return out


# ---------------------------------------------------------------- трансформер
def train_transformer(train: list[dict], val: list[dict], *, lora: bool, epochs: int, lr: float, bs: int = 32):
    seed_all(0)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
    if lora:
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])
        model = get_peft_model(model, cfg)
    model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = math.ceil(len(train) / bs) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, best_state, history = -1.0, None, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        for i in range(0, len(order), bs):
            batch = [train[j] for j in order[i:i + bs]]
            enc = tok([b["text"] for b in batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            labels = torch.tensor([L2I[b["label"]] for b in batch], device=DEVICE)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        pred = predict(model, tok, [v["text"] for v in val])
        f1 = f1_score([v["label"] for v in val], pred, labels=LABELS, average="macro")
        history.append({"epoch": ep + 1, "val_macro_f1": round(f1, 3), "loss_last_batch": round(loss.item(), 4)})
        print(f"  {'LoRA' if lora else 'full'} epoch {ep + 1}: val macro-F1 {f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items() if (not lora) or ("lora" in k or "classifier" in k)}
    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state, strict=False)
    return model, tok, {"trainable_params": trainable, "total_params": total, "train_seconds": round(train_s, 1), "epochs": history}


def predict(model, tok, texts: list[str], bs: int = 64) -> list[str]:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            out += [LABELS[j] for j in model(**enc).logits.argmax(-1).tolist()]
    return out


def timed_predict(model, tok, texts):
    if DEVICE == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    pred = predict(model, tok, texts)
    if DEVICE == "mps":
        torch.mps.synchronize()
    return pred, round((time.perf_counter() - t0) / len(texts) * 1000 * 1000, 1)  # мс на 1000 текстів


def dir_mb(path) -> float:
    return round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6, 1)


# ---------------------------------------------------------------- LLM без навчання
class Labeled(BaseModel):
    labels: list[str]


def llm_zero_shot(texts: list[str], model_key: str = "haiku", batch: int = 40):
    from finmon.llm import LLM
    llm = LLM()
    cats = "\n".join(f"- {k}: {v}" for k, v in CLASSES.items())
    sys_prompt = f"Класифікуй призначення банківських платежів. Категорії:\n{cats}\nДля кожного рядка поверни рівно одну назву категорії, в тому ж порядку."
    preds, cost = [], 0.0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        body = "\n".join(f"{n + 1}. {t}" for n, t in enumerate(chunk))
        msg, u = llm.call(node="triage", model_key=model_key, system=sys_prompt, output_format=Labeled, max_tokens=4000,
                          messages=[{"role": "user", "content": f"{len(chunk)} рядків:\n{body}"}])
        got = msg.parsed_output.labels
        got = (got + ["none"] * len(chunk))[:len(chunk)]
        preds += [g if g in L2I else "none" for g in got]
        cost += u.cost_usd
    return preds, round(cost / len(texts) * 1000, 3), round((time.perf_counter() - t0) / len(texts) * 1000 * 1000, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    a = ap.parse_args()
    data, test = load("train"), load("test")
    rng = random.Random(1)
    rng.shuffle(data)
    n_val = len(data) // 10
    val, train = data[:n_val], data[n_val:]
    X, y, sub = [t["text"] for t in test], [t["label"] for t in test], [t["subset"] for t in test]
    print(f"train={len(train)} val={len(val)} test={len(test)} device={DEVICE}")
    rows = []

    t0 = time.perf_counter()
    kw = [keyword_predict(x) for x in X]
    rows.append({"model": "Ключові слова", **scores(y, kw, sub), "trainable_params": 0, "train_seconds": 0, "ms_per_1000": round((time.perf_counter() - t0) / len(X) * 1e6, 1), "usd_per_1000": 0})

    tfidf = make_pipeline(make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                                     TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
                          LogisticRegression(max_iter=3000, C=8.0))
    t0 = time.perf_counter()
    tfidf.fit([t["text"] for t in train], [t["label"] for t in train])
    tr_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = list(tfidf.predict(X))
    rows.append({"model": "TF-IDF + логістична регресія", **scores(y, p, sub), "trainable_params": int(tfidf[-1].coef_.size), "train_seconds": round(tr_s, 1),
                 "ms_per_1000": round((time.perf_counter() - t0) / len(X) * 1e6, 1), "usd_per_1000": 0})

    for lora, epochs, lr in ((False, 4, 3e-5), (True, 8, 6e-4)):
        model, tok, info = train_transformer(train, val, lora=lora, epochs=epochs, lr=lr)
        p, ms = timed_predict(model, tok, X)
        name = "XLM-RoBERTa + LoRA" if lora else "XLM-RoBERTa, повне дотренування"
        row = {"model": name, **scores(y, p, sub), **{k: v for k, v in info.items()}, "ms_per_1000": ms, "usd_per_1000": 0}
        if lora:
            ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ADAPTER_DIR))
            row["adapter_mb"] = dir_mb(ADAPTER_DIR)
            row["errors_sample"] = [{"text": x, "true": t, "pred": q, "subset": s} for x, t, q, s in zip(X, y, p, sub) if t != q][:40]
        else:
            row["model_mb"] = round(info["total_params"] * 4 / 1e6, 1)
        rows.append(row)
        print({k: row[k] for k in ("model", "accuracy", "acc_standard", "acc_hard", "macro_f1", "risky_recall", "trainable_params", "train_seconds", "ms_per_1000")}, flush=True)
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    if not a.skip_llm:
        p, usd, ms = llm_zero_shot(X, "haiku")
        rows.append({"model": "Claude Haiku без навчання", **scores(y, p, sub), "trainable_params": 0, "train_seconds": 0, "ms_per_1000": ms, "usd_per_1000": usd})

    for r in rows:
        if r["model"] != "Claude Haiku без навчання":
            print({k: r[k] for k in ("model", "accuracy", "acc_standard", "acc_hard", "macro_f1", "risky_recall")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"train": len(train), "val": len(val), "test": len(test), "test_standard": sub.count("standard"), "test_hard": sub.count("hard"),
            "device": DEVICE, "base_model": BASE_MODEL, "torch": torch.__version__}
    (RESULTS_DIR / "purpose_clf.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| Модель | Точність | Звичайні | Складні | Macro-F1 | Recall ризикових | Навчається параметрів | Навчання, с | мс на 1000 | $ на 1000 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['accuracy']:.1%} | {r['acc_standard']:.1%} | {r['acc_hard']:.1%} | {r['macro_f1']:.2f} | {r['risky_recall']:.0%} | "
                  f"{r['trainable_params']:,} | {r['train_seconds']} | {r['ms_per_1000']} | {r['usd_per_1000']} |".replace(f"{r['trainable_params']:,}", f"{r['trainable_params']:,}".replace(",", " ")))
    (RESULTS_DIR / "purpose_clf.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
