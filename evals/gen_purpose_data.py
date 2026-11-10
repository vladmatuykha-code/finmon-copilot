"""Дані для класифікатора призначень платежів.

Навчальний набір пише Claude Haiku, тестовий — Gemini з іншими інструкціями. Так класифікатор
перевіряється на чужому стилі письма, а не на манері тієї ж моделі. Тест має «складну» частину:
приклади без очевидних ключових слів.

    python -m evals.gen_purpose_data
"""
from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from finmon.config import DATA_DIR
from finmon.llm import LLM
from finmon.nlp.purposes import CLASSES

OUT = DATA_DIR / "purposes"


class Examples(BaseModel):
    items: list[str]


TRAIN_STYLES = [
    "офіційні призначення платежів у платіжних дорученнях і переказах між рахунками, українською і російською",
    "коментарі до переказу в мобільному банку, як пишуть звичайні люди: коротко, суржик, помилки, скорочення, трохи транслітом",
    "різні формати: з сумами, датами, номерами договорів чи рахунків, іменами; від 1–2 слів до довгих речень",
]
TEST_STYLES = {
    "standard": "реалістичні призначення платежів і коментарі до переказів, які можна побачити у виписці українського банку; українська, російська, суржик",
    "hard": "непрямі формулювання без найочевидніших слів категорії: сленг, евфемізми, назви сервісів і брендів, скорочення, помилки; але категорія має однозначно вгадуватися людиною",
}


def prompt(label: str, style: str, n: int) -> str:
    others = "; ".join(f"{k}: {v}" for k, v in CLASSES.items() if k != label)
    return (f"Згенеруй {n} різних прикладів призначення платежу для категорії «{label}»: {CLASSES[label]}.\n"
            f"Стиль: {style}.\nКожен приклад має однозначно належати цій категорії і не підходити до інших: {others}.\n"
            "Не повторюйся, не нумеруй, без лапок навколо прикладів. Імена, суми і номери вигадані.")


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def generate() -> None:
    llm = LLM()
    jobs = [("train", "haiku", lab, st, 100, "train") for lab in CLASSES for st in TRAIN_STYLES]
    jobs += [("test", "gemini", lab, TEST_STYLES[kind], 50, kind) for lab in CLASSES for kind in TEST_STYLES]

    def run(job):
        split, model, lab, style, n, kind = job
        msg, u = llm.call(node="datagen", model_key=model, system="Ти генеруєш реалістичні синтетичні дані для навчання класифікатора банківських операцій.",
                          messages=[{"role": "user", "content": prompt(lab, style, n)}], output_format=Examples, max_tokens=12000)
        return split, lab, kind, msg.parsed_output.items, u.cost_usd

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run, jobs))
    cost = sum(r[4] for r in results)
    train, test, seen = [], [], set()
    for split, lab, kind, items, _ in results:
        if split != "train":
            continue
        for t in items:
            k = norm(t)
            if t.strip() and k not in seen:
                seen.add(k)
                train.append({"text": t.strip(), "label": lab})
    dropped = 0
    for split, lab, kind, items, _ in results:
        if split != "test":
            continue
        for t in items:
            k = norm(t)
            if not t.strip() or k in seen:  # тест не перетинається з навчанням
                dropped += 1
                continue
            seen.add(k)
            test.append({"text": t.strip(), "label": lab, "subset": kind})
    random.Random(0).shuffle(train)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} test={len(test)} (видалено з тесту через збіг з навчанням: {dropped}) cost=${cost:.3f}")


if __name__ == "__main__":
    generate()
