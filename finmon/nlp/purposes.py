"""Класифікація призначень платежів (NLP).

Правила фінмоніторингу часто спираються на текст призначення платежу: «позика», «ставка», «USDT».
Ключові слова ламаються на суржику, скороченнях і помилках. Тут — категорії, базове правило на
ключових словах і класифікатор на дотренованій мовній моделі (XLM-RoBERTa + LoRA).

Навчання і порівняння моделей: python -m evals.run_purpose_clf
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import ROOT

CLASSES = {
    "salary": "Заробітна плата, аванс, премія, відпускні, лікарняні від роботодавця",
    "family": "Перекази рідним і друзям: подарунок, на день народження, кишенькові, допомога батькам",
    "loan": "Позика, поворотна фінансова допомога, повернення боргу або позики між людьми чи компаніями",
    "gambling": "Азартні ігри: ставки, казино, покер, букмекер, поповнення ігрового рахунку, виграш",
    "crypto": "Купівля чи продаж криптовалюти: USDT, біткоїн, P2P на біржі, обмін крипти",
    "goods_services": "Оплата товарів і послуг за рахунком, договором, актом, замовленням",
    "rent_utilities": "Оренда житла чи приміщення, комунальні послуги: світло, газ, вода, інтернет",
    "charity": "Благодійність і волонтерство: донат на ЗСУ, збір, благодійний внесок",
}
LABELS = list(CLASSES)
RISKY = ["gambling", "loan", "crypto"]  # категорії, що стосуються індикаторів НБУ №65

# Базове правило: перша категорія, ключове слово якої знайдено. Так працюють прості правила в банках.
KEYWORDS = {
    "gambling": r"ставк|казин|покер|букмек|bet|слот|рулетк|джекпот|виграш|ігров",
    "crypto": r"usdt|крипт|біткоїн|биткоин|bitcoin|btc|eth|binance|p2p|тезер|tether",
    "loan": r"позик|займ|заём|позич|борг|долг|поворотн|фінансова допомога|финансовая помощь",
    "charity": r"благод|донат|збір|сбор|зсу|всу|волонт|на дрон|на армію",
    "salary": r"зарплат|заробітн|заработн|аванс|премі|премия|відпускн|отпускн|лікарнян|больничн",
    "rent_utilities": r"оренд|аренд|комунал|коммунал|світло|свет|газ|вод[аи]|інтернет|интернет|квартплат",
    "family": r"мам|тат|пап|сестр|брат|бабус|дід|дед|подар|день народж|др\b|кишеньков|сину|доньц|дочк",
    "goods_services": r"рахун|счет|договір|договор|акт|товар|послуг|услуг|замовлен|заказ|оплата",
}
_KW = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORDS.items()}


def keyword_predict(text: str) -> str | None:
    for label, rx in _KW.items():
        if rx.search(text):
            return label
    return None


ADAPTER_DIR = ROOT / "models" / "purpose-lora"
BASE_MODEL = "FacebookAI/xlm-roberta-base"


class PurposeClassifier:
    """Дотренована XLM-RoBERTa з LoRA-адаптером. Ваги — models/purpose-lora (після python -m evals.run_purpose_clf)."""

    def __init__(self, adapter_dir: Path = ADAPTER_DIR, device: str | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).to(self.device).eval()

    def predict(self, texts: list[str], batch_size: int = 64) -> list[str]:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out += [LABELS[j] for j in self.model(**enc).logits.argmax(-1).tolist()]
        return out
