"""Генератор синтетичних даних банку: клієнти, транзакції, сесії онлайн-банкінгу, спрацювання правил.

Усі дані вигадані. Сценарії побудовані за індикаторами підозрілості з Додатка 20
до Положення НБУ №65 та статтями 11, 12, 20, 21 Закону 361-IX. Для кожного
спрацювання відома правильна відповідь (label), тому якість агентів можна виміряти.

    python -m finmon.data_gen            # набір для оцінки (40 спрацювань)
    python -m finmon.data_gen --train    # великий набір для ML-бейзлайну
    python -m finmon.data_gen --large    # 200 спрацювань для масштабної оцінки агентів
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import SYNTH_DIR

FIRST_M = ["Олександр", "Андрій", "Дмитро", "Сергій", "Максим", "Іван", "Богдан", "Тарас", "Віталій", "Роман", "Павло", "Юрій", "Остап", "Назар", "Денис"]
FIRST_F = ["Олена", "Ірина", "Наталія", "Оксана", "Марія", "Юлія", "Тетяна", "Анна", "Катерина", "Софія", "Дарина", "Вікторія", "Людмила", "Галина"]
LAST = ["Коваленко", "Бондаренко", "Ткаченко", "Шевченко", "Кравченко", "Мельник", "Олійник", "Савченко", "Руденко", "Лисенко", "Мороз", "Поліщук", "Гончар", "Марченко", "Левченко", "Тимошенко", "Захарченко", "Клименко", "Павленко", "Кузьменко", "Гнатюк", "Дячук", "Сидоренко", "Литвин"]
CITIES = ["Дніпро", "Київ", "Львів", "Харків", "Одеса", "Запоріжжя", "Вінниця", "Полтава"]
COMPANY_WORDS = ["Буд", "Трейд", "Агро", "Лайн", "Сервіс", "Груп", "Альфа", "Вест", "Нова", "Інвест", "Мега", "Фаворит", "Стандарт", "Прайм"]
START = datetime(2026, 9, 1)


@dataclass
class World:
    rng: random.Random
    clients: dict = field(default_factory=dict)
    transactions: list = field(default_factory=list)
    counterparties: dict = field(default_factory=dict)
    logins: list = field(default_factory=list)
    atm_events: list = field(default_factory=list)
    watchlist: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    n_client: int = 1000
    n_cp: int = 5000
    n_tx: int = 0

    # ---------- допоміжні генератори ----------
    def person_name(self, sex: str | None = None) -> str:
        sex = sex or self.rng.choice("MF")
        first = self.rng.choice(FIRST_M if sex == "M" else FIRST_F)
        last = self.rng.choice(LAST)
        if sex == "F" and last.endswith(("ий",)):
            last = last[:-2] + "а"
        return f"{last} {first}"

    def company_name(self) -> str:
        return f"ТОВ «{self.rng.choice(COMPANY_WORDS)}{self.rng.choice(COMPANY_WORDS).lower()}»"

    def date(self, day_from: int = 0, day_to: int = 60) -> datetime:
        d = START + timedelta(days=self.rng.randint(day_from, day_to))
        return d.replace(hour=self.rng.randint(8, 21), minute=self.rng.randint(0, 59))

    def new_client(self, **kw) -> str:
        self.n_client += 1
        cid = f"C{self.n_client}"
        base = {
            "client_id": cid,
            "type": "individual",
            "is_pep": False,
            "country": "UA",
            "city": self.rng.choice(CITIES),
            "account_opened": (START - timedelta(days=self.rng.randint(200, 2500))).date().isoformat(),
            "documents_on_file": ["Паспорт / ID-картка", "РНОКПП"],
            "notes": "",
        }
        base.update(kw)
        self.clients[cid] = base
        return cid

    def new_cp(self, name: str, kind: str, **kw) -> str:
        self.n_cp += 1
        cpid = f"K{self.n_cp}"
        self.counterparties[cpid] = {"counterparty_id": cpid, "name": name, "kind": kind, "country": "UA", **kw}
        return cpid

    def tx(self, cid: str, when: datetime, direction: str, amount: float, channel: str, cp: str | None, purpose: str, **kw) -> str:
        self.n_tx += 1
        tid = f"T{self.n_tx:06d}"
        cpd = self.counterparties.get(cp, {}) if cp else {}
        self.transactions.append({
            "tx_id": tid, "client_id": cid, "ts": when.isoformat(timespec="minutes"),
            "direction": direction, "amount_uah": round(amount, 2), "channel": channel,
            "counterparty_id": cp, "counterparty_name": cpd.get("name"), "counterparty_country": cpd.get("country"),
            "purpose": purpose, **kw,
        })
        return tid

    def background(self, cid: str, monthly_income: float, employer: str | None = None, n_spend: int = 25) -> None:
        """Звичайна активність: зарплата і витрати, щоб профіль виглядав живим."""
        emp = self.new_cp(employer or self.company_name(), "company", registered="2015-03-11")
        for m in (0, 30):
            self.tx(cid, START + timedelta(days=m + 4, hours=10), "in", monthly_income * self.rng.uniform(0.95, 1.05), "transfer", emp, "Заробітна плата")
        shops = [self.new_cp(n, "merchant") for n in ("АТБ-Маркет", "Сільпо", "Нова пошта", "Rozetka", "WOG", "Аптека Доброго Дня")]
        for _ in range(n_spend):
            self.tx(cid, self.date(), "out", self.rng.uniform(80, 2500), "pos", self.rng.choice(shops), "Оплата карткою")

    def add_alert(self, cid: str, rule: str, rule_text: str, label: str, typology: str, key_tx: list[str], refs: list[str], extra: dict | None = None) -> str:
        aid = f"A{len(self.alerts) + 1:03d}"
        self.alerts.append({
            "alert_id": aid, "client_id": cid, "rule_id": rule, "rule_text": rule_text,
            "created_at": (START + timedelta(days=61)).isoformat(timespec="minutes"),
            "window": {"from": START.date().isoformat(), "to": (START + timedelta(days=60)).date().isoformat()},
            **(extra or {}),
        })
        self.labels[aid] = {"label": label, "typology": typology, "key_tx": key_tx, "expected_refs": refs}
        return aid


# ---------------- СЦЕНАРІЇ: ESCALATE ----------------

def s_cash_structuring(w: World) -> None:
    sex = w.rng.choice("MF")
    cid = w.new_client(name=w.person_name(sex), type="individual", birth_year=w.rng.randint(1975, 1995),
                       occupation="Менеджер з продажу", declared_monthly_income=w.rng.choice([18000, 22000, 25000]),
                       expected_activity="Зарплатний проєкт, побутові витрати")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    branches = ["Відділення №12", "Відділення №4", "Відділення №27", "Відділення №9", "Відділення №31"]
    key = []
    day = w.rng.randint(15, 35)
    for i in range(w.rng.randint(4, 7)):
        key.append(w.tx(cid, START + timedelta(days=day + i * w.rng.randint(1, 3), hours=11), "in", w.rng.uniform(362000, 399500), "cash_deposit", None,
                        "Внесення готівки, власні кошти", branch=w.rng.choice(branches)))
    target = w.new_cp(w.person_name(), "person", country="UA")
    key.append(w.tx(cid, START + timedelta(days=day + 16, hours=15), "out", sum(t["amount_uah"] for t in w.transactions if t["tx_id"] in key) * 0.97, "transfer", target, "Поворотна фінансова допомога"))
    w.add_alert(cid, "R03", "Серія готівкових внесків 350–399 тис. грн за 14 днів (нижче порогу 400 тис.)", "ESCALATE", "cash_structuring", key,
                ["law:20", "law:21", "nbu:annex20"])


def s_money_mule(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(2003, 2007),
                       occupation="Студент", declared_monthly_income=w.rng.choice([0, 6000, 9000]),
                       expected_activity="Стипендія, перекази від батьків")
    key = []
    senders = [w.new_cp(w.person_name(), "person") for _ in range(w.rng.randint(28, 45))]
    exch = w.new_cp(w.rng.choice(["WhiteBIT (картковий шлюз)", "P2P-обмінник «CryptoFast»", "Картка 5168 **** 2291 (невідомий банк)"]), "unknown", country="UA")
    for d in range(w.rng.randint(5, 9)):
        base_day = START + timedelta(days=30 + d)
        incoming = 0.0
        for _ in range(w.rng.randint(5, 12)):
            amt = w.rng.uniform(1500, 14500)
            incoming += amt
            key.append(w.tx(cid, base_day + timedelta(hours=w.rng.randint(9, 20), minutes=w.rng.randint(0, 59)), "in", amt, "card_p2p", w.rng.choice(senders), w.rng.choice(["", "за товар", "борг", "+", "переказ"])))
        key.append(w.tx(cid, base_day + timedelta(hours=22), "out", incoming * 0.985, "card_p2p", exch, "переказ"))
    w.add_alert(cid, "R07", "Вхідні P2P-перекази від понад 25 унікальних відправників за 7 днів", "ESCALATE", "money_mule", key[:12] + key[-3:],
                ["nbu:annex20", "law:21", "law:11"])


def s_transit_company(w: World) -> None:
    reg = (START - timedelta(days=w.rng.randint(25, 80))).date().isoformat()
    cid = w.new_client(name=w.company_name(), type="legal_entity", registered=reg, employees=1,
                       director=w.person_name(), kved="46.73 Оптова торгівля деревиною, будівельними матеріалами",
                       expected_activity="Оптова торгівля будматеріалами, очікуваний оборот до 1 млн грн/міс",
                       documents_on_file=["Статут", "Витяг з ЄДР", "Паспорт директора"])
    payers = [w.new_cp(w.company_name(), "company", registered="2019-06-02") for _ in range(4)]
    receivers = [w.new_cp(w.rng.choice(["Delta Consulting LLP", "Orion Trade Ltd", "ТОВ «Консалт-Профі»", "Brightway Services LP"]), "company",
                          country=w.rng.choice(["UA", "GB", "CY", "AE"]), registered=(START - timedelta(days=w.rng.randint(30, 300))).date().isoformat()) for _ in range(2)]
    key = []
    for d in range(w.rng.randint(4, 7)):
        day = START + timedelta(days=20 + d * 4, hours=10)
        amt = w.rng.uniform(1.2e6, 3.8e6)
        key.append(w.tx(cid, day, "in", amt, "transfer", w.rng.choice(payers), "Оплата за будівельні матеріали згідно рахунку"))
        key.append(w.tx(cid, day + timedelta(hours=w.rng.randint(2, 6)), "out", amt * w.rng.uniform(0.95, 0.99), "transfer", w.rng.choice(receivers), "Оплата за консультаційні послуги"))
    w.add_alert(cid, "R11", "Транзит: вихідні платежі ≥90% вхідних протягом однієї доби, компанія зареєстрована < 6 міс", "ESCALATE", "transit_company", key,
                ["nbu:annex20", "law:21", "law:12"])


def s_third_party_online(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1945, 1955), occupation="Пенсіонер",
                       declared_monthly_income=w.rng.choice([4500, 6200, 8000]), expected_activity="Пенсія, комунальні платежі")
    w.background(cid, w.clients[cid]["declared_monthly_income"], employer="Пенсійний фонд України", n_spend=10)
    for i in range(w.rng.randint(5, 8)):
        w.logins.append({"client_id": cid, "ts": (START + timedelta(days=40 + i // 3, hours=w.rng.randint(0, 23))).isoformat(timespec="minutes"),
                         "device": f"device-{w.rng.randint(1000, 9999)}", "os": w.rng.choice(["Android 14", "Windows 11", "iOS 18", "Linux"]),
                         "ip_country": w.rng.choice(["UA", "PL", "NL", "RU", "DE", "TR"]), "new_device": True})
    key = []
    inc = w.new_cp(w.company_name(), "company")
    key.append(w.tx(cid, START + timedelta(days=41, hours=9), "in", w.rng.uniform(250000, 380000), "transfer", inc, "Повернення коштів"))
    for _ in range(w.rng.randint(3, 6)):
        key.append(w.tx(cid, START + timedelta(days=41, hours=w.rng.randint(10, 20)), "out", w.rng.uniform(30000, 70000), "card_p2p", w.new_cp(w.person_name(), "person"), "переказ"))
    w.add_alert(cid, "R15", "Вхід в онлайн-банкінг з 4+ нових пристроїв за 3 доби, IP з різних країн", "ESCALATE", "third_party_online_banking", key,
                ["nbu:annex20", "law:21"])


def s_illegal_gambling(w: World) -> None:
    cid = w.new_client(name=w.person_name("M"), type="individual", birth_year=w.rng.randint(1988, 1999), occupation="Не вказано",
                       declared_monthly_income=w.rng.choice([12000, 15000]), expected_activity="Особисті перекази")
    key = []
    players = [w.new_cp(w.person_name(), "person") for _ in range(w.rng.randint(30, 50))]
    for _ in range(w.rng.randint(45, 70)):
        key.append(w.tx(cid, w.date(25, 58), "in", w.rng.choice([200, 300, 500, 1000, 1500, 2000, 5000]), "card_p2p", w.rng.choice(players),
                        w.rng.choice(["поповнення", "депозит", "деп 777", "поповнення рахунку гри", ""])))
    for _ in range(w.rng.randint(12, 20)):
        key.append(w.tx(cid, w.date(25, 58), "out", w.rng.uniform(2000, 25000), "card_p2p", w.rng.choice(players), w.rng.choice(["вивід", "виграш", "вивід коштів"])))
    w.add_alert(cid, "R09", "Ключові слова «депозит/вивід/виграш» у призначенні P2P-переказів, >40 контрагентів", "ESCALATE", "illegal_gambling", key[:10] + key[-5:],
                ["law:20", "law:21", "nbu:annex20"])


def s_sanctions_hit(w: World) -> None:
    sanctioned = w.person_name("M")
    dob = f"19{w.rng.randint(60, 85)}-0{w.rng.randint(1, 9)}-1{w.rng.randint(0, 9)}"
    code = str(w.rng.randint(1000000000, 3999999999))
    w.watchlist.append({"name": sanctioned, "birth_date": dob, "rnokpp": code, "list": "Санкційний перелік РНБО (синтетичний)", "reason": "Фінансування збройної агресії"})
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1970, 1990), occupation="ФОП, ремонтні роботи",
                       declared_monthly_income=w.rng.choice([35000, 45000]), expected_activity="Розрахунки з замовниками ремонту")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    cp = w.new_cp(sanctioned, "person", birth_date=dob, rnokpp=code)
    key = [w.tx(cid, START + timedelta(days=w.rng.randint(30, 55), hours=13), "out", w.rng.uniform(80000, 260000), "transfer", cp, "Повернення боргу")]
    w.add_alert(cid, "R01", "Збіг контрагента з санкційним переліком (ПІБ + дата народження)", "ESCALATE", "sanctions_hit", key,
                ["law:22", "law:23", "law:21"])


def s_pep_unexplained(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1968, 1985), occupation="Депутат міської ради, заступник голови комісії з питань землекористування",
                       is_pep=True, declared_monthly_income=w.rng.choice([38000, 42000]), expected_activity="Зарплата, депутатські виплати")
    w.background(cid, w.clients[cid]["declared_monthly_income"], employer="Виконавчий комітет міської ради")
    co = w.new_cp(w.company_name(), "company", registered="2024-11-20", kved="68.10 Купівля та продаж власного нерухомого майна")
    key = [w.tx(cid, START + timedelta(days=w.rng.randint(20, 50), hours=12), "in", w.rng.uniform(1.8e6, 5.5e6), "transfer", co, w.rng.choice(["Повернення поворотної фінансової допомоги", "Оплата за консультаційні послуги"]))]
    key.append(w.tx(cid, START + timedelta(days=55, hours=12), "out", w.rng.uniform(1.0e6, 1.6e6), "cash_withdrawal", None, "Зняття готівки"))
    w.add_alert(cid, "R05", "Публічний діяч: надходження перевищують задекларований дохід у 10+ разів", "ESCALATE", "pep_unexplained", key,
                ["law:12", "law:11", "law:21"])


def s_loan_third_party(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1980, 1995), occupation="Водій",
                       declared_monthly_income=w.rng.choice([20000, 24000]), expected_activity="Зарплата, іпотечний кредит",
                       loans=[{"type": "іпотека", "amount": 1_600_000, "opened": "2025-02-10", "term_years": 15}])
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    key = []
    for _ in range(w.rng.randint(6, 10)):
        key.append(w.tx(cid, START + timedelta(days=w.rng.randint(40, 45), hours=w.rng.randint(9, 19)), "in", w.rng.uniform(120000, 240000), "card_p2p",
                        w.new_cp(w.person_name(), "person"), "переказ"))
    key.append(w.tx(cid, START + timedelta(days=46, hours=10), "out", 1_480_000, "loan_repayment", None, "Дострокове погашення іпотечного кредиту"))
    w.add_alert(cid, "R13", "Дострокове погашення кредиту коштами від третіх осіб", "ESCALATE", "loan_third_party_repayment", key,
                ["nbu:annex20", "law:21"])


def s_atm_cluster(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1995, 2004), occupation="Кур'єр",
                       declared_monthly_income=w.rng.choice([15000, 18000]), expected_activity="Зарплата, побутові витрати")
    atm = f"ATM-{w.rng.randint(100, 999)}"
    source = w.new_cp(w.person_name(), "person")
    key = []
    for d in range(w.rng.randint(3, 5)):
        day = START + timedelta(days=35 + d * 3, hours=23, minutes=w.rng.randint(0, 20))
        amt = w.rng.uniform(40000, 95000)
        key.append(w.tx(cid, day - timedelta(hours=3), "in", amt, "card_p2p", source, "переказ"))
        key.append(w.tx(cid, day, "out", amt * 0.99, "cash_atm", None, "Зняття готівки", atm_id=atm))
        for k in range(w.rng.randint(3, 5)):
            w.atm_events.append({"atm_id": atm, "ts": (day + timedelta(minutes=w.rng.randint(-6, 6))).isoformat(timespec="minutes"),
                                 "card": f"4149 **** {w.rng.randint(1000, 9999)}", "amount_uah": round(w.rng.uniform(40000, 95000), 2), "funded_by": source})
    w.add_alert(cid, "R12", f"Кластер зняття готівки: 3+ різні картки в {atm} протягом 10 хв, нічний час", "ESCALATE", "atm_cluster", key,
                ["nbu:annex20", "law:21"], extra={"atm_id": atm})


# ---------------- СЦЕНАРІЇ: CLOSE (хибні спрацювання) ----------------

def s_fop_coffee(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="fop", birth_year=w.rng.randint(1985, 1995), occupation="ФОП 2 група, кав'ярня",
                       kved="56.30 Обслуговування напоями", declared_monthly_income=w.rng.choice([180000, 240000]),
                       expected_activity="Еквайринг кав'ярні: 250–500 платежів на день по 60–250 грн; закупівля кави і молока",
                       documents_on_file=["Паспорт", "РНОКПП", "Виписка з ЄДР ФОП", "Договір еквайрингу"])
    acq = w.new_cp("Еквайринг mono (POS-термінал кав'ярні)", "acquiring")
    sup = [w.new_cp(n, "company", registered="2016-04-01") for n in ("ТОВ «Кава Роастерс»", "ТОВ «Молочний край»")]
    key = []
    for d in range(55):
        day = START + timedelta(days=d)
        key.append(w.tx(cid, day + timedelta(hours=22), "in", w.rng.uniform(6000, 11000), "acquiring", acq, f"Відшкодування еквайрингу за {day:%d.%m.%Y}"))
    for _ in range(8):
        w.tx(cid, w.date(), "out", w.rng.uniform(15000, 40000), "transfer", w.rng.choice(sup), "Оплата за товар згідно накладної")
    w.add_alert(cid, "R07", "Понад 50 вхідних надходжень на рахунок фізичної особи-підприємця за 60 днів", "CLOSE", "fp_fop_business", key[:6],
                ["law:7", "nbu:annex20"])


def s_car_sale(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1978, 1992), occupation="Інженер",
                       declared_monthly_income=w.rng.choice([45000, 55000]), expected_activity="Зарплата",
                       documents_on_file=["Паспорт", "РНОКПП", "Договір купівлі-продажу ТЗ Toyota RAV4 від 12.10.2026 на 650 000 грн (надано клієнтом)", "Свідоцтво про реєстрацію ТЗ на ім'я клієнта (2021)"])
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    buyer = w.new_cp(w.person_name(), "person")
    key = [w.tx(cid, datetime(2026, 10, 12, 15, 30), "in", 650000, "transfer", buyer, "Оплата за автомобіль Toyota RAV4 згідно договору купівлі-продажу від 12.10.2026")]
    w.add_alert(cid, "R02", "Разове надходження ≥400 тис. грн, нехарактерне для профілю клієнта", "CLOSE", "fp_car_sale", key, ["law:20"])


def s_salary_bonus(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1980, 1990), occupation="Головний інженер, ПрАТ «Дніпроважмаш»",
                       declared_monthly_income=w.rng.choice([95000, 110000]), expected_activity="Зарплатний проєкт ПрАТ «Дніпроважмаш»")
    emp = "ПрАТ «Дніпроважмаш»"
    w.background(cid, w.clients[cid]["declared_monthly_income"], employer=emp)
    empid = next(k for k, v in w.counterparties.items() if v["name"] == emp)
    key = [w.tx(cid, START + timedelta(days=35, hours=10), "in", w.rng.uniform(410000, 480000), "transfer", empid, "Виплата річної премії за 2025–2026 фінансовий рік, наказ №214-к")]
    w.add_alert(cid, "R02", "Разове надходження ≥400 тис. грн, нехарактерне для профілю клієнта", "CLOSE", "fp_salary_bonus", key, ["law:20"])


def s_family_abroad(w: World) -> None:
    cid = w.new_client(name=w.person_name("F"), type="individual", birth_year=w.rng.randint(1996, 2004), occupation="Вчителька",
                       declared_monthly_income=w.rng.choice([14000, 17000]), expected_activity="Зарплата, перекази від матері з Польщі",
                       notes="Мати клієнтки працює в Польщі з 2023 року, перекази щомісяця (зафіксовано під час оновлення анкети 2025).")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    mom = w.new_cp(w.person_name("F") + " (мати)", "person", country="PL")
    key = []
    for m in range(2):
        key.append(w.tx(cid, START + timedelta(days=m * 30 + 7, hours=12), "in", w.rng.uniform(9000, 18000), "swift", mom, "Family support / допомога родині"))
    w.add_alert(cid, "R06", "Регулярні міжнародні перекази на рахунок фізичної особи", "CLOSE", "fp_family_support", key, ["law:7"])


def s_it_freelancer(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="fop", birth_year=w.rng.randint(1990, 1999), occupation="ФОП 3 група, розробка ПЗ",
                       kved="62.01 Комп'ютерне програмування", declared_monthly_income=w.rng.choice([150000, 200000]),
                       expected_activity="Надходження від іноземних замовників через Payoneer/Upwork, $3–6 тис./міс",
                       documents_on_file=["Паспорт", "РНОКПП", "Виписка з ЄДР ФОП", "Договір з Upwork (Terms of Service)", "Податкові декларації ФОП 2025"])
    w.background(cid, 0, employer="(немає)", n_spend=20)
    pay = w.new_cp("Payoneer Inc. (Upwork Global)", "company", country="US")
    key = [w.tx(cid, START + timedelta(days=m * 30 + 15, hours=11), "in", w.rng.uniform(130000, 230000), "swift", pay, "Payment for software development services, Upwork contract") for m in range(2)]
    w.add_alert(cid, "R06", "Регулярні міжнародні надходження від іноземної компанії", "CLOSE", "fp_it_freelancer", key, ["law:7"])


def s_volunteer(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1985, 1998), occupation="Волонтер, ГО «Сила тилу»",
                       declared_monthly_income=w.rng.choice([25000, 30000]), expected_activity="Збори на потреби ЗСУ через «Банку», закупівля дронів і авто",
                       documents_on_file=["Паспорт", "РНОКПП", "Статут ГО «Сила тилу»", "Публічні звіти про збори (посилання в анкеті)"],
                       notes="Клієнт задекларував волонтерську діяльність під час оновлення анкети, звіти публікуються в соцмережах.")
    jar = w.new_cp("Банка «На FPV для 93 ОМБр»", "jar")
    sup = w.new_cp("ТОВ «Дрон Тех Україна»", "company", registered="2022-09-15", kved="26.30 Виробництво обладнання зв'язку")
    key = []
    for _ in range(w.rng.randint(5, 8)):
        key.append(w.tx(cid, w.date(10, 55), "in", w.rng.uniform(40000, 160000), "internal", jar, "Виведення коштів з Банки"))
    for i in range(3):
        key.append(w.tx(cid, w.date(15, 58), "out", w.rng.uniform(120000, 300000), "transfer", sup, f"Оплата за FPV-комплекти згідно рахунку №{w.rng.randint(100, 999)}"))
    w.add_alert(cid, "R07", "Значні регулярні надходження на рахунок фізичної особи без підтвердженого доходу", "CLOSE", "fp_volunteer", key, ["law:7"])


def s_name_similarity(w: World) -> None:
    real = w.person_name("M")
    w.watchlist.append({"name": real, "birth_date": "1961-04-17", "rnokpp": "2241507733", "list": "Санкційний перелік РНБО (синтетичний)", "reason": "Діяльність на ТОТ"})
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1985, 1995), occupation="Бухгалтер",
                       declared_monthly_income=w.rng.choice([30000, 35000]), expected_activity="Зарплата, перекази родичам")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    namesake = w.new_cp(real, "person", birth_date="1999-11-02", rnokpp="3640812219", note="Брат клієнта (вказано в анкеті як довірена особа)")
    w.clients[cid]["notes"] = f"Довірена особа: {real}, 1999 р.н. (брат)."
    key = [w.tx(cid, START + timedelta(days=44, hours=18), "out", w.rng.uniform(5000, 20000), "card_p2p", namesake, "на день народження")]
    w.add_alert(cid, "R01", "Нечіткий збіг ПІБ контрагента з санкційним переліком (схожість 92%)", "CLOSE", "fp_name_match", key, ["law:23"])


def s_student_trip(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(2004, 2006), occupation="Студент ДНУ, староста групи",
                       declared_monthly_income=w.rng.choice([3000, 5000]), expected_activity="Стипендія, перекази від батьків")
    friends = [w.new_cp(w.person_name(), "person") for _ in range(w.rng.randint(26, 32))]
    key = [w.tx(cid, START + timedelta(days=33, hours=w.rng.randint(10, 22)), "in", 3800, "card_p2p", f, w.rng.choice(["за Карпати", "Карпати, хатка", "поїздка", "за поїздку в Карпати"])) for f in friends]
    agency = w.new_cp("ФОП Гуцуляк (садиба «Смерекова хата»)", "company", registered="2018-05-14")
    key.append(w.tx(cid, START + timedelta(days=35, hours=12), "out", 3800 * len(friends), "transfer", agency, "Передоплата за проживання групи 27–30.10.2026"))
    w.add_alert(cid, "R07", "Вхідні P2P-перекази від понад 25 унікальних відправників за 7 днів", "CLOSE", "fp_group_payment", key[:8] + key[-1:], ["nbu:annex20"])


# ---------------- СЦЕНАРІЇ: REQUEST_INFO ----------------

def s_fin_aid_unknown(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1975, 1990), occupation="Не вказано",
                       declared_monthly_income=w.rng.choice([20000, 25000]), expected_activity="Особисті перекази")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    co = w.new_cp(w.company_name(), "company", registered="2017-03-03", kved="46.90 Неспеціалізована оптова торгівля")
    key = [w.tx(cid, START + timedelta(days=38, hours=11), "in", w.rng.uniform(700000, 950000), "transfer", co, "Поворотна фінансова допомога згідно договору")]
    w.add_alert(cid, "R02", "Разове надходження ≥400 тис. грн, нехарактерне для профілю клієнта", "REQUEST_INFO", "unexplained_inflow", key, ["law:11", "law:20"])


def s_real_estate_foreign(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="individual", birth_year=w.rng.randint(1970, 1985), occupation="Підприємець",
                       declared_monthly_income=w.rng.choice([60000, 80000]), expected_activity="Особисті витрати, планує купівлю нерухомості")
    w.background(cid, w.clients[cid]["declared_monthly_income"])
    co = w.new_cp(w.rng.choice(["Aster Holdings Ltd", "Meridian Capital Ltd"]), "company", country="CY")
    key = [w.tx(cid, START + timedelta(days=42, hours=12), "in", w.rng.uniform(4.6e6, 5.4e6), "swift", co, "Loan agreement / for apartment purchase")]
    w.add_alert(cid, "R06", "Надходження з-за кордону > 4 млн грн від компанії з Кіпру", "REQUEST_INFO", "source_of_funds_unclear", key, ["law:11", "law:12"])


def s_fop_spike(w: World) -> None:
    cid = w.new_client(name=w.person_name(), type="fop", birth_year=w.rng.randint(1980, 1992), occupation="ФОП 3 група, вантажні перевезення",
                       kved="49.41 Вантажний автомобільний транспорт", declared_monthly_income=w.rng.choice([120000, 150000]),
                       expected_activity="Оплати від 3–5 логістичних компаній, до 200 тис. грн/міс")
    buyer = w.new_cp(w.company_name(), "company", registered="2021-08-19", kved="52.29 Інша допоміжна діяльність у сфері транспорту")
    key = []
    for _ in range(w.rng.randint(6, 9)):
        key.append(w.tx(cid, w.date(30, 58), "in", w.rng.uniform(110000, 180000), "transfer", buyer, "Оплата за транспортні послуги згідно акту"))
    for _ in range(4):
        w.tx(cid, w.date(30, 58), "out", w.rng.uniform(40000, 90000), "transfer", w.new_cp("ТОВ «ОККО-Драйв»", "company"), "Паливні картки")
    w.add_alert(cid, "R08", "Оборот ФОП за місяць зріс у 5+ разів порівняно з середнім", "REQUEST_INFO", "turnover_spike", key, ["law:11", "law:7"])


ESCALATE = [s_cash_structuring, s_money_mule, s_transit_company, s_third_party_online, s_illegal_gambling, s_sanctions_hit, s_pep_unexplained, s_loan_third_party, s_atm_cluster]
CLOSE = [s_fop_coffee, s_car_sale, s_salary_bonus, s_family_abroad, s_it_freelancer, s_volunteer, s_name_similarity, s_student_trip]
REQUEST = [s_fin_aid_unknown, s_real_estate_foreign, s_fop_spike]


def build(seed: int, per_scenario: int) -> World:
    w = World(rng=random.Random(seed))
    scenarios = [(f, i) for f in ESCALATE + CLOSE + REQUEST for i in range(per_scenario)]
    w.rng.shuffle(scenarios)
    for f, _ in scenarios:
        f(w)
    return w


def save(w: World, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "clients": w.clients, "transactions": w.transactions, "counterparties": w.counterparties,
        "logins": w.logins, "atm_events": w.atm_events, "watchlist": w.watchlist, "alerts": w.alerts,
    }
    (out_dir / "bank.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(w.labels, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="великий набір для навчання ML-бейзлайну")
    ap.add_argument("--large", action="store_true", help="200 спрацювань для масштабної оцінки агентів (окремий seed)")
    ap.add_argument("--dev", action="store_true", help="40 спрацювань для налаштування промптів (окремий seed, не eval_large)")
    args = ap.parse_args()
    if args.dev:
        # усі типології по 2 + по 4 додаткові спірні випадки: саме на них налаштовується v3
        w = build(seed=31, per_scenario=2)
        for f in REQUEST:
            for _ in range(4):
                f(w)
        save(w, SYNTH_DIR / "dev")
    elif args.train:
        w = build(seed=7, per_scenario=12)
        save(w, SYNTH_DIR / "train")
    elif args.large:
        w = build(seed=2027, per_scenario=10)
        save(w, SYNTH_DIR / "eval_large")
    else:
        w = build(seed=2026, per_scenario=2)
        save(w, SYNTH_DIR / "eval")
    counts: dict[str, int] = {}
    for v in w.labels.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    print(f"alerts={len(w.alerts)} clients={len(w.clients)} tx={len(w.transactions)} labels={counts}")


if __name__ == "__main__":
    main()
