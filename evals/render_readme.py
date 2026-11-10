"""Вставляє актуальні таблиці результатів у README між маркерами <!-- name:start --> і <!-- name:end -->."""
from __future__ import annotations

import re

from finmon.config import RESULTS_DIR, ROOT


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name in ("agent", "retrieval", "purpose_clf"):
        table = (RESULTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
        text = re.sub(rf"<!-- {name}:start -->.*?<!-- {name}:end -->", f"<!-- {name}:start -->\n{table}\n<!-- {name}:end -->", text, flags=re.S)
    readme.write_text(text, encoding="utf-8")
    print("README оновлено")


if __name__ == "__main__":
    main()
