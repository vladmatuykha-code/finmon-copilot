"""Запуск копілота на одному спрацюванні: python cli.py A010 --config haiku"""
import argparse
import json

from finmon.agents.graph import Copilot

ap = argparse.ArgumentParser()
ap.add_argument("alert_id")
ap.add_argument("--config", default="opus")
ap.add_argument("--dataset", default="eval")
a = ap.parse_args()
res = Copilot(a.dataset).run(a.alert_id, a.config)
print(json.dumps(res["final"], ensure_ascii=False, indent=1))
for t in res["tool_log"]:
    print("tool:", t["tool"], json.dumps(t["args"], ensure_ascii=False)[:120])
