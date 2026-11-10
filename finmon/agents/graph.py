"""Мультиагентний граф на LangGraph.

    load_context → triage → investigator ⇄ (інструменти) → verify → reviewer ─┬─ approve → finalize
                                  ▲                                          │
                                  └────────────── revise (до 1 разу) ────────┘

* triage       — LLM, структурована відповідь: гіпотези і план перевірки;
* investigator — LLM з інструментами (цикл ReAct): збирає факти, шукає в нормативці;
* verify       — без LLM: перевіряє, що кожна транзакція і кожне посилання на закон справжні;
* reviewer     — LLM, другий погляд: приймає висновок або повертає на доопрацювання;
* finalize     — без LLM: збирає звіт, вартість і час.
"""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .. import obs
from ..config import AGENT_CONFIGS, MAX_REVISIONS, MAX_TOOL_TURNS
from ..llm import LLM
from ..store import Bank, get_bank
from ..tools import TOOLS, ToolBox
from . import prompts
from .schemas import InvestigationResult, ReviewResult, TriageResult


class State(TypedDict, total=False):
    alert_id: str
    config: str
    alert: dict
    context: dict
    triage: dict
    investigation: dict
    verification: dict
    review: dict
    revisions: int
    feedback: list[str]
    rerun: bool
    usage: Annotated[list, operator.add]
    tool_log: Annotated[list, operator.add]
    final: dict
    started: float


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class Copilot:
    def __init__(self, dataset: str = "eval", llm: LLM | None = None, retriever_strategy: str = "structural_short", embed_model: str = "minilm", prompt_version: str = "v2"):
        self.prompt_version = prompt_version
        self.bank: Bank = get_bank(dataset)
        self.llm = llm or LLM()
        self.retriever_strategy = retriever_strategy
        self.embed_model = embed_model

    # ------------------------------------------------------------------
    def build(self, toolbox: ToolBox, roles: dict[str, str]):
        llm, bank = self.llm, self.bank
        P = {n: prompts.get(n, self.prompt_version) for n in ("triage", "investigator", "reviewer")}
        inv_messages: list = []

        def load_context(s: State) -> State:
            alert = bank.alert(s["alert_id"])
            prof = bank.profile(alert["client_id"])
            return {"alert": alert, "context": {"profile": prof, "account_stats": bank.features(alert["client_id"])}, "revisions": 0, "feedback": [], "started": time.perf_counter()}

        def triage(s: State) -> State:
            msg, u = llm.call(node="triage", model_key=roles["triage"], system=P["triage"], output_format=TriageResult, max_tokens=8000,
                              messages=[{"role": "user", "content": f"Спрацювання:\n{_j(s['alert'])}\n\nКлієнт і показники рахунку:\n{_j(s['context'])}"}])
            return {"triage": msg.parsed_output.model_dump(), "usage": [u.as_dict()]}

        def investigator(s: State) -> State:
            usage, start_log = [], len(toolbox.log)
            if not inv_messages:
                inv_messages.append({"role": "user", "content": (
                    f"Спрацювання:\n{_j(s['alert'])}\n\nГіпотези і план від агента первинного розбору:\n{_j(s['triage'])}\n\n"
                    f"client_id = {s['alert']['client_id']}. Розслідуй і дай висновок.")})
            else:
                inv_messages.append({"role": "user", "content": "Старший аналітик повернув висновок на доопрацювання:\n- " + "\n- ".join(s["feedback"]) + "\nДоперевір і дай оновлений висновок."})
            result = None
            for turn in range(MAX_TOOL_TURNS + 1):
                last = turn == MAX_TOOL_TURNS
                if last:
                    inv_messages.append({"role": "user", "content": "Ліміт викликів інструментів вичерпано. Дай висновок на основі зібраного."})
                msg, u = llm.call(node="investigator", model_key=roles["investigator"], system=P["investigator"],
                                  messages=inv_messages, tools=None if last else TOOLS, output_format=InvestigationResult)
                usage.append(u.as_dict())
                inv_messages.append({"role": "assistant", "content": msg.content})
                calls = [b for b in msg.content if b.type == "tool_use"]
                if msg.stop_reason == "tool_use" and calls:
                    results = []
                    for c in calls:
                        out, err = toolbox.run(c.name, c.input)
                        results.append({"type": "tool_result", "tool_use_id": c.id, "content": out, "is_error": err})
                    inv_messages.append({"role": "user", "content": results})
                    continue
                result = msg.parsed_output
                if result is not None:
                    break
                inv_messages.append({"role": "user", "content": "Дай висновок строго у форматі відповіді."})
            if result is None:
                raise RuntimeError("Розслідувач не дав висновку")
            return {"investigation": result.model_dump(), "usage": usage, "tool_log": toolbox.log[start_log:]}

        def verify(s: State) -> State:
            cid = s["alert"]["client_id"]
            bad_tx, unseen_tx, bad_refs, n_tx, n_refs = [], [], [], 0, 0
            for ind in s["investigation"]["indicators"]:
                for t in ind["evidence_tx_ids"]:
                    n_tx += 1
                    tx = bank.tx_by_id.get(t)
                    if not tx or tx["client_id"] != cid:
                        bad_tx.append(t)
                    elif t not in toolbox.seen_tx:
                        unseen_tx.append(t)
                for r in ind["legal_refs"]:
                    n_refs += 1
                    if r not in toolbox.seen_chunks:
                        bad_refs.append(r)
            return {"verification": {
                "evidence_tx_total": n_tx, "nonexistent_or_foreign_tx": bad_tx, "tx_not_retrieved_by_agent": unseen_tx,
                "legal_refs_total": n_refs, "legal_refs_not_from_search": bad_refs,
                "regulations_searched": bool(toolbox.seen_chunks), "tool_calls": len(toolbox.log),
            }}

        def reviewer(s: State) -> State:
            chunks = [{"chunk_id": k, "source": v["source"], "text": v["text"][:350]} for k, v in toolbox.seen_chunks.items()]
            content = (f"Спрацювання:\n{_j(s['alert'])}\n\nКлієнт і показники:\n{_j(s['context'])}\n\nГіпотези первинного розбору:\n{_j(s['triage'])}\n\n"
                       f"Висновок розслідувача:\n{_j(s['investigation'])}\n\nАвтоматична перевірка посилань:\n{_j(s['verification'])}\n\n"
                       f"Знайдені фрагменти нормативки:\n{_j(chunks)}\n\nДоопрацювань уже було: {s['revisions']} з {MAX_REVISIONS}.")
            msg, u = llm.call(node="reviewer", model_key=roles["reviewer"], system=P["reviewer"], output_format=ReviewResult,
                              messages=[{"role": "user", "content": content}])
            r = msg.parsed_output.model_dump()
            out: State = {"review": r, "usage": [u.as_dict()], "rerun": False}
            if r["verdict"] == "revise" and r["issues"] and s["revisions"] < MAX_REVISIONS:
                out.update(revisions=s["revisions"] + 1, feedback=r["issues"], rerun=True)
            return out

        def route_after_review(s: State) -> str:
            return "investigator" if s.get("rerun") else "finalize"

        def finalize(s: State) -> State:
            r = s["review"]
            basis = [b for b in r["legal_basis"] if b["chunk_id"] in toolbox.seen_chunks]
            for b in basis:
                b["source"] = toolbox.seen_chunks[b["chunk_id"]]["source"]
            return {"final": {
                "alert_id": s["alert_id"], "config": s["config"], "prompt_version": self.prompt_version, "decision": r["final_decision"], "confidence": r["confidence"],
                "summary_ua": r["summary_ua"], "legal_basis": basis, "next_steps": r["next_steps"],
                "indicators": s["investigation"]["indicators"], "mitigating_factors": s["investigation"]["mitigating_factors"],
                "investigator_decision": s["investigation"]["proposed_decision"], "revisions": s["revisions"],
                "verification": s["verification"],
                "cost_usd": round(sum(u["cost_usd"] for u in s["usage"]), 5),
                "latency_s": round(time.perf_counter() - s["started"], 1),
                "llm_calls": len(s["usage"]), "tool_calls": len(toolbox.log),
                "provider_fallbacks": sum(1 for u in s["usage"] if u.get("fallback_from")),
                "models_used": sorted({u["model"] for u in s["usage"]}),
            }}

        g = StateGraph(State)
        for name, fn in [("load_context", load_context), ("triage", triage), ("investigator", investigator),
                         ("verify", verify), ("reviewer", reviewer), ("finalize", finalize)]:
            g.add_node(name, fn)
        g.add_edge(START, "load_context")
        g.add_edge("load_context", "triage")
        g.add_edge("triage", "investigator")
        g.add_edge("investigator", "verify")
        g.add_edge("verify", "reviewer")
        g.add_conditional_edges("reviewer", route_after_review, {"investigator": "investigator", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------
    def run(self, alert_id: str, config: str = "opus") -> dict:
        roles = AGENT_CONFIGS[config]
        client_id = self.bank.alert(alert_id)["client_id"]
        token = obs.alert_ctx.set(alert_id)
        t0 = time.perf_counter()
        try:
            toolbox = ToolBox(self.bank, client_id, self.retriever_strategy, self.embed_model)
            graph = self.build(toolbox, roles)
            state = graph.invoke({"alert_id": alert_id, "config": config, "usage": [], "tool_log": []}, {"recursion_limit": 30})
        except Exception as e:
            obs.INVESTIGATION_ERRORS.labels(config, type(e).__name__).inc()
            obs.RECENT.add({"alert_id": alert_id, "config": config, "status": "error", "error": f"{type(e).__name__}: {str(e)[:160]}",
                            "latency_s": round(time.perf_counter() - t0, 1), "cost_usd": 0.0})
            obs.event("investigation_failed", obs.logging.ERROR, config=config, error=type(e).__name__, detail=str(e)[:300])
            raise
        finally:
            obs.alert_ctx.reset(token)
        fin = state["final"]
        v = fin["verification"]
        bad = {"foreign_tx": len(v["nonexistent_or_foreign_tx"]), "unseen_tx": len(v["tx_not_retrieved_by_agent"]), "legal_ref": len(v["legal_refs_not_from_search"])}
        for kind, n in bad.items():
            if n:
                obs.GROUNDING_VIOLATIONS.labels(kind).inc(n)
        obs.INVESTIGATIONS.labels(config, fin["decision"]).inc()
        obs.INVESTIGATION_SECONDS.labels(config).observe(fin["latency_s"])
        obs.INVESTIGATION_COST.labels(config).observe(fin["cost_usd"])
        if fin["revisions"]:
            obs.REVISIONS.labels(config).inc()
        obs.RECENT.add({"alert_id": alert_id, "config": config, "status": "ok", "decision": fin["decision"], "latency_s": fin["latency_s"],
                        "cost_usd": fin["cost_usd"], "grounding_violations": sum(bad.values()), "fallbacks": fin["provider_fallbacks"],
                        "ts": time.strftime("%H:%M:%S")})
        obs.event("investigation_done", alert_id=alert_id, config=config, decision=fin["decision"], latency_s=fin["latency_s"],
                  cost_usd=fin["cost_usd"], llm_calls=fin["llm_calls"], tool_calls=fin["tool_calls"], revisions=fin["revisions"],
                  grounding_violations=sum(bad.values()), provider_fallbacks=fin["provider_fallbacks"])
        return {"final": fin, "usage": state["usage"], "tool_log": state["tool_log"], "triage": state["triage"], "review": state["review"]}
