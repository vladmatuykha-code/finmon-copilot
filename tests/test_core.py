"""Тести без звернень до API: нарізка, ознаки, перевірка посилань і маршрутизація графа з фейковою моделлю."""
from __future__ import annotations

from types import SimpleNamespace

from finmon.agents.graph import Copilot
from finmon.agents.schemas import InvestigationResult, ReviewResult, TriageResult
from finmon.llm import Usage
from finmon.rag.chunking import law_leaves, load_law, ref_matches
from finmon.store import get_bank


def test_ref_matching():
    assert ref_matches("nbu:annex20:p68", ["nbu:annex20:III:p68"])
    assert not ref_matches("nbu:annex20:p68", ["nbu:annex20:III:p6"])
    assert ref_matches("law:art20", ["law:art20:ch1"])


def test_law_structure():
    refs = {lf.ref.split(":")[1] for lf in law_leaves(load_law())}
    assert {"art11", "art12", "art20", "art21", "art23"} <= refs


def test_features_detect_structuring():
    bank = get_bank("eval")
    aid = next(a for a, v in bank.alerts.items() if v["rule_id"] == "R03")
    f = bank.features(bank.alert(aid)["client_id"])
    assert f["near_threshold_cash_ops"] >= 3


class FakeLLM:
    """Відтворює поведінку моделі: розслідувач викликає один інструмент, рецензент спершу повертає на доопрацювання."""

    def __init__(self, tx_id: str):
        self.tx_id = tx_id
        self.calls: list[str] = []
        self.inv_turn = 0

    def call(self, *, node, model_key, system, messages, tools=None, output_format=None, max_tokens=0):
        self.calls.append(node)
        u = Usage(node, "fake", 10, 10, 0, 0, 0.0, 0.0, "end_turn")
        if node == "triage":
            return SimpleNamespace(stop_reason="end_turn", content=[], parsed_output=TriageResult(risk_level="high", hypotheses=[], plan=["перевірити готівку"])), u
        if node == "investigator":
            self.inv_turn += 1
            if self.inv_turn in (1, 3):
                block = SimpleNamespace(type="tool_use", id=f"t{self.inv_turn}", name="search_regulations", input={"query": "готівка нижче порогу"})
                return SimpleNamespace(stop_reason="tool_use", content=[block], parsed_output=None), u
            ref = next(iter(self.toolbox.seen_chunks))
            res = InvestigationResult(indicators=[{"description": "дроблення", "evidence_tx_ids": [self.tx_id, "T999999"], "legal_refs": [ref, "fake-chunk"]}],
                                      mitigating_factors=[], missing_information=[], proposed_decision="ESCALATE", confidence="high", rationale="...")
            return SimpleNamespace(stop_reason="end_turn", content=[], parsed_output=res), u
        verdict = "revise" if self.calls.count("reviewer") == 1 else "approve"
        res = ReviewResult(verdict=verdict, issues=["доперевір"] if verdict == "revise" else [], final_decision="ESCALATE", confidence="high",
                           summary_ua="...", legal_basis=[{"chunk_id": "fake-chunk", "why": "..."}], next_steps=[])
        return SimpleNamespace(stop_reason="end_turn", content=[], parsed_output=res), u


def test_graph_revision_loop_and_grounding(monkeypatch):
    cp = Copilot("eval")
    aid = next(a for a, v in cp.bank.alerts.items() if v["rule_id"] == "R03")
    cid = cp.bank.alert(aid)["client_id"]
    tx = cp.bank.tx_by_client[cid][0]["tx_id"]
    fake = FakeLLM(tx)
    cp.llm = fake
    from finmon import tools as tools_mod
    orig = tools_mod.ToolBox.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        fake.toolbox = self
    monkeypatch.setattr(tools_mod.ToolBox, "__init__", init)
    res = cp.run(aid, "haiku")
    fin = res["final"]
    assert fake.calls == ["triage", "investigator", "investigator", "reviewer", "investigator", "investigator", "reviewer"]
    assert fin["revisions"] == 1
    assert fin["verification"]["nonexistent_or_foreign_tx"] == ["T999999"]
    assert fin["verification"]["tx_not_retrieved_by_agent"] == [tx]
    assert fin["verification"]["legal_refs_not_from_search"] == ["fake-chunk"]
    assert fin["legal_basis"] == []  # вигадане посилання не потрапляє у фінальний звіт


# ---------------------------------------------------------------- провайдери і резерв
def _gemini_response(text: str):
    from google.genai import types
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5))


class _FakeGemini:
    def __init__(self, text):
        self.text, self.calls = text, 0
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        return _gemini_response(self.text)


class _DownAnthropic:
    """Anthropic, що не відповідає: SDK після повторів кидає APIConnectionError."""

    def __init__(self):
        import httpx
        import anthropic
        err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))

        def fail(**kw):
            raise err
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=fail, create=fail))


def test_fallback_to_second_provider():
    from finmon.llm import LLM
    gem = _FakeGemini('{"risk_level": "high", "hypotheses": [{"typology": "дроблення", "why": "x"}], "plan": ["a"]}')
    llm = LLM(client=_DownAnthropic(), gemini_client=gem)
    msg, u = llm.call(node="triage", model_key="haiku", system="s", messages=[{"role": "user", "content": "x"}], output_format=TriageResult)
    assert msg.parsed_output.risk_level == "high"
    assert u.fallback_from == "claude-haiku-5-5" and u.model == "gemini-3.8-flash"
    assert u.output_tokens == 25 and u.cost_usd > 0


def test_no_fallback_when_disabled(monkeypatch):
    import pytest
    from finmon import llm as L
    monkeypatch.setattr(L, "FALLBACK_ENABLED", False)
    with pytest.raises(L.ProviderUnavailable):
        L.LLM(client=_DownAnthropic(), gemini_client=_FakeGemini("{}")).call(
            node="triage", model_key="haiku", system="s", messages=[{"role": "user", "content": "x"}], output_format=TriageResult)


def test_history_conversion_between_providers():
    """Історія Claude з викликом інструмента → Gemini: кожен виклик має відповідь, ролі чергуються."""
    from finmon.llm import Block, to_anthropic, to_gemini
    msgs = [
        {"role": "user", "content": "розслідуй"},
        {"role": "assistant", "content": [{"type": "text", "text": "дивлюсь"}, {"type": "tool_use", "id": "toolu_1", "name": "get_client_profile", "input": {"client_id": "C1"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"type": "fop"}', "is_error": False}]},
        {"role": "assistant", "content": [Block("submit", id="g1", name="submit_result", input={"x": 1})]},
        {"role": "user", "content": "доопрацюй"},
    ]
    g = to_gemini(msgs)
    assert [c.role for c in g] == ["user", "model", "user", "model", "user"]
    assert g[2].parts[0].function_response.name == "get_client_profile"
    assert g[2].parts[0].function_response.response == {"result": {"type": "fop"}}
    assert g[4].parts[0].function_response.name == "submit_result"  # здача висновку отримала відповідь
    a = to_anthropic(msgs)
    assert a[3]["content"] == [{"type": "text", "text": "Висновок надано."}]


def test_purpose_keywords():
    from finmon.nlp.purposes import keyword_predict
    assert keyword_predict("Ставка на матч, купон 2301") == "gambling"
    assert keyword_predict("купівля USDT на p2p") == "crypto"
    assert keyword_predict("повернення позики за серпень") == "loan"
    assert keyword_predict("зпшка зайшла") is None  # сленг правило не бачить — для цього потрібна модель
