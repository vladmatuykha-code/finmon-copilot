"""Виклик моделей двох провайдерів через один інтерфейс.

* Anthropic (Claude) і Google (Gemini): граф агентів не знає, яка модель відповідає;
* облік токенів і вартості за прайсом, кешування промптів;
* повтори з експоненційною затримкою (у SDK), потім переключення на резервного провайдера;
* метрики Prometheus і структуровані логи на кожен виклик.

Формат повідомлень — як у Messages API Anthropic. Для Gemini історія перекладається в contents,
а відповідь повертається блоками tool_use / text, які розуміє граф.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from . import obs
from .config import EFFORT, FALLBACK_BETA, FALLBACK_ENABLED, FALLBACK_MODELS, MODELS, PRICES, PROVIDER_FALLBACK, provider_of


@dataclass
class Usage:
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    cost_usd: float
    latency_s: float
    stop_reason: str
    fallback_from: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def cost_of(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout, pcache = PRICES[model]
    return (input_tokens * pin + output_tokens * pout + cache_read * pcache + cache_write * pin * 1.25) / 1e6


class ModelRefused(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    """Провайдер не відповів після повторів: мережа, 429 або 5xx. Привід переключитись на резерв."""

    def __init__(self, provider: str, reason: str, detail: str = ""):
        super().__init__(f"{provider} недоступний: {reason} {detail}".strip())
        self.provider, self.reason = provider, reason


# ---------------------------------------------------------------- спільний формат відповіді
@dataclass
class Block:
    """Блок відповіді Gemini у форматі, який розуміє граф (як content block Anthropic)."""
    type: str  # text | tool_use | submit | other
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    raw: Any = None  # оригінальна частина відповіді Gemini разом з thought_signature


@dataclass
class Reply:
    content: list
    stop_reason: str
    parsed_output: Any
    model: str


SUBMIT = "submit_result"
DUMMY_SIGNATURE = b"skip_thought_signature_validator"  # для викликів інструментів, які зробила інша модель


def _get(b, key, default=None):
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


def _chaos() -> set[str]:
    """FINMON_CHAOS=anthropic імітує збій провайдера — щоб перевірити переключення на резерв."""
    return {p.strip() for p in os.getenv("FINMON_CHAOS", "").split(",") if p.strip()}


def to_anthropic(messages: list) -> list:
    """Блоки Gemini в історії → словники Anthropic. Повідомлення без них повертаються як є."""
    if not any(m["role"] == "assistant" and isinstance(m["content"], list) and any(isinstance(b, Block) for b in m["content"]) for m in messages):
        return messages
    out = []
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant" and isinstance(c, list):
            blocks = []
            for b in c:
                if not isinstance(b, Block):
                    blocks.append(b)
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                elif b.type == "text" and b.text.strip():
                    blocks.append({"type": "text", "text": b.text})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "Висновок надано."}]})
        else:
            out.append(m)
    return out


def inline_refs(schema: dict) -> dict:
    """JSON Schema з $defs/$ref → вбудована схема без title: так її надійніше приймає Gemini."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


# ---------------------------------------------------------------- клієнт
class LLM:
    def __init__(self, client: anthropic.Anthropic | None = None, gemini_client=None):
        self._client = client  # клієнти створюються при першому виклику: тести і CI працюють без ключів
        self._gemini = gemini_client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=4)
        return self._client

    # ----- публічний метод, який викликає граф
    def call(self, *, node: str, model_key: str, system: str, messages: list, tools: list | None = None,
             output_format: type[BaseModel] | None = None, max_tokens: int = 16000):
        """Один запит до моделі. Повертає (message, Usage). З output_format — message.parsed_output."""
        try:
            return self._call_one(node, model_key, system, messages, tools, output_format, max_tokens)
        except ProviderUnavailable as e:
            backup = PROVIDER_FALLBACK.get(model_key) if FALLBACK_ENABLED else None
            if not backup:
                raise
            src, dst = MODELS[model_key], MODELS[backup]
            obs.FALLBACKS.labels(src, dst, e.reason).inc()
            obs.RECENT.add_fallback({"ts": time.strftime("%H:%M:%S"), "node": node, "from": src, "to": dst, "reason": e.reason, "alert_id": obs.alert_ctx.get()})
            obs.event("provider_fallback", logging.WARNING, node=node, from_model=src, to_model=dst, reason=e.reason)
            msg, u = self._call_one(node, backup, system, messages, tools, output_format, max_tokens)
            u.fallback_from = src
            return msg, u

    def _call_one(self, node, model_key, system, messages, tools, output_format, max_tokens):
        model = MODELS[model_key]
        provider = provider_of(model)
        t0 = time.perf_counter()
        try:
            if provider in _chaos():
                raise ProviderUnavailable(provider, "chaos", "(імітація збою)")
            if provider == "anthropic":
                msg, usage = self._anthropic(node, model, system, messages, tools, output_format, max_tokens)
            else:
                msg, usage = self._gemini_call(node, model, system, messages, tools, output_format, max_tokens)
        except ProviderUnavailable:
            obs.LLM_CALLS.labels(provider, model, node, "unavailable").inc()
            raise
        except ModelRefused:
            obs.LLM_CALLS.labels(provider, model, node, "refused").inc()
            raise
        except Exception:
            obs.LLM_CALLS.labels(provider, model, node, "error").inc()
            raise
        usage.latency_s = round(time.perf_counter() - t0, 2)
        obs.LLM_CALLS.labels(provider, model, node, "ok").inc()
        obs.LLM_SECONDS.labels(provider, model).observe(usage.latency_s)
        obs.LLM_TOKENS.labels(model, "input").inc(usage.input_tokens)
        obs.LLM_TOKENS.labels(model, "output").inc(usage.output_tokens)
        obs.LLM_TOKENS.labels(model, "cache_read").inc(usage.cache_read)
        obs.LLM_COST.labels(model, node).inc(usage.cost_usd)
        obs.event("llm_call", logging.DEBUG, node=node, model=model, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                  cache_read=usage.cache_read, cost_usd=usage.cost_usd, latency_s=usage.latency_s, stop_reason=usage.stop_reason)
        return msg, usage

    # ----- Anthropic
    def _anthropic(self, node, model, system, messages, tools, output_format, max_tokens):
        kw: dict = dict(
            model=model, max_tokens=max_tokens, system=system, messages=to_anthropic(messages),
            cache_control={"type": "ephemeral"},  # автокеш довгого префікса (system + історія інструментів)
            output_config={"effort": EFFORT.get(node, "medium")},
        )
        if tools:
            kw["tools"] = tools
        if model in FALLBACK_MODELS:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        try:
            if output_format is not None:
                msg = self.client.beta.messages.parse(output_format=output_format, **kw)
            else:
                msg = self.client.beta.messages.create(**kw)
        except anthropic.APIConnectionError as e:  # включно з тайм-аутом; SDK уже зробив повтори
            raise ProviderUnavailable("anthropic", "connection", type(e).__name__) from e
        except anthropic.RateLimitError as e:
            raise ProviderUnavailable("anthropic", "rate_limit") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ProviderUnavailable("anthropic", f"http_{e.status_code}") from e
            raise
        if msg.stop_reason == "refusal":
            raise ModelRefused(f"{node}: модель відмовилась ({getattr(msg, 'stop_details', None)})")
        u = msg.usage
        cr, cw = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = Usage(node, msg.model, u.input_tokens, u.output_tokens, cr, cw,
                      round(cost_of(model, u.input_tokens, u.output_tokens, cr, cw), 6), 0.0, msg.stop_reason)
        return msg, usage

    # ----- Google Gemini
    @property
    def gemini(self):
        if self._gemini is None:
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ProviderUnavailable("google", "no_api_key")
            from google import genai
            from google.genai import types
            self._gemini = genai.Client(api_key=key, http_options=types.HttpOptions(
                timeout=180_000,
                retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0, exp_base=2.0, http_status_codes=[408, 429, 500, 502, 503, 504])))
        return self._gemini

    def _gemini_call(self, node, model, system, messages, tools, output_format, max_tokens):
        from google.genai import errors, types

        cfg: dict = dict(system_instruction=system, max_output_tokens=max_tokens,
                         thinking_config=types.ThinkingConfig(thinking_level=EFFORT.get(node, "medium").upper()))
        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"]) for t in tools or []]
        if output_format is not None and decls:
            # Висновок разом з інструментами: модель здає його викликом submit_result зі схемою відповіді.
            decls.append(types.FunctionDeclaration(name=SUBMIT, description="Здати фінальний висновок. Виклич один раз, коли факти зібрано.",
                                                   parameters_json_schema=inline_refs(output_format.model_json_schema())))
            cfg["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
        elif output_format is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = inline_refs(output_format.model_json_schema())
        if decls:
            cfg["tools"] = [types.Tool(function_declarations=decls)]
        config = types.GenerateContentConfig(**cfg)
        contents = to_gemini(messages)

        tot = {"in": 0, "out": 0, "cache": 0}
        for _ in range(3):
            try:
                resp = self.gemini.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as e:
                if e.code == 429 or (e.code or 0) >= 500:
                    raise ProviderUnavailable("google", "rate_limit" if e.code == 429 else f"http_{e.code}") from e
                raise
            except Exception as e:  # мережеві помилки httpx після повторів
                if type(e).__module__.startswith("httpx"):
                    raise ProviderUnavailable("google", "connection", type(e).__name__) from e
                raise
            um = resp.usage_metadata
            cached = um.cached_content_token_count or 0
            tot["in"] += (um.prompt_token_count or 0) - cached
            tot["cache"] += cached
            tot["out"] += (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
            reply, retry_note = self._gemini_reply(resp, model, tools, output_format)
            if retry_note is None:
                break
            contents = contents + retry_note  # відповідь не пройшла схему: показуємо помилку і просимо ще раз
        usage = Usage(node, model, tot["in"], tot["out"], tot["cache"], 0, round(cost_of(model, tot["in"], tot["out"], tot["cache"]), 6), 0.0, reply.stop_reason)
        return reply, usage

    def _gemini_reply(self, resp, model, tools, output_format):
        """Відповідь Gemini → Reply. Другий елемент — додаткові contents для повтору, якщо вивід не пройшов схему."""
        from google.genai import types

        if not resp.candidates:
            raise ModelRefused(f"Gemini не дав відповіді: {getattr(resp, 'prompt_feedback', None)}")
        cand = resp.candidates[0]
        reason = str(cand.finish_reason or "")
        if any(x in reason for x in ("SAFETY", "PROHIBITED", "BLOCKLIST", "SPII")):
            raise ModelRefused(f"Gemini відмовився: {reason}")
        parts = (cand.content.parts if cand.content else None) or []
        blocks: list[Block] = []
        for p in parts:
            if p.function_call:
                fc = p.function_call
                if not fc.id:
                    fc.id = f"gcall_{uuid.uuid4().hex[:16]}"
                blocks.append(Block("submit" if fc.name == SUBMIT else "tool_use", id=fc.id, name=fc.name, input=dict(fc.args or {}), raw=p))
            elif p.text and not p.thought:
                blocks.append(Block("text", text=p.text, raw=p))
            else:
                blocks.append(Block("other", raw=p))
        calls = [b for b in blocks if b.type == "tool_use"]
        submit = next((b for b in blocks if b.type == "submit"), None)

        def retry_with(error: str):
            responses = [types.Part(function_response=types.FunctionResponse(
                name=b.name, id=b.id, response={"error": error} if b is submit else {"result": "skipped"})) for b in blocks if b.type in ("tool_use", "submit")]
            follow = types.Content(role="user", parts=responses or [types.Part(text=f"Відповідь не відповідає схемі: {error}. Дай відповідь ще раз строго за схемою.")])
            return [cand.content, follow] if cand.content else [follow]

        if output_format is not None and tools:
            if submit is not None:
                try:
                    return Reply(blocks, "end_turn", output_format.model_validate(submit.input), model), None
                except ValidationError as e:
                    return Reply(blocks, "end_turn", None, model), retry_with(str(e)[:600])
            return Reply(blocks, "tool_use" if calls else "end_turn", None, model), None
        if output_format is not None:
            text = "".join(b.text for b in blocks if b.type == "text")
            try:
                return Reply(blocks, "end_turn", output_format.model_validate_json(text), model), None
            except ValidationError as e:
                return Reply(blocks, "end_turn", None, model), retry_with(str(e)[:600])
        return Reply(blocks, "tool_use" if calls else "end_turn", None, model), None


def to_gemini(messages: list) -> list:
    """Історія у форматі Anthropic (блоки SDK, словники або Block) → contents для Gemini.

    Кожен виклик інструмента в попередній репліці моделі отримує function_response — Gemini цього вимагає.
    Виклики, зроблені іншою моделлю, позначаються службовим thought_signature.
    """
    from google.genai import types

    contents, id2name, pending = [], {}, []
    for m in messages:
        c = m["content"]
        if m["role"] == "assistant":
            parts, pending = [], []
            for b in c if isinstance(c, list) else [c]:
                if isinstance(b, str):
                    parts.append(types.Part(text=b))
                    continue
                typ = _get(b, "type")
                if isinstance(b, Block) and b.raw is not None:
                    parts.append(b.raw)
                elif typ in ("tool_use", "submit"):
                    parts.append(types.Part(function_call=types.FunctionCall(name=_get(b, "name"), args=_get(b, "input") or {}, id=_get(b, "id")),
                                            thought_signature=DUMMY_SIGNATURE))
                elif typ == "text" and (_get(b, "text") or "").strip():
                    parts.append(types.Part(text=_get(b, "text")))
                if typ in ("tool_use", "submit"):
                    id2name[_get(b, "id")] = _get(b, "name")
                    pending.append(_get(b, "id"))
            contents.append(types.Content(role="model", parts=parts or [types.Part(text="…")]))
            continue
        responses, texts, answered = [], [], set()
        for b in [c] if isinstance(c, str) else c:
            if isinstance(b, str):
                texts.append(types.Part(text=b))
            elif _get(b, "type") == "tool_result":
                tid, body = _get(b, "tool_use_id"), _get(b, "content")
                try:
                    body = json.loads(body)
                except (TypeError, ValueError):
                    pass
                answered.add(tid)
                responses.append(types.Part(function_response=types.FunctionResponse(
                    name=id2name.get(tid, "tool"), id=tid, response={"error" if _get(b, "is_error") else "result": body})))
            elif _get(b, "type") == "text":
                texts.append(types.Part(text=_get(b, "text")))
        missing = [types.Part(function_response=types.FunctionResponse(name=id2name.get(i, "tool"), id=i, response={"result": "accepted"}))
                   for i in pending if i not in answered]
        contents.append(types.Content(role="user", parts=responses + missing + texts))
        pending = []
    return contents
