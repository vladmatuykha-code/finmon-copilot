"""Нарізка нормативних документів на фрагменти (chunking).

Три стратегії, щоб порівняти їх на оцінці пошуку:
  * structural        — за структурою документа (стаття → частина → пункт), з заголовком-контекстом
                        («Закон 361-IX, ст. 20 «Порогові фінансові операції», ч. 1»);
  * structural_noctx  — та сама нарізка, але без заголовка (щоб виміряти внесок контексту);
  * fixed             — RecursiveCharacterTextSplitter з LangChain: шматки фіксованої довжини з перекриттям.

Кожен фрагмент має refs — посилання на структурні одиниці («law:art20:ch1», «nbu:annex20:III:p68»),
які він покриває. За ними оцінюємо, чи знайшов пошук правильне місце в документі.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import RAW_DIR

MAX_CHARS = 900
LAW_NAME = "Закон України №361-IX «Про запобігання та протидію легалізації (відмиванню) доходів…»"
NBU_NAME = "Положення НБУ №65 «Про здійснення банками фінансового моніторингу»"


@dataclass
class Leaf:
    """Найменша структурна одиниця з точним місцем у тексті документа."""
    doc: str
    ref: str
    context: str
    start: int
    end: int
    short: str = ""


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    text: str
    header: str
    refs: list[str] = field(default_factory=list)

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text


def _clean(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not ln.lstrip().startswith("{")]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out)


def load_law() -> str:
    t = (RAW_DIR / "law_361-IX.txt").read_text(encoding="utf-8")
    start = re.search(r"^Стаття 1\. ", t, re.M).start()
    end_m = re.search(r"^Стаття 209\.", t, re.M)
    return _clean(t[start: end_m.start() if end_m else len(t)])


def load_nbu() -> str:
    t = (RAW_DIR / "nbu_65_2020.txt").read_text(encoding="utf-8")
    start = re.search(r"^ПОЛОЖЕННЯ\s*$", t, re.M).start()
    return _clean(t[start:])


def _split(text: str, base: int, pattern: str):
    """Ділить text на шматки за рядками, що відповідають pattern. Повертає (мітка, start, end) в абсолютних позиціях."""
    ms = list(re.finditer(pattern, text, re.M))
    if not ms:
        return []
    parts = []
    if ms[0].start() > 0 and text[: ms[0].start()].strip():
        parts.append((None, base, base + ms[0].start()))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        parts.append((m.group(1), base + m.start(), base + end))
    return parts


def law_leaves(text: str) -> list[Leaf]:
    leaves: list[Leaf] = []
    for art, a0, a1 in _split(text, 0, r"^Стаття (\d+(?:-\d+)?)\. "):
        if art is None:
            continue
        title = text[a0:a1].split("\n", 1)[0].split(". ", 1)[-1].strip()
        ctx = f"{LAW_NAME}, стаття {art} «{title}»"
        sh = f"Закон 361-IX, ст. {art} «{title}»"
        body = text[a0:a1]
        parts = _split(body, a0, r"^(\d+)\. ")
        if not parts:
            leaves.append(Leaf("law", f"law:art{art}", ctx, a0, a1, sh))
            continue
        for ch, p0, p1 in parts:
            ref = f"law:art{art}" + (f":ch{ch}" if ch else "")
            pctx = ctx + (f", частина {ch}" if ch else "")
            psh = sh + (f", ч. {ch}" if ch else "")
            if p1 - p0 > MAX_CHARS * 2:
                items = _split(text[p0:p1], p0, r"^(\d+)\) ")
                if items:
                    for it, i0, i1 in items:
                        leaves.append(Leaf("law", ref + (f":i{it}" if it else ""), pctx + (f", пункт {it}" if it else ""), i0, i1, psh))
                    continue
            leaves.append(Leaf("law", ref, pctx, p0, p1, psh))
    return leaves


def nbu_leaves(text: str) -> list[Leaf]:
    leaves: list[Leaf] = []
    blocks = _split(text, 0, r"^Додаток (\d+)\s*$")
    for annex, b0, b1 in blocks:
        block = text[b0:b1]
        if annex is None:
            unit, name, short = "main", NBU_NAME, "НБУ №65"
        else:
            lines = [ln.strip() for ln in block.splitlines()[1:] if ln.strip()]
            lines = [ln for ln in lines if ln not in ("до Положення", "про здійснення банками", "фінансового моніторингу") and not ln.startswith("(")]
            title = lines[0] if lines else ""
            if title.isupper() and len(lines) > 1:
                title = f"{title.capitalize()} {lines[1]}"
            unit, name = f"annex{annex}", f"{NBU_NAME}, Додаток {annex} «{title}»"
            short = f"НБУ №65, Дод. {annex} «{title[:60]}»"
        sections = _split(block, b0, r"^([IVX]+)\. ") or [(None, b0, b1)]
        for sec, s0, s1 in sections:
            sec_title = text[s0:s1].split("\n", 1)[0].split(". ", 1)[-1].strip() if sec else ""
            sctx = name + (f", розділ {sec} «{sec_title}»" if sec else "")
            sref = f"nbu:{unit}" + (f":{sec}" if sec else "")
            ssh = short + (f", розд. {sec}" if sec else "")
            points = _split(text[s0:s1], s0, r"^(\d+(?:-\d+)?)\. ")
            if not points:
                leaves.append(Leaf("nbu", sref, sctx, s0, s1, ssh))
                continue
            for pt, p0, p1 in points:
                first = text[p0:p1].split("\n", 1)[0].split(". ", 1)[-1].strip().rstrip(":")[:70]
                leaves.append(Leaf("nbu", sref + (f":p{pt}" if pt else ""), sctx + (f", пункт {pt}" if pt else ""), p0, p1, ssh + (f", п. {pt} «{first}»" if pt else "")))
    return leaves


_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=120, separators=["\n\n", "\n", "; ", ". ", " ", ""])


def structural_chunks(doc: str, text: str, leaves: list[Leaf], with_header: bool = True, short: bool = False) -> list[Chunk]:
    """Групує сусідні одиниці з однаковим контекстом до MAX_CHARS; довгі одиниці ріже далі."""
    chunks: list[Chunk] = []

    def emit(ctx: str, refs: list[str], body: str) -> None:
        body = body.strip()
        if len(body) < 40:
            return
        chunks.append(Chunk(f"{doc}-s{len(chunks):04d}", doc, body, ctx if with_header else "", refs))

    def ctx_of(lf: Leaf, base: bool) -> str:
        if short:
            return lf.short.rsplit(", п. ", 1)[0].rsplit(", ч. ", 1)[0] if base else lf.short
        return lf.context.rsplit(", пункт", 1)[0].rsplit(", частина", 1)[0] if base else lf.context

    buf, buf_refs, buf_ctx = "", [], None
    for lf in leaves:
        piece = text[lf.start: lf.end].strip()
        base_ctx = ctx_of(lf, True)
        if len(piece) > MAX_CHARS:
            if buf:
                emit(buf_ctx, buf_refs, buf)
                buf, buf_refs, buf_ctx = "", [], None
            for sub in _splitter.split_text(piece):
                emit(ctx_of(lf, False), [lf.ref], sub)
            continue
        if buf and (base_ctx != buf_ctx or len(buf) + len(piece) > MAX_CHARS):
            emit(buf_ctx, buf_refs, buf)
            buf, buf_refs = "", []
        buf_ctx = base_ctx
        buf = f"{buf}\n{piece}" if buf else piece
        buf_refs.append(lf.ref)
    if buf:
        emit(buf_ctx, buf_refs, buf)
    return chunks


def fixed_chunks(doc: str, text: str, leaves: list[Leaf]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=120, add_start_index=True)
    chunks = []
    for i, d in enumerate(splitter.create_documents([text])):
        s = d.metadata["start_index"]
        e = s + len(d.page_content)
        refs = [lf.ref for lf in leaves if lf.start < e and lf.end > s]
        chunks.append(Chunk(f"{doc}-f{i:04d}", doc, d.page_content, "", refs))
    return chunks


def build_all() -> dict[str, list[Chunk]]:
    law, nbu = load_law(), load_nbu()
    ll, nl = law_leaves(law), nbu_leaves(nbu)
    return {
        "structural": structural_chunks("law", law, ll) + structural_chunks("nbu", nbu, nl),
        "structural_short": structural_chunks("law", law, ll, short=True) + structural_chunks("nbu", nbu, nl, short=True),
        "structural_noctx": structural_chunks("law", law, ll, with_header=False) + structural_chunks("nbu", nbu, nl, with_header=False),
        "fixed": fixed_chunks("law", law, ll) + fixed_chunks("nbu", nbu, nl),
    }


def ref_matches(expected: str, refs: list[str]) -> bool:
    """«nbu:annex20:p68» збігається з «nbu:annex20:III:p68»: усі частини очікуваного посилання є у фрагменті."""
    exp = set(expected.split(":"))
    return any(exp <= set(r.split(":")) for r in refs)


if __name__ == "__main__":
    for name, chs in build_all().items():
        lens = sorted(len(c.text) for c in chs)
        print(f"{name:18s} chunks={len(chs):4d} median_len={lens[len(lens) // 2]} max_len={lens[-1]}")
    s = build_all()["structural"]
    for c in s:
        if any(r.startswith("nbu:annex20:III:p68") for r in c.refs):
            print("\nПРИКЛАД:", c.header, c.refs[:4], "\n", c.text[:300])
            break
