"""The claim chart: every claim element against every retrieved reference.

One LLM call per reference reads that reference in depth against ALL
elements at once (so it can see that, say, the reference's "multi-level
discrete wavelet features" are or are not a scattering transform). Two
things are then done in code, never trusted to the model:

1. Quote verification. A cell marked disclosed/partial must carry a quote
   that actually appears in the reference text. If it doesn't, the cell is
   downgraded to `unverified` and never counts toward anticipation.
2. The legal arithmetic. A claim is anticipated only if ONE reference has
   every required element disclosed; it is at inventive-step risk if TWO
   references together cover every element. Counting, not judgment.
"""
from __future__ import annotations

import itertools
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm import LLMClient, LLMUnavailable
from .claims import all_elements, format_claims, required_elements
from .context import Reference, normalize

STATUSES = ("disclosed", "partial", "absent")

CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "closeness": {"type": "string", "enum": ["very_close", "related", "distant"]},
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "quote": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["element_id", "status", "quote", "reasoning"],
            },
        },
    },
    "required": ["summary", "closeness", "cells"],
}

CHART_SYSTEM = """You are a patent examiner preparing a claim chart. You are given
a draft claim set split into elements, and ONE published prior-art document.
Decide, for EVERY element, what this document discloses.

- disclosed: the document explicitly teaches this element, or something a
  person skilled in the art would read as the same thing. Same function by a
  clearly different technique is NOT disclosed.
- partial: the document teaches part of the element, a generic version of it
  (e.g. "a neural network" against "a 1D CNN with global average pooling"),
  or a close alternative that would make the element an obvious substitution.
- absent: nothing in the document addresses it.

For disclosed and partial, `quote` must be copied EXACTLY, word for word,
from the document text (at most 40 words, one contiguous span). For absent,
quote is "". `reasoning` is one or two sentences saying precisely what the
document does for this element and how it differs, naming techniques.

`summary`: two sentences on what the document actually does technically.
`closeness`: very_close if it shares the core technical approach AND purpose,
related if it shares one of them, distant otherwise.

Be strict and specific. Same field is not the same technique."""


def _norm_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", normalize(text).lower())


def quote_verified(quote: str, ref_text: str) -> bool:
    """Every segment of the quote (split on ellipses) must appear verbatim,
    ignoring case, punctuation and whitespace."""
    hay = _norm_for_match(ref_text)
    segments = [s for s in re.split(r"\.\.\.|…", quote or "") if s.strip()]
    if not segments:
        return False
    for seg in segments:
        needle = _norm_for_match(seg).strip()
        if len(needle) < 8 or needle not in hay:
            return False
    return True


def chart_reference(llm: LLMClient, claims: list[dict], ref: Reference,
                    thinking: bool = True) -> dict:
    """Returns {"summary", "closeness", "cells": {element_id: cell}}."""
    elements = all_elements(claims)
    user = (f"DRAFT CLAIMS (element ids in the left column)\n{format_claims(claims)}\n\n"
            f"PRIOR-ART DOCUMENT\n{ref.digest(max_chars=14000)}\n\n"
            f"Chart all {len(elements)} elements against this document now.")
    out = llm.converse_json(
        [{"role": "system", "content": CHART_SYSTEM}, {"role": "user", "content": user}],
        CHART_SCHEMA, thinking=thinking, max_tokens=6000)

    known = {e["id"] for e in elements}
    source_text = f"{ref.text.title} {ref.text.full}"
    cells: dict[str, dict] = {}
    for c in out.get("cells") or []:
        eid = (c.get("element_id") or "").strip()
        if eid not in known:
            continue
        status = c.get("status") if c.get("status") in STATUSES else "absent"
        quote = (c.get("quote") or "").strip()
        reasoning = (c.get("reasoning") or "").strip()
        if status != "absent" and not quote_verified(quote, source_text):
            reasoning = f"[quote not found in the document text] {reasoning}"
            status = "unverified"
        cells[eid] = {"status": status, "quote": quote, "reasoning": reasoning}
    for eid in known - set(cells):
        cells[eid] = {"status": "absent", "quote": "", "reasoning": "not addressed by the model"}
    closeness = out.get("closeness") if out.get("closeness") in ("very_close", "related", "distant") else "distant"
    return {"summary": (out.get("summary") or "").strip(), "closeness": closeness, "cells": cells}


def build_chart(llm: LLMClient, store, conversation_id: str, claims: list[dict],
                refs: list[Reference], thinking: bool = True, workers: int = 4) -> int:
    """Charts every reference not yet charted. Returns how many were done.
    A failed reference is reported and skipped, so the next run retries it."""
    done = set(store.get_chart_refs(conversation_id))
    todo = [r for r in refs if r.application_number not in done]
    if not todo:
        return 0
    print(f"Reading {len(todo)} reference(s) against {len(all_elements(claims))} claim elements"
          f" ({workers} at a time)...")
    count = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(chart_reference, llm, claims, r, thinking): r for r in todo}
        for fut in as_completed(futures):
            ref = futures[fut]
            try:
                result = fut.result()
            except LLMUnavailable as exc:
                print(f"  {ref.application_number}: FAILED ({exc}) -- will retry next run")
                continue
            for eid, cell in result["cells"].items():
                store.set_chart_cell(conversation_id, eid, ref.application_number,
                                     cell["status"], cell["quote"], cell["reasoning"])
            store.set_chart_ref(conversation_id, ref.application_number,
                                result["summary"], result["closeness"])
            count += 1
            hits = sum(1 for c in result["cells"].values() if c["status"] in ("disclosed", "partial"))
            print(f"  [{count}/{len(todo)}] {ref.application_number:20} {result['closeness']:10}"
                  f" {hits} element(s) touched")
    return count


# ------------------------------------------------------------- assessment


def cells_by_ref(cells: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for c in cells:
        out.setdefault(c["application_number"], {})[c["element_id"]] = c
    return out


def assess(claims: list[dict], cells: list[dict], max_pairs: int = 5) -> dict[str, dict]:
    """Per claim: anticipated / inventive_step_risk / all_elements_known /
    partially_covered / clear. Only `disclosed` counts; `unverified` never does."""
    by_ref = cells_by_ref(cells)
    out = {}
    for claim in claims:
        req = required_elements(claims, claim["id"])
        covered = {appno: {e for e in req if refcells.get(e, {}).get("status") == "disclosed"}
                   for appno, refcells in by_ref.items()}
        touched = {appno: {e for e in req if refcells.get(e, {}).get("status") in ("disclosed", "partial")}
                   for appno, refcells in by_ref.items()}
        full = set(req)

        anticipating = sorted(a for a, cov in covered.items() if cov >= full)
        pairs = []
        if not anticipating:
            useful = [a for a, cov in covered.items() if cov]
            for a, b in itertools.combinations(sorted(useful), 2):
                if covered[a] | covered[b] >= full:
                    pairs.append((a, b))
                    if len(pairs) >= max_pairs:
                        break
        union_disclosed = set().union(*covered.values()) if covered else set()
        union_touched = set().union(*touched.values()) if touched else set()

        if anticipating:
            status = "anticipated"
        elif pairs:
            status = "inventive_step_risk"
        elif full and union_disclosed >= full:
            status = "all_elements_known"
        elif union_touched:
            status = "partially_covered"
        else:
            status = "clear"

        element_refs = {e: sorted(a for a, cov in covered.items() if e in cov) for e in req}
        element_partial = {e: sorted(a for a, t in touched.items() if e in t and e not in covered[a])
                           for e in req}
        out[claim["id"]] = {
            "status": status, "required": req, "anticipating": anticipating,
            "pairs": pairs, "element_refs": element_refs, "element_partial": element_partial,
            "uncovered": [e for e in req if e not in union_touched],
        }
    return out


def closest_refs(cells: list[dict], refs_meta: dict[str, dict], n: int = 12) -> list[str]:
    """References ranked by how much of the claim set they touch."""
    weight = {"disclosed": 2, "partial": 1}
    score: dict[str, float] = {}
    for c in cells:
        score[c["application_number"]] = score.get(c["application_number"], 0) + weight.get(c["status"], 0)
    bonus = {"very_close": 3, "related": 1}
    for appno, meta in refs_meta.items():
        score[appno] = score.get(appno, 0) + bonus.get(meta.get("closeness"), 0)
    return [a for a, s in sorted(score.items(), key=lambda kv: -kv[1]) if s > 0][:n]


_SYMBOL = {"disclosed": "#", "partial": "~", "absent": ".", "unverified": "?"}


def format_chart(claims: list[dict], cells: list[dict], refs: list[str]) -> str:
    """Element rows x reference columns, compact enough for a terminal."""
    by_ref = cells_by_ref(cells)
    elements = all_elements(claims)
    lines = ["legend: # disclosed  ~ partial  . absent  ? quote unverified",
             "        " + " ".join(f"R{i + 1:<2}" for i in range(len(refs)))]
    for e in elements:
        row = " ".join(f"{_SYMBOL.get(by_ref.get(a, {}).get(e['id'], {}).get('status'), ' '):<3}"
                       for a in refs)
        lines.append(f"{e['id']:7} {row}  {e['text'][:60]}")
    lines.append("")
    for i, a in enumerate(refs):
        lines.append(f"R{i + 1:<2} = {a}")
    return "\n".join(lines)
