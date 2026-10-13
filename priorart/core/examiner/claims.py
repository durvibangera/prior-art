"""Claims the inventors could plausibly file, broken into elements.

Novelty and inventive step are judged per ELEMENT (limitation), not per
document: a claim is anticipated only if one reference discloses every
element, and is obvious if a combination does. So everything downstream --
the chart, the interview's choice of question, the report -- is keyed on
element ids like "C1.b".

Each element is tagged `context` (the known setting the claim operates in:
"receiving an audio signal from a microphone") or `distinguishing` (what the
inventors say is new). The tag drives the search safety gate: queries may
never combine distinguishing elements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..llm import LLMClient
from .context import Invention

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["method", "system"]},
                    "depends_on": {"type": "string"},
                    "text": {"type": "string"},
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "role": {"type": "string", "enum": ["context", "distinguishing"]},
                            },
                            "required": ["text", "role"],
                        },
                    },
                },
                "required": ["category", "text", "elements"],
            },
        },
    },
    "required": ["claims"],
}

CLAIMS_SYSTEM = """You are a senior patent attorney drafting the claim set for an
Indian patent application (with possible PCT filing) from an inventor's
disclosure. Software per se is excluded in India under Section 3(k), so
claims must be framed as a technical method or system producing a technical
effect, not as an algorithm or program.

Draft 3 to 7 claims the inventors could realistically file:
- one or two broad independent claims (method and/or system) capturing the
  core technical contribution,
- dependent claims for each genuinely distinct variant or refinement the
  disclosure actually describes (alternative architectures, fusion schemes,
  windowing/decision logic, dataset or operating-condition features only if
  they carry a technical effect).

Break every claim into ELEMENTS -- one technical limitation each, the way a
claim chart would split it. Tag each element:
- "context": the known setting or a conventional step (e.g. capturing audio
  with a microphone, outputting a detection decision)
- "distinguishing": a feature the disclosure presents as new or as the
  reason the invention works better

Rules:
- Use only what the disclosure actually says. Do not invent parameters,
  layers or results that are not in it.
- If the disclosure is vague about something an examiner will need (e.g. how
  features are fused), still draft the claim, but keep the element at the
  level of detail actually disclosed.
- For a dependent claim, set depends_on to the 1-based number of its parent
  claim and list only the ADDITIONAL elements it adds.
- Plain technical language, no "comprising: a) ... b)" lettering inside text."""


def _number(raw: list[dict]) -> list[dict]:
    """Assign stable ids C1, C1.a ... in code, never trusting model ids."""
    out = []
    for i, c in enumerate(raw, start=1):
        cid = f"C{i}"
        dep = (c.get("depends_on") or "").strip().lstrip("Cc")
        elements = [{"id": f"{cid}.{chr(ord('a') + j)}", "text": e["text"].strip(),
                     "role": e.get("role", "distinguishing")}
                    for j, e in enumerate(c.get("elements") or []) if e.get("text", "").strip()]
        out.append({"id": cid, "category": c.get("category", "method"),
                    "depends_on": f"C{dep}" if dep.isdigit() and int(dep) < i else None,
                    "text": c.get("text", "").strip(), "elements": elements})
    return out


def derive_claims(llm: LLMClient, invention: Invention,
                  draft: list[dict] | None = None, thinking: bool = True) -> list[dict]:
    user = f"INVENTOR'S DISCLOSURE\n\n{invention.text()}"
    if draft:
        user += ("\n\nAN EARLIER ROUGH CLAIM LIST (improve on it; do not just copy it):\n"
                 + "\n".join(f"- {d.get('text', '')}" for d in draft))
    user += "\n\nDraft the claim set now."
    result = llm.converse_json(
        [{"role": "system", "content": CLAIMS_SYSTEM}, {"role": "user", "content": user}],
        CLAIMS_SCHEMA, thinking=thinking, max_tokens=8000)
    return _number(result.get("claims") or [])


def all_elements(claims: list[dict]) -> list[dict]:
    """Every element once, tagged with the claim that introduces it."""
    out = []
    for c in claims:
        for e in c["elements"]:
            out.append({**e, "claim_id": c["id"]})
    return out


def required_elements(claims: list[dict], claim_id: str) -> list[str]:
    """Elements a claim actually requires, including those inherited from
    the claim it depends on -- anticipating a dependent claim needs the
    parent's elements too."""
    by_id = {c["id"]: c for c in claims}
    ids, seen = [], set()
    cur = by_id.get(claim_id)
    while cur and cur["id"] not in seen:
        seen.add(cur["id"])
        ids = [e["id"] for e in cur["elements"]] + ids
        cur = by_id.get(cur.get("depends_on") or "")
    return ids


def claims_hash(claims: list[dict]) -> str:
    return hashlib.sha1(json.dumps(claims, sort_keys=True).encode()).hexdigest()[:12]


def format_claims(claims: list[dict]) -> str:
    lines = []
    for c in claims:
        dep = f" (depends on {c['depends_on']})" if c.get("depends_on") else ""
        lines.append(f"{c['id']} [{c['category']}]{dep}: {c['text']}")
        for e in c["elements"]:
            mark = "*" if e["role"] == "distinguishing" else " "
            lines.append(f"   {mark} {e['id']}  {e['text']}")
    return "\n".join(lines)


def review_gate(claims: list[dict], path: Path, regenerate) -> list[dict]:
    """Show the claims, let the inventor edit the JSON file, accept. Returns
    the accepted claims (re-numbered, so hand edits can't break ids)."""
    while True:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(claims, indent=2, ensure_ascii=False))
        print("\nDRAFT CLAIMS  (* = distinguishing element)\n" + "-" * 68)
        print(format_claims(claims))
        print("-" * 68)
        print(f"Saved to {path}")
        choice = input("[a]ccept  [e]dit the file then press enter  [r]egenerate  > ").strip().lower()
        if choice.startswith("a"):
            return claims
        if choice.startswith("r"):
            claims = regenerate()
            continue
        if choice.startswith("e"):
            input(f"Edit {path} now, save it, then press enter... ")
            try:
                edited = json.loads(path.read_text())
                claims = _number([{**c, "depends_on": (c.get("depends_on") or "").lstrip("C")}
                                  for c in edited])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"  could not read the edited file ({exc}) -- keeping the previous version")
