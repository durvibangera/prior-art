"""Compare harvested documents against claims -- the actual "how am I
different" answer, not a similarity score.

For each document, the question isn't "is this relevant" (the scorer already
answered that) -- it's "for EACH claim we intend to file, does this document
anticipate it, partially overlap it, or is it unrelated". That per-claim
verdict is what rolls up into the sentence the law firm needs: "C1 and C3 are
exposed; C4 and C5 appear clear."

One LLM call compares a document against ALL claims at once, not one call per
(claim, document) pair -- with 6 claims and dozens of shortlisted documents,
pairwise calls would mean hundreds of round-trips to a 4B model for no
quality gain, since the model reads the same abstract either way.

The rules fallback is deliberately weak: keyword overlap can support "these
touch related facets" but never "this document anticipates that claim" --
that requires actually reading and judging text, which is exactly what rules
can't do. Rules never emit "anticipates", only "partial" or "unrelated", so a
no-LLM run under-claims rather than over-claims novelty risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Claim
from .llm import LLMClient, LLMUnavailable
from .scorer import Scorer

VERDICTS = ("anticipates", "partial", "unrelated")


@dataclass
class ClaimVerdict:
    claim_id: str
    verdict: str          # one of VERDICTS
    quote: str = ""        # supporting text FROM THE DOCUMENT, not the claim
    reasoning: str = ""


@dataclass
class DocumentComparison:
    application_number: str
    title: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    method: str = "rules"   # llm | rules
    fallback_reason: str = ""   # set when method=="rules" because the LLM call failed

    def verdict_for(self, claim_id: str) -> ClaimVerdict | None:
        return next((v for v in self.verdicts if v.claim_id == claim_id), None)


COMPARE_PROMPT_SYSTEM = """You are assessing patent prior-art risk. You will be
given a list of CLAIMS from an invention we may file, and the ABSTRACT of one
already-published document found in a prior-art search.

For EACH claim, decide:
- "anticipates": the abstract describes the same core technical mechanism as
  the claim, not merely the same application domain. Two systems that both
  detect drones acoustically are NOT automatically "anticipates" -- the
  SPECIFIC technique named in the claim must actually appear.
- "partial": the abstract is in the same technical area and shares a related
  technique, but does not describe the same specific mechanism.
- "unrelated": no meaningful technical overlap.

Be conservative. A shared application (drone detection, UAV surveillance) with
no shared TECHNIQUE is "unrelated", not "partial". Quote the exact phrase from
the ABSTRACT (not the claim) that supports your verdict; use an empty string
for "unrelated".

Return ONLY a JSON array, one object per claim, in the SAME ORDER given:
[{"claim_id": "...", "verdict": "anticipates|partial|unrelated",
  "quote": "...", "reasoning": "<one short sentence>"}]
"""


def _compare_llm(document: dict, claims: list[Claim], llm: LLMClient,
                 temperature: float = 0.0) -> DocumentComparison:
    claim_list = "\n".join(f"- {c.id}: {c.text}" for c in claims)
    user = f"""CLAIMS:
{claim_list}

DOCUMENT TITLE: {document.get('title', '')}

DOCUMENT ABSTRACT:
{(document.get('abstract') or '')[:3000]}

Assess all {len(claims)} claims now, as a JSON array only, same order as given."""

    raw = llm.chat_json(COMPARE_PROMPT_SYSTEM, user, max_tokens=2500, temperature=temperature)
    if not isinstance(raw, list):
        raise LLMUnavailable(f"expected a JSON array, got {type(raw).__name__}")

    by_id = {c.id: c for c in claims}
    verdicts = []
    for item in raw:
        cid = item.get("claim_id", "")
        if cid not in by_id:
            continue
        verdict = item.get("verdict", "unrelated")
        if verdict not in VERDICTS:
            verdict = "unrelated"
        verdicts.append(ClaimVerdict(
            claim_id=cid, verdict=verdict,
            quote=str(item.get("quote", "")).strip(),
            reasoning=str(item.get("reasoning", "")).strip(),
        ))
    # Any claim the model silently dropped is recorded, not lost -- a missing
    # verdict must never be read as "unrelated by default".
    seen = {v.claim_id for v in verdicts}
    for c in claims:
        if c.id not in seen:
            verdicts.append(ClaimVerdict(claim_id=c.id, verdict="unrelated",
                                         reasoning="model did not return a verdict for this claim"))
    return DocumentComparison(document["application_number"], document.get("title", ""),
                              verdicts, method="llm")


def _compare_rules(document: dict, claims: list[Claim], profile: dict) -> DocumentComparison:
    scorer = Scorer(profile)
    text = f"{document.get('title','')} {document.get('abstract','')}"
    doc_facets = set(scorer.match_facets(text))

    verdicts = []
    for c in claims:
        overlap = doc_facets & set(c.facets) - {"modality", "platform", "task"}
        if overlap:
            verdicts.append(ClaimVerdict(
                claim_id=c.id, verdict="partial",
                reasoning=f"shares facet(s) {sorted(overlap)} with this claim "
                         f"(keyword overlap only, not verified)"))
        else:
            verdicts.append(ClaimVerdict(claim_id=c.id, verdict="unrelated"))
    return DocumentComparison(document["application_number"], document.get("title", ""),
                              verdicts, method="rules")


def compare_document(document: dict, claims: list[Claim], profile: dict,
                     llm: LLMClient | None = None) -> DocumentComparison:
    if llm is not None:
        last_exc = None
        # Calls run at temperature 0 for reproducibility, which means retrying
        # the IDENTICAL prompt reliably reproduces the IDENTICAL failure -- a
        # retry only helps if it asks for something different. So the retry
        # bumps temperature instead of repeating the same request, giving the
        # model an actual second chance rather than a guaranteed re-failure.
        for temp in (0.0, 0.4):
            try:
                return _compare_llm(document, claims, llm, temperature=temp)
            except LLMUnavailable as exc:
                last_exc = exc
        result = _compare_rules(document, claims, profile)
        result.fallback_reason = str(last_exc)
        return result
    return _compare_rules(document, claims, profile)


def compare_documents(documents: list[dict], claims: list[Claim], profile: dict,
                      llm: LLMClient | None = None) -> list[DocumentComparison]:
    claiming = [c for c in claims if c.status == "claiming"]
    return [compare_document(d, claiming, profile, llm) for d in documents]


def summarize_by_claim(comparisons: list[DocumentComparison], claims: list[Claim]
                       ) -> dict[str, dict]:
    """Roll up per-document verdicts into the answer the law firm needs:
    which claims are exposed by real prior art, which appear clear."""
    summary: dict[str, dict] = {c.id: {"claim": c.text, "anticipates": [],
                                       "partial": [], "status": "clear"}
                                for c in claims if c.status == "claiming"}
    for comp in comparisons:
        for v in comp.verdicts:
            if v.claim_id not in summary or v.verdict == "unrelated":
                continue
            summary[v.claim_id][v.verdict].append({
                "application_number": comp.application_number,
                "title": comp.title, "quote": v.quote, "reasoning": v.reasoning,
                "method": comp.method,
                "verified": comp.method == "llm",
            })
    for cid, s in summary.items():
        verified_anticipates = [h for h in s["anticipates"] if h["verified"]]
        verified_partial = [h for h in s["partial"] if h["verified"]]
        s["status"] = ("exposed" if verified_anticipates else
                       "watch" if verified_partial else
                       "watch_unverified" if (s["anticipates"] or s["partial"]) else
                       "clear")
    return summary
