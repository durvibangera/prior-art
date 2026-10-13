"""Turn a claim into a safe, generic search question -- InPASS rows included.

The natural-language question is a nicety for the operator to read intent at a
glance; the PORTAL ROWS are what actually get submitted, and those are always
built deterministically in code, never left to the LLM's judgement. An LLM can
be told not to reveal the inventive combination and comply nine times out of
ten -- that tenth time is a disclosure, so the row-safety check reuses the same
structural guard as the query planner (max facets per query, no chaining the
distinguishing method facets together) and runs regardless of which path
produced the question.

Two facets always anchor the query -- modality and platform, i.e. the
application context ("acoustic", "drone") -- plus at most one claim-specific
facet. That single extra facet is deliberately generic public vocabulary
(e.g. "wavelet scattering" alone, not "wavelet scattering + transformer +
statistically guided fusion" together): see PLAN.md 3 for why this is safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Claim
from .llm import LLMClient, LLMUnavailable
from .planner import MAX_FACETS_PER_QUERY, QueryPlanner

ANCHOR_FACETS = ("modality", "platform")


@dataclass
class GeneratedQuestion:
    claim_ids: list[str]             # claims this one search covers
    question: str
    rows: list[tuple[str, str]] = field(default_factory=list)
    facets_used: list[str] = field(default_factory=list)
    note: str = ""


# How many distinct technical angles to search per claim. A claim naming both
# wavelet scattering and CNN/LSTM needs to be searched two ways: narrowly (does
# prior art use OUR specific transform?) and broadly (does prior art solve this
# with ANY ML architecture, wavelet-scattering or not?). Picking only the
# single highest-weighted facet collapses every claim onto the same narrow
# query and silently drops the broader competitive-landscape search -- which
# matters, since the disclosure's own prior-art section names exactly that
# kind of competitor (MFCC/CNN detectors with no wavelet scattering at all).
MAX_ANGLES_PER_CLAIM = 3


def _claim_search_facet_sets(claim: Claim, profile: dict) -> list[list[str]]:
    """One facet-set per distinct technical angle this claim touches.

    Each returned set is still anchors + AT MOST ONE technical facet -- the
    safety cap doesn't loosen, there are just more single-facet queries now
    instead of one facet winning by weight and the rest being dropped.
    """
    anchors = [f for f in ANCHOR_FACETS if f in profile["facets"]]
    slots_left = MAX_FACETS_PER_QUERY - len(anchors)
    if slots_left <= 0:
        return [anchors]

    extra = [f for f in claim.facets
             if f not in ANCHOR_FACETS and f in profile["facets"]]
    if not extra:
        return [anchors]  # nothing to narrow by - the baseline novelty sweep

    ranked = sorted(extra, key=lambda f: profile["facets"][f]["weight"], reverse=True)
    return [anchors + [f] for f in ranked[:MAX_ANGLES_PER_CLAIM]]


def _rows_for_facets(facets: list[str], profile: dict, field_code: str = "ABS"
                     ) -> list[tuple[str, str]]:
    planner = QueryPlanner(profile, field_code=field_code)
    rows = []
    for f in facets:
        rows.extend(planner._facet_rows(f))
    return rows


def _fallback_question_text(claim: Claim, profile: dict, facets: list[str]) -> str:
    labels = [profile["facets"][f]["label"].lower() for f in facets]
    return f"What prior art exists combining {', '.join(labels)}?"


QUESTION_PROMPT_SYSTEM = """You write ONE short, generic patent-search question
for a claim, in the style of a search engine query someone would type to find
prior art -- NOT a description of the invention.

Rules, strictly:
- Use only generic, publicly common technical terms. Never use a specific
  named technique that is the actual point of novelty (e.g. write "signal
  transform" or "feature extraction technique", never name the specific
  transform if that is the inventive step).
- Do not combine more than two technical concepts in one question.
- Do not restate the claim text. Generalise it into a question about whether
  prior art exists for that KIND of technique, in that KIND of application.
- One sentence. No preamble, no explanation. Return ONLY the question text.
"""


def _make_question(facets: list[str], profile: dict, llm: LLMClient | None
                   ) -> tuple[str, str]:
    """Returns (question_text, note). Isolated so it runs once per distinct
    facet set, not once per claim -- several claims often share one query."""
    if llm is not None:
        try:
            # The LLM sees only the abstracted facet labels, never a claim's
            # source_quote -- it cannot leak what it was never given.
            labels = [profile["facets"][f]["label"] for f in facets]
            user = f"Technical concepts involved: {', '.join(labels)}."
            text = llm.chat(QUESTION_PROMPT_SYSTEM, user, max_tokens=120).strip()
            if text:
                return text, ""
        except LLMUnavailable as exc:
            return _fallback_text(facets, profile), f"LLM unavailable, used template ({exc})"
    return _fallback_text(facets, profile), ""


def _fallback_text(facets: list[str], profile: dict) -> str:
    labels = [profile["facets"][f]["label"].lower() for f in facets]
    return f"What prior art exists combining {', '.join(labels)}?"


def generate_questions(claims: list[Claim], profile: dict, llm: LLMClient | None = None
                       ) -> list[GeneratedQuestion]:
    """One question per DISTINCT facet combination, not per claim.

    Several claims routinely reduce to the same search -- e.g. every sentence
    describing a facet of the wavelet-scattering front end anchors on the same
    (modality, platform, method_core) triple. Generating one query per claim
    would resubmit that identical search repeatedly, which costs a captcha
    each time for zero additional information.
    """
    pillars = profile.get("pillars", {})
    groups: dict[tuple[str, ...], list[str]] = {}
    for claim in claims:
        # Never generate a search question FOR a claim that isn't ours to
        # claim -- searching "is this taken" only makes sense for something
        # we intend to file. An out_of_scope/not_ours claim is recorded (see
        # claims.py) but does not need a question of its own.
        if pillars.get(claim.pillar, {}).get("status") not in ("claiming",):
            continue
        for facets in _claim_search_facet_sets(claim, profile):
            key = tuple(facets)
            if claim.id not in groups.setdefault(key, []):
                groups[key].append(claim.id)

    out = []
    for facets, claim_ids in groups.items():
        facets = list(facets)
        text, note = _make_question(facets, profile, llm)
        rows = _rows_for_facets(facets, profile)
        out.append(GeneratedQuestion(claim_ids=claim_ids, question=text,
                                     rows=rows, facets_used=facets, note=note))
    return out
