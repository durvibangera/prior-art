"""Step 7: novelty comparison -- a separate, final, post-hoc step.

Deliberately kept apart from the live conversational narrowing (dialogue.py),
which stays vague/hint-based exactly as before and never sees this text. This
module runs only when explicitly invoked (see scripts/run_novelty_comparison.py),
after a shortlist already exists, and compares the REAL invention description
(from the disclosure docx, via ingest.parse_docx) against each shortlisted
document's REAL, already-downloaded abstract.

Confidentiality: this is the one place in the whole tool that deliberately
sends the real invention text to an LLM. It must only ever go to
`endpoints.llm` (the self-hosted intranet model) -- never to InPASS, never to
any external service. It is never wired into anything that touches the portal.

One LLM call per document, never batched: this is a harder, more open-ended
judgment than the semantic grouping in clusters.py (which reads multiple
documents together for a simpler "shared theme" task), so each comparison is
kept isolated -- one document's read can't bleed into or template another's.

Architecture-level grounding: comparing "how novel is mine" honestly requires
comparing actual architecture components (fixed transform front-end, CNN,
sequence model, etc), not just a free-text vibe. Rather than trusting the LLM
to invent this comparison from scratch, this reuses match_facets() -- the
SAME deterministic, whole-word keyword match already used everywhere else in
this codebase to decide real cluster membership -- against both the
invention's own text and each document's real abstract. The LLM is given
this grounded overlap as fact and writes around it, and the final CLOSENESS
verdict is itself capped by it (see compare_to_invention): a document with
literally zero matching architecture keywords can never be reported
VERY_CLOSE purely off the model's free-text read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .clusters import APPROACH_FACETS, match_facets
from .llm import FieldSpec, LLMClient, RecordSpec, max_words, menu_key, one_of_menu

CLOSENESS_MENU = {
    "1": "VERY_CLOSE -- uses essentially the same technique for essentially the same purpose",
    "2": "SOMEWHAT_CLOSE -- shares a real technical element, but meaningfully different overall",
    "3": "DISTANT -- only loosely related, different technique or different purpose",
}
_CLOSENESS_LABEL = {"1": "VERY_CLOSE", "2": "SOMEWHAT_CLOSE", "3": "DISTANT"}

NOVELTY_COMPARE_SYSTEM = f"""You compare a person's own invention against ONE
real, already-published prior art document, to help them judge how novel
their invention is relative to this specific document.

You are given MY INVENTION (a real, detailed technical description), PRIOR
ART (the real abstract of one already-published patent filing), and a
GROUNDED ARCHITECTURE CHECK -- an independent keyword match (not your
judgment) over a small, fixed set of known architecture categories. Absence
of a category there means that word wasn't found in the short prior-art
abstract, not proof the document truly lacks it -- abstracts are short, so
also read the actual text for anything the keyword check could have missed.

Assess honestly, using ONLY what is actually stated in each -- never invent
detail that isn't there for either side, and never assume MY INVENTION and
PRIOR ART are similar just because they're in the same general field.

Compare more than just the processing pipeline -- also consider what each is
actually FOR, what it claims to achieve, and how it's used, not only the
technical steps.

TECHNIQUE: <what the prior art document actually does, one or two sentences>
SIMILARITY: <what's genuinely similar to my invention, one or two sentences, or "none" if nothing real>
DIFFERENCE: <what's genuinely different about my invention that this prior art does not have, one or two sentences>
CLOSENESS: <copy one full menu line below, exactly>
1 = {CLOSENESS_MENU['1']}
2 = {CLOSENESS_MENU['2']}
3 = {CLOSENESS_MENU['3']}

Do NOT explain your reasoning beyond what each field above asks for. Write
plain text lines and nothing else. Write END on the last line."""

NOVELTY_COMPARE_SPEC = RecordSpec(fields=[
    FieldSpec("TECHNIQUE", validate=max_words(60)),
    FieldSpec("SIMILARITY", validate=max_words(60)),
    FieldSpec("DIFFERENCE", validate=max_words(60)),
    FieldSpec("CLOSENESS", validate=one_of_menu(CLOSENESS_MENU)),
])

_INVENTION_TEXT_SECTIONS = ("nature", "description", "inventive_features", "technical_advancement")


def invention_text_from_disclosure(disclosure_path: str) -> str:
    """The real technical description used for comparison -- content sections
    only (never applicant/declaration/boilerplate). Confidential; stays local,
    used only for the intranet LLM call in compare_to_invention.
    """
    from .ingest import parse_docx
    disclosure = parse_docx(disclosure_path)
    return disclosure.joined(*_INVENTION_TEXT_SECTIONS)


def _facet_label(key: str) -> str:
    return key.replace("_", " ")


def _facet_context_line(invention_facets: list[str], prior_art_facets: list[str]) -> str:
    shared = sorted(set(invention_facets) & set(prior_art_facets))
    invention_only = sorted(set(invention_facets) - set(prior_art_facets))
    known = ", ".join(_facet_label(k) for k in APPROACH_FACETS)
    return (
        "GROUNDED ARCHITECTURE CHECK (keyword match, not your judgment):\n"
        f"- known categories checked: {known}\n"
        f"- both my invention and this prior art appear to use: "
        f"{', '.join(_facet_label(k) for k in shared) or 'none detected'}\n"
        f"- my invention appears to use that this prior art does NOT: "
        f"{', '.join(_facet_label(k) for k in invention_only) or 'none detected'}"
    )


@dataclass
class NoveltyComparison:
    application_number: str
    title: str
    technique: str
    similarity: str
    difference: str
    closeness: str                              # VERY_CLOSE | SOMEWHAT_CLOSE | DISTANT | UNKNOWN
    method: str                                  # llm | fallback
    shared_facets: list[str] = field(default_factory=list)
    invention_only_facets: list[str] = field(default_factory=list)


def compare_to_invention(invention_text: str, document: dict, llm: LLMClient | None,
                         invention_facets: list[str] | None = None) -> NoveltyComparison:
    """One call, one document. Honest fallback (closeness UNKNOWN, empty
    fields) if the LLM is unavailable or the call fails outright -- never a
    fabricated confident-sounding comparison.

    `invention_facets` lets a caller comparing many documents compute the
    invention's own facets ONCE (match_facets(invention_text) never changes
    across documents) rather than redoing it per call; computed here if not
    given.
    """
    appno = document["application_number"]
    title = document.get("title") or ""
    abstract = document.get("abstract") or ""

    invention_facets = (match_facets(invention_text) if invention_facets is None
                       else invention_facets)
    prior_art_facets = match_facets(f"{title} {abstract}")
    shared_facets = sorted(set(invention_facets) & set(prior_art_facets))
    invention_only_facets = sorted(set(invention_facets) - set(prior_art_facets))

    if not llm or not llm.enabled:
        return NoveltyComparison(appno, title, "", "", "", "UNKNOWN", "fallback",
                                 shared_facets, invention_only_facets)

    facet_context = _facet_context_line(invention_facets, prior_art_facets)
    user = (f"{facet_context}\n\nMY INVENTION:\n{invention_text}\n\n"
           f"PRIOR ART:\n{title}: {abstract[:1500]}\n\nWrite the comparison now.")
    result = llm.ask(NOVELTY_COMPARE_SYSTEM, user, NOVELTY_COMPARE_SPEC,
                     max_tokens=400, prompt_version="novelty-compare-v2")
    if not result.ok:
        return NoveltyComparison(appno, title, "", "", "", "UNKNOWN", "fallback",
                                 shared_facets, invention_only_facets)

    rec = result.one
    closeness = _CLOSENESS_LABEL.get(menu_key(CLOSENESS_MENU, rec.get("CLOSENESS", "")), "UNKNOWN")
    method = "llm"

    # Structural cap, not just a prompt instruction: a document with literally
    # NO matching architecture keyword at all can never be reported as
    # VERY_CLOSE purely off the model's free-text read -- mirrors the same
    # "don't trust an ungrounded strongest-tier verdict alone" backstop used
    # for branch confirmation in dialogue.py's interpret_hint. This only
    # ever demotes a verdict, never promotes one -- a document CAN still be
    # genuinely DISTANT despite sharing facets (e.g. same sensor, wholly
    # different purpose), so shared facets alone never force VERY_CLOSE.
    if closeness == "VERY_CLOSE" and not shared_facets:
        closeness = "SOMEWHAT_CLOSE"
        method = "llm (closeness capped, no shared architecture keyword detected)"

    return NoveltyComparison(
        appno, title,
        rec.get("TECHNIQUE", "").strip(),
        rec.get("SIMILARITY", "").strip(),
        rec.get("DIFFERENCE", "").strip(),
        closeness, method, shared_facets, invention_only_facets)


def compare_shortlist_to_invention(invention_text: str, documents: list[dict],
                                   llm: LLMClient | None, limit: int = 50
                                   ) -> list[NoveltyComparison]:
    """Runs compare_to_invention across the shortlist, one call per document,
    capped at `limit` as a safety valve (the shortlist itself is already
    capped, typically at 50)."""
    invention_facets = match_facets(invention_text)
    return [compare_to_invention(invention_text, d, llm, invention_facets=invention_facets)
           for d in documents[:limit]]
