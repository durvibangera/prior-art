"""Derive filed-claim candidates from a parsed disclosure, tagged by pillar.

Two paths:

  LLM     -- reads the disclosure sections and proposes claims directly.
             Better quality: it can phrase a claim as a claim, not just lift
             a sentence, and it can merge scattered mentions of one idea.
  Rules   -- splits the inventive-features and description sections into
             sentences, keeps the ones that touch a claimable facet (reusing
             Scorer.match_facets so the vocabulary lives in one place), and
             tags each by the facet -> pillar map. Works with no endpoint.

Both paths only ever touch out_of_scope / not_ours pillars to LABEL a claim as
such, never to draft a claim there -- an "acoustic_signal_processing" or
"counter_drone_interception" claim is explicitly not something we are filing
(per the meeting), so surfacing one should read as "not ours", not compete for
a shortlist slot.

Output stays entirely local: this is pre-filing invention detail derived from
the disclosure, same handling as the profile fingerprint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Disclosure
from .llm import LLMClient, LLMUnavailable
from .scorer import Scorer

CLAIM_SECTIONS = ("inventive_features", "description", "nature")


@dataclass
class Claim:
    id: str
    pillar: str
    text: str
    status: str                      # claiming | out_of_scope | not_ours
    source_section: str = ""
    source_quote: str = ""
    facets: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    # Keep bullet-style fragments ("-WST with Transformer (not CNN)") as their
    # own sentence: splitting on '.' alone would fuse them into neighbours.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -\u2022\t") for p in parts if len(p.strip(" -\u2022\t")) > 12]


# Captions, cross-references and results narrative are not inventive steps --
# a formula walkthrough or a "Table 2 summarises..." sentence describes HOW
# something works or reports a number, it does not name a new one. Matched
# against the sentence lowercased.
_NOT_A_CLAIM = re.compile(
    r"^(figure|table|fig\.)\s*\d|"
    r"shown in figure|illustrated in figure|as shown in|as depicted in|"
    r"results and discussion|training and validation|"
    r"^(these (results|trends)|the results (generally )?show|"
    r"in conclusion|in particular,? an existing)",
    re.I,
)

# Caps keep a formula walked through in six consecutive sentences from
# becoming six near-identical claims. Diversity, not volume, is the goal --
# one representative claim per distinct technique beats fifteen restatements
# of the same one.
MAX_CLAIMS = 15
MAX_PER_GROUP = 2


def extract_claims_rules(disclosure: Disclosure, profile: dict) -> list[Claim]:
    """No-endpoint fallback: sentence-level facet matching.

    Deliberately conservative and expected to be rougher than the LLM path --
    its job is to make the tool usable with zero endpoint dependency, not to
    replace a human (or the LLM) reading the disclosure. Treat its output as a
    draft to sanity-check ingest, not a final claim set.
    """
    scorer = Scorer(profile)
    pillar_map = profile.get("facet_pillar_map", {})
    pillars = profile.get("pillars", {})

    text = disclosure.joined(*CLAIM_SECTIONS)
    candidates = []
    seen_text: set[str] = set()

    for sent in _sentences(text):
        if _NOT_A_CLAIM.search(sent.lower()):
            continue
        matched = scorer.match_facets(sent)
        claim_facets = [f for f in matched if f in pillar_map]
        if not claim_facets:
            continue
        dedupe_key = re.sub(r"\W+", "", sent.lower())[:60]
        if dedupe_key in seen_text:
            continue
        seen_text.add(dedupe_key)

        top_facet = max(claim_facets, key=lambda f: profile["facets"][f]["weight"])
        weight_sum = sum(profile["facets"][f]["weight"] for f in claim_facets)
        candidates.append((weight_sum, top_facet, sent, matched))

    # Richest sentences first, so the cap keeps the most substantive claim per
    # technique rather than whichever happened to appear first in the text.
    candidates.sort(key=lambda c: -c[0])

    claims: list[Claim] = []
    group_counts: dict[str, int] = {}
    for weight_sum, top_facet, sent, matched in candidates:
        if len(claims) >= MAX_CLAIMS:
            break
        pillar = pillar_map[top_facet]
        group_key = f"{pillar}:{top_facet}"
        if group_counts.get(group_key, 0) >= MAX_PER_GROUP:
            continue
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        claims.append(Claim(
            id=f"C{len(claims) + 1}", pillar=pillar, text=sent,
            status=pillars.get(pillar, {}).get("status", "claiming"),
            source_section="inventive_features/description", source_quote=sent,
            facets=sorted(set(matched)),
        ))
    return claims


CLAIM_PROMPT_SYSTEM = """You extract patent CLAIM CANDIDATES from a technical disclosure.

A claim candidate is one discrete inventive step -- something specific enough
that a patent examiner could compare it against a piece of prior art and say
whether it is anticipated. NOT a broad restatement of the whole invention, and
NOT a marketing benefit ("low cost", "robust") unless it names a technical
mechanism that achieves it.

You are given the four IP pillars this applicant recognises, with which are
actually being claimed. Tag every claim with exactly one pillar id from the
list given. If a claim's substance belongs to a pillar marked out_of_scope or
not_ours, still extract and tag it -- do not invent a different pillar for it
-- the caller needs to know it exists so it can be labelled correctly, not
treated as one of our claims.

Return ONLY a JSON array, no prose, no markdown fence. Each element:
{"pillar": "<pillar id>", "text": "<claim, one or two sentences, your words>",
 "source_quote": "<the exact sentence(s) from the disclosure this is based on>"}
"""


def extract_claims_llm(disclosure: Disclosure, profile: dict, llm: LLMClient) -> list[Claim]:
    pillars = profile.get("pillars", {})
    pillar_desc = "\n".join(f"- {k}: {v['label']} (status: {v['status']}) - {v['note']}"
                            for k, v in pillars.items())
    user = f"""PILLARS:
{pillar_desc}

DISCLOSURE SECTIONS:

## Nature of invention
{disclosure.get('nature')}

## Prior art (what already exists)
{disclosure.get('prior_art')}

## Inventive / innovative features
{disclosure.get('inventive_features')}

## Complete description
{disclosure.get('description')[:6000]}

Extract the claim candidates now, as a JSON array only."""

    raw = llm.chat_json(CLAIM_PROMPT_SYSTEM, user, max_tokens=3000)
    if not isinstance(raw, list):
        raise LLMUnavailable(f"expected a JSON array, got {type(raw).__name__}")

    # The LLM phrases claims in its own words, so the same facet-matching used
    # by the rules path is run over its output too -- otherwise a claim's
    # `facets` stays empty and question generation (core/questions.py) can't
    # tell WHICH technique a claim is about, and every claim collapses into
    # one generic query missing its actual distinguishing term. Matched
    # against text + source_quote together: the quote carries the original
    # document's exact vocabulary, which the paraphrased text may not repeat.
    scorer = Scorer(profile)
    claims: list[Claim] = []
    for i, item in enumerate(raw, 1):
        pillar = item.get("pillar", "")
        if pillar not in pillars:
            continue
        text = str(item.get("text", "")).strip()
        quote = str(item.get("source_quote", "")).strip()
        claims.append(Claim(
            id=f"C{i}", pillar=pillar, text=text,
            status=pillars[pillar]["status"],
            source_section="llm-extracted", source_quote=quote,
            facets=sorted(scorer.match_facets(f"{text} {quote}")),
        ))
    return claims


def extract_claims(disclosure: Disclosure, profile: dict, llm: LLMClient | None = None
                   ) -> tuple[list[Claim], str]:
    """Try the LLM path, fall back to rules. Returns (claims, method_used)."""
    if llm is not None:
        try:
            claims = extract_claims_llm(disclosure, profile, llm)
            if claims:
                return claims, "llm"
        except LLMUnavailable as exc:
            import logging
            logging.getLogger(__name__).warning("LLM claim extraction failed, "
                                                 "falling back to rules: %s", exc)
    return extract_claims_rules(disclosure, profile), "rules"
