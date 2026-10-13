"""Derive a search profile (facets, pillars, fingerprint) from ANY disclosure.

This is what makes "upload any document" actually true. Without it, a new
document would be searched using the acoustic-drone vocabulary hand-written
for THIS invention -- wrong for anything else. See PLAN.md 7b.

Schema compatibility: three anchor facets are ALWAYS named modality, platform,
task, regardless of what the invention actually is -- every technical
disclosure has a sensing/input mode, a target/subject, and an objective. That
keeps `questions.py`'s ANCHOR_FACETS and the query planner working unmodified
for a profile about anything. Beyond those three, the LLM invents its own
facet keys for whatever distinguishes THIS invention, and names ONE of them
as the "core technique" facet, which becomes axes.B_whitespace's anchor --
the white-space question ("is this technique used anywhere else") generalises
to any domain this way, not just wavelet scattering.

IPC codes are deliberately NOT auto-generated: correctly assigning a patent
classification code needs domain expertise a small local model does not have,
and a wrong code is worse than an absent one (Axis C would silently search the
wrong technology area). Left empty for the operator to fill in if wanted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Disclosure
from .llm import LLMClient, LLMUnavailable

REQUIRED_ANCHORS = ("modality", "platform", "task")

PROFILE_SECTIONS = ("title", "nature", "prior_art", "deficiencies", "purpose",
                   "technical_advancement", "description", "inventive_features",
                   "advantages")


@dataclass
class ProposedProfile:
    title: str
    facets: dict            # matches the profile.yaml `facets:` shape
    pillars: dict            # matches `pillars:`
    facet_pillar_map: dict
    core_facet: str          # which facet key is the distinguishing technique
    fingerprint: str
    method: str = "rules"    # llm | rules


PROFILE_PROMPT_SYSTEM = """You design a SEARCH VOCABULARY for finding patent
prior art related to a technical disclosure. You are not evaluating the
invention -- you are building the word-lists a patent search tool will use.

You MUST produce exactly these THREE anchor facets, always with these exact
keys, because every technical disclosure has them:
  - "modality": how the system senses or takes its input (e.g. what kind of
    signal, sensor, or data it reads)
  - "platform": the subject/target/object the system acts on or detects
  - "task": the action or objective performed (detect, classify, monitor...)

Then propose 3 to 6 ADDITIONAL facets for the specific techniques that make
this invention distinctive (e.g. a named algorithm family, a signal-processing
method, a data-fusion approach, a deployment constraint). Pick your own short
snake_case key for each. Mark exactly ONE of these additional facets as the
"core_technique" -- but choose it for RARITY, not breadth. Ask yourself: if you
searched a patent database for each facet's words ALONE, which one would
return the FEWEST hits? A generic technique used across the whole field (e.g.
"feature fusion", "deep learning", "sensor fusion") is a BAD choice even if it
sounds important, because a white-space search on it just finds everything.
The best core_technique is the specific, unusual, less commonly seen technique
-- the one that would surprise someone if they found it used elsewhere.

For every facet give:
  - "label": short human-readable name
  - "weight": 1.0-3.0 for a supporting facet, 5.0-7.0 for the core technique
  - "words": SINGLE WORDS only (5-12 of them) -- include plurals and common
    variants explicitly (e.g. both "SENSOR" and "SENSORS"), because search
    matching is whole-word only, it will NOT find "sensors" from "sensor".
    ALL CAPS.
  - "phrases": 0-4 multi-word phrases for this concept, ALL CAPS.

Also propose 2-5 "pillars" -- the distinct areas of intellectual property this
disclosure touches (e.g. "signal processing hardware", "the ML architecture",
"data security"). Default every pillar's status to "claiming" -- a human will
correct any that should be "out_of_scope" or "not_ours" afterward, you cannot
know the applicant's business intent from the text alone. Map each additional
facet (not the three anchors) to exactly one pillar id.

Finally write a "fingerprint": a 100-200 word technical paragraph describing
the invention's mechanism, for local text-similarity scoring only.

Return ONLY this JSON object, no prose, no markdown fence:
{
  "title": "<short invention title>",
  "facets": {
    "modality": {"label":"...", "weight":3.0, "words":[...], "phrases":[...]},
    "platform": {"label":"...", "weight":3.0, "words":[...], "phrases":[...]},
    "task":     {"label":"...", "weight":2.0, "words":[...], "phrases":[...]},
    "<your_key>": {"label":"...", "weight":N, "words":[...], "phrases":[...]}
  },
  "core_technique": "<the key of your core-technique facet>",
  "pillars": {"<pillar_key>": {"label":"...", "status":"claiming", "note":"..."}},
  "facet_pillar_map": {"<facet_key>": "<pillar_key>"},
  "fingerprint": "..."
}

EXAMPLE (a different, unrelated invention, to show the shape -- do not reuse
any of its wording):
{
  "title": "Dust accumulation detection on solar panels using thermal imaging",
  "facets": {
    "modality": {"label":"Thermal sensing","weight":3.0,
      "words":["THERMAL","INFRARED","TEMPERATURE","HEAT"],"phrases":["THERMAL IMAGING"]},
    "platform": {"label":"Solar panel","weight":3.0,
      "words":["SOLAR","PANEL","PANELS","PHOTOVOLTAIC","MODULE","MODULES"],"phrases":["SOLAR PANEL"]},
    "task": {"label":"Detection","weight":2.0,
      "words":["DETECTION","DETECTING","MONITORING","ASSESSMENT"],"phrases":[]},
    "segmentation_model": {"label":"Segmentation model","weight":6.0,
      "words":["SEGMENTATION","MASK","PIXEL","CONVOLUTIONAL"],"phrases":["SEMANTIC SEGMENTATION"]},
    "edge_deploy": {"label":"Edge deployment","weight":1.5,
      "words":["EDGE","EMBEDDED","REALTIME"],"phrases":["REAL TIME"]}
  },
  "core_technique": "segmentation_model",
  "pillars": {
    "ml_architecture": {"label":"ML segmentation architecture","status":"claiming","note":""},
    "deployment": {"label":"Edge deployment method","status":"claiming","note":""}
  },
  "facet_pillar_map": {"segmentation_model":"ml_architecture","edge_deploy":"deployment"},
  "fingerprint": "A thermal-imaging based system for detecting dust accumulation..."
}
"""


def derive_profile_llm(disclosure: Disclosure, llm: LLMClient) -> ProposedProfile:
    text = disclosure.joined(*PROFILE_SECTIONS)[:8000]
    user = f"DISCLOSURE:\n{text}\n\nProduce the JSON object now."

    raw = llm.chat_json(PROFILE_PROMPT_SYSTEM, user, max_tokens=3000, temperature=0.2)
    if not isinstance(raw, dict):
        raise LLMUnavailable(f"expected a JSON object, got {type(raw).__name__}")

    facets = raw.get("facets", {})
    for anchor in REQUIRED_ANCHORS:
        if anchor not in facets:
            raise LLMUnavailable(f"model omitted required anchor facet '{anchor}'")

    # Normalise: uppercase every word/phrase (matching portal whole-word
    # semantics), ensure every facet has the fields downstream code expects.
    clean_facets = {}
    for key, f in facets.items():
        clean_facets[key] = {
            "label": str(f.get("label", key)),
            "weight": float(f.get("weight", 2.0)),
            "required": key in REQUIRED_ANCHORS,
            "words": [str(w).strip().upper() for w in f.get("words", []) if str(w).strip()],
            "phrases": [str(p).strip().upper() for p in f.get("phrases", []) if str(p).strip()],
        }

    core = raw.get("core_technique", "")
    if core not in clean_facets or core in REQUIRED_ANCHORS:
        # Fall back to the highest-weighted non-anchor facet rather than fail
        # outright -- losing the white-space axis is a degradation, not a
        # reason to discard an otherwise-usable profile.
        candidates = [k for k in clean_facets if k not in REQUIRED_ANCHORS]
        core = max(candidates, key=lambda k: clean_facets[k]["weight"]) if candidates else ""

    facet_pillar_map = {k: v for k, v in (raw.get("facet_pillar_map", {}) or {}).items()
                       if k not in REQUIRED_ANCHORS}

    return ProposedProfile(
        title=str(raw.get("title", disclosure.get("title", "Untitled"))),
        facets=clean_facets,
        pillars=raw.get("pillars", {}) or {},
        facet_pillar_map=facet_pillar_map,
        core_facet=core,
        fingerprint=str(raw.get("fingerprint", "")).strip(),
        method="llm",
    )


def derive_profile_rules(disclosure: Disclosure) -> ProposedProfile:
    """No-endpoint fallback: crude frequency-based term extraction.

    Deliberately weak and clearly labelled as such -- there is no profile yet
    to match sentences against (that's what we're building), so this can't
    reuse Scorer the way claims.py's rules fallback does. It produces a
    structurally valid but low-quality skeleton the operator must edit via
    the Vocabulary tab before trusting it.
    """
    text = disclosure.joined(*PROFILE_SECTIONS)
    # Acronyms and repeated capitalised technical terms are usually the
    # closest thing to "the technique" a keyword approach can find -- but a
    # capitalised sentence-starter ("The", "Existing", "Limited", "Figure")
    # is just common English, not a technical term, and would poison a query
    # with a near-universal word ("ABS:THE" matches almost everything).
    _STOPWORDS = {
        "THE", "THIS", "THAT", "THESE", "THOSE", "AND", "FOR", "WITH", "FROM",
        "INTO", "ONTO", "OVER", "UNDER", "WHICH", "WHERE", "WHILE", "EXISTING",
        "LIMITED", "FIGURE", "TABLE", "FURTHER", "ANOTHER", "PRESENT", "PROPOSED",
        "PROVIDING", "PROVIDE", "SUCH", "ALSO", "EACH", "THEIR", "OTHER", "THEREOF",
        "SEVERAL", "VARIOUS", "PARTICULAR", "ACCORDINGLY", "THEREFORE", "HOWEVER",
    }
    words = re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", text)
    freq: dict[str, int] = {}
    for w in words:
        up = w.upper()
        if up not in _STOPWORDS:
            freq[up] = freq.get(up, 0) + 1
    common = [w for w, n in sorted(freq.items(), key=lambda x: -x[1]) if n >= 2][:10]

    facets = {
        "modality": {"label": "Modality (needs review)", "weight": 3.0, "required": True,
                    "words": ["SYSTEM", "DEVICE", "SENSOR", "SENSORS"], "phrases": []},
        "platform": {"label": "Platform (needs review)", "weight": 3.0, "required": True,
                    "words": ["APPARATUS", "PLATFORM"], "phrases": []},
        "task": {"label": "Task (needs review)", "weight": 2.0, "required": True,
                "words": ["METHOD", "PROCESS", "DETECTION", "MONITORING"], "phrases": []},
        "core_terms": {"label": "Frequent technical terms (needs review)", "weight": 5.0,
                      "required": False, "words": common or ["TECHNIQUE"], "phrases": []},
    }
    return ProposedProfile(
        title=disclosure.get("title", "Untitled") or "Untitled",
        facets=facets,
        pillars={"main": {"label": "Primary technical contribution",
                          "status": "claiming", "note": "auto-generated, please review"}},
        facet_pillar_map={"core_terms": "main"},
        core_facet="core_terms",
        fingerprint=disclosure.joined("nature", "inventive_features")[:1000],
        method="rules",
    )


def derive_profile(disclosure: Disclosure, llm: LLMClient | None = None
                   ) -> ProposedProfile:
    if llm is not None and llm.enabled:
        try:
            return derive_profile_llm(disclosure, llm)
        except LLMUnavailable:
            pass
    return derive_profile_rules(disclosure)


def to_profile_dict(p: ProposedProfile, profile_id: str) -> dict:
    """Assemble the full shape claims.py / questions.py / scorer.py / the
    query planner expect -- same schema as the hand-written acoustic_uav.yaml,
    just populated from `p` instead of by hand."""
    return {
        "meta": {"id": profile_id, "title": p.title,
                "jurisdiction_note": "InPASS = India only. Not sufficient alone to establish novelty.",
                "derivation_method": p.method},
        "facets": p.facets,
        "pillars": p.pillars,
        "facet_pillar_map": p.facet_pillar_map,
        "ipc_codes": [],   # deliberately not auto-generated -- see module docstring
        "axes": {
            "A_novelty": {
                "description": "Application context, progressively refined by technique",
                "anchor_facets": list(REQUIRED_ANCHORS),
                "refine_facets": [k for k in p.facets if k not in REQUIRED_ANCHORS],
            },
            "B_whitespace": {
                "description": "Is the core technique used in any other domain?",
                "anchor_facets": [p.core_facet] if p.core_facet else [],
                "refine_facets": ["task"],
            },
        },
        "fingerprint": p.fingerprint,
        "known_art_controls": [],
        "presets": [
            {"name": "Core novelty sweep", "rows": list(REQUIRED_ANCHORS)},
        ] + ([{"name": f"{p.facets[p.core_facet]['label']} + application",
              "rows": [*REQUIRED_ANCHORS[:2], p.core_facet]},
             {"name": f"White space: {p.facets[p.core_facet]['label']} anywhere",
              "rows": [p.core_facet, "task"]}] if p.core_facet else []),
    }
