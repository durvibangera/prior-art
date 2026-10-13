"""Stage A -- turn a disclosure into a structured Invention Model.

NOT a keyword list. A decomposition of the architecture into the stages data
passes through, where each stage records three things:

    function            the JOB, named so it stays true if someone else does
                        that same job a different way
    our_approach        the method THIS document uses
    known_alternatives  the methods OTHERS use for that same job

`known_alternatives` is the field that makes competitor search possible. To
find your real competitors you must search for what THEY use, not what you
use: an MFCC+CNN acoustic drone detector is the closest real competitor to a
wavelet-scattering invention, and no wavelet search will ever surface it.

Five calls, not one (see PLAN 'Stage A'):
    A1   pipeline skeleton          1 call
    A2   component card             1 call per component
    A2b  alternatives from the disclosure's OWN prior-art section
    A3   rarity -- MEASURED against the portal, not asked (core/rarity.py)
    A4   self-consistency -- run A1 twice and compare

Prompt devices used throughout, each aimed at a failure this model actually
exhibited on this task:
  * decoy slot      -- never instruct exclusion; give the excluded item its own
                       slot (FIELD1 = ours) and drop it in code
  * numbered slots  -- SETTING1..4 cannot return eleven items; a list can
  * off-domain example -- a conveyor-belt example cannot bleed audio vocabulary
  * template anchors -- "takes X and produces Y", asserted by regex
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .ingest import Disclosure
from .llm import (
    FieldSpec, KVResult, LLMClient, RecordSpec, contains_all, max_words,
    menu_key, non_empty,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "A-v1"

# The four IP pillars. Defaults to this project's taxonomy (from the meeting),
# but is a parameter everywhere so a different invention can supply its own.
DEFAULT_PILLARS: dict[str, dict] = {
    # Labels are deliberately explicit about what each pillar covers. With the
    # bare label "Acoustic signal processing", the model filed the wavelet
    # front-end here purely because the word "acoustic" appeared -- but this
    # pillar means the custom electronic hardware (which was never built),
    # and any software feature extraction belongs under the ML pillar.
    "1": {"key": "acoustic_signal_processing",
          "label": "Acoustic signal processing implemented in custom electronic "
                   "hardware such as a board, card or FPGA",
          "status": "out_of_scope"},
    "2": {"key": "ai_ml_classification",
          "label": "Software feature extraction, machine learning or "
                   "classification architecture",
          "status": "claiming"},
    "3": {"key": "counter_drone_interception",
          "label": "Counter-drone interception",
          "status": "not_ours"},
    "4": {"key": "secure_data_handling",
          "label": "Secure data handling",
          "status": "claiming"},
}

SKELETON_SECTIONS = ("title", "nature", "purpose", "technical_advancement",
                     "inventive_features")
CARD_SECTIONS = ("description", "inventive_features", "hardware")
PRIOR_SECTIONS = ("prior_art", "deficiencies")

# Pillar coverage must read only what THIS invention does. Measured on the real
# disclosure, including every section scored `secure_data_handling` as "thin" on
# the strength of three "authentication" hits that live in the B&P form's
# boilerplate TV-subscription appendix, plus one "privacy" that appears in the
# prior-art section describing why drones are a public concern. Neither is a
# mechanism this invention has. `prior_art`/`deficiencies` describe OTHERS'
# work; `examples`/`declaration` are form template text.
CONTENT_SECTIONS = ("nature", "purpose", "technical_advancement", "description",
                    "inventive_features", "hardware", "advantages")

MIN_ALTERNATIVES = 3      # below this, flag the component as weak
STABILITY_MIN_MATCH = 4   # A4: components that must agree across two runs


# ----------------------------------------------------------------- data shapes


@dataclass
class Alternative:
    name: str
    source: str            # disclosure | model_memory

    def __hash__(self):
        return hash(self.name.lower())


@dataclass
class Component:
    id: str
    name: str
    function: str                              # the DOES line
    our_approach: str = ""
    settings: list[str] = field(default_factory=list)
    known_alternatives: list[Alternative] = field(default_factory=list)
    novelty_claim: str = ""
    pillar: str = ""                           # pillar KEY, not the menu digit
    corpus_frequency: int | None = None        # filled by A3
    rarity_rank: int | None = None
    alternatives_weak: bool = False
    # True when the card call came back with nothing to say about this stage.
    # Usually means A1 invented a stage the document does not actually
    # describe, so there is nothing to search for and downstream stages must
    # skip it rather than probe for a mechanism that was never disclosed.
    unsupported: bool = False
    evidence: str = ""              # A1's quote showing this stage exists
    evidence_verified: bool = False # does that quote appear in the document?
    misplaced: bool = False         # OURS reads like a DIFFERENT stage's job
    provenance: dict = field(default_factory=dict)

    @property
    def alternative_names(self) -> list[str]:
        return [a.name for a in self.known_alternatives]


@dataclass
class InventionModel:
    title: str = ""
    components: list[Component] = field(default_factory=list)
    pillars: dict = field(default_factory=lambda: dict(DEFAULT_PILLARS))
    pillar_coverage: dict[str, dict] = field(default_factory=dict)
    stability: str = "unknown"                 # stable | unstable | unknown
    method: str = "rules"                      # llm | rules
    notes: list[str] = field(default_factory=list)
    # Methods the disclosure itself names as existing prior art. Document-level
    # on purpose: most cannot be tied to one stage by any reliable means, and
    # the competitor probe wants the whole set regardless.
    prior_art_methods: list[str] = field(default_factory=list)

    def by_pillar(self, pillar_key: str) -> list[Component]:
        return [c for c in self.components if c.pillar == pillar_key]


# ------------------------------------------------- pillar coverage (code only)
# Deterministic. Answers "does this document actually contain anything for this
# pillar", so an empty claiming pillar is REPORTED rather than invented.
#
# The exclusions are not decoration. Measured on the real disclosure, a naive
# substring match for counter-drone terms scored 9 hits, 4 of which were the
# word "Jammu" (as in IIT Jammu) matching "jam". Whole-word matching plus an
# explicit stoplist is the difference between "this pillar has content" and a
# hallucinated search direction.

PILLAR_PROBES: dict[str, dict] = {
    "acoustic_signal_processing": {
        "terms": ["board", "card", "fpga", "asic", "microcontroller", "circuit",
                  "adc", "preamplifier", "analog", "daq"],
        "exclude": [],
    },
    "ai_ml_classification": {
        "terms": ["cnn", "lstm", "neural", "network", "classifier", "training",
                  "transformer", "feature", "model", "layer", "wavelet"],
        "exclude": [],
    },
    "counter_drone_interception": {
        # "capture" and "net" are deliberately absent: measured on the real
        # disclosure, all 6 "capture" hits were signal processing ("capture
        # temporal characteristics", "capture higher-order interactions"),
        # nothing to do with capturing a drone.
        "terms": ["jamming", "jammer", "intercept", "interception", "interceptor",
                  "neutralize", "neutralise", "neutralization", "kinetic",
                  "defeat", "entangle", "entanglement"],
        # "Jammu" is an institution name here, not jamming. "non-jammable" is a
        # property of the sensing approach, not an interception mechanism.
        "exclude": ["jammu", "jammable"],
    },
    "secure_data_handling": {
        "terms": ["encryption", "encrypted", "authentication", "authenticated",
                  "privacy", "tamper", "integrity", "federated", "confidential",
                  "signing", "keystore"],
        "exclude": [],
    },
}

RICH_THRESHOLD = 12    # distinct whole-word hits
THIN_THRESHOLD = 3


def measure_pillar_coverage(disclosure: Disclosure,
                            pillars: dict = None) -> dict[str, dict]:
    """How much does this document actually say about each pillar?

    Returns {pillar_key: {level, hits, matched}} where level is
    rich | thin | empty. A `claiming` pillar coming back `empty` is the single
    most important thing this function exists to surface.
    """
    pillars = pillars or DEFAULT_PILLARS
    text = disclosure.joined(*CONTENT_SECTIONS).lower()
    out: dict[str, dict] = {}

    for meta in pillars.values():
        key = meta["key"]
        probe = PILLAR_PROBES.get(key, {"terms": [], "exclude": []})
        excluded = set(probe.get("exclude", []))
        matched: dict[str, int] = {}
        for term in probe["terms"]:
            # Whole-word only -- this is what stops "jam" matching "Jammu".
            n = len(re.findall(rf"\b{re.escape(term)}\b", text))
            if n:
                matched[term] = n
        # Report what the exclusions actually suppressed, so the reason a
        # pillar reads "empty" is inspectable rather than mysterious.
        suppressed = {t: len(re.findall(rf"\b{re.escape(t)}\b", text))
                      for t in excluded}
        suppressed = {k: v for k, v in suppressed.items() if v}

        total = sum(matched.values())
        level = ("rich" if total >= RICH_THRESHOLD
                 else "thin" if total >= THIN_THRESHOLD else "empty")
        out[key] = {"level": level, "hits": total, "matched": matched,
                    "suppressed": suppressed, "status": meta["status"]}
    return out


# ------------------------------------------------------------------ A1 prompt

_FORMAT_BLOCK = """OUTPUT FORMAT
Write plain text lines and nothing else.
Every line starts with a field name in capital letters, then a colon, then one
space, then the value.
Write END on the last line.
Write values as plain words. Do not put quotation marks, brackets or braces
around a value. Keep each value on one line."""

A1_SYSTEM = f"""You break a technical system into the stages that data passes through, in order.

A stage is one place where data arrives in one form and leaves in another form.
Split the system into 4 to 7 stages.
Number them C1, C2, C3 and so on, in the order data flows through them.

For each stage write three lines.

NAME - three to six words naming the stage by the JOB it does, using words that
       would still be correct if a different system did that same job a
       different way.
DOES - one line in exactly this shape:
       takes <what arrives> and produces <what leaves>
       The word "takes" and the word "produces" must both appear on the line.
SEEN - copy five to fifteen words from the document, word for word, that show
       this stage is really there. Copy them exactly as they appear.

Name the job, not the brand. If the document says the system uses a Kalman
filter, the NAME is "State estimation stage". A later step records which
specific method is used. This step records only the job.

{_FORMAT_BLOCK}

Fields, repeated once per stage, in this order:
COMPONENT: <C1, C2, ...>
NAME: <...>
DOES: takes <...> and produces <...>
SEEN: <words copied from the document>

WORKED EXAMPLE
This example is about sorting recycling on a conveyor belt. It is a different
system from the one you will be given. Copy its shape. Do not copy its words.

COMPONENT: C1
NAME: Item imaging stage
DOES: takes objects moving on a belt and produces a digital image of each object
SEEN: a line-scan camera images each item as it passes
COMPONENT: C2
NAME: Object outlining stage
DOES: takes a digital image and produces one outlined region per object
SEEN: the captured frame is segmented into candidate regions
COMPONENT: C3
NAME: Material identification stage
DOES: takes an outlined region and produces a material label
SEEN: each region is classified as glass, paper or plastic
COMPONENT: C4
NAME: Diversion stage
DOES: takes a material label and produces a mechanical arm movement
SEEN: a pneumatic arm diverts the item into the matching bin
END"""

A1_SPEC = RecordSpec(
    record_key="COMPONENT",
    min_records=4,
    max_records=7,
    fields=[
        FieldSpec("COMPONENT", validate=non_empty),
        FieldSpec("NAME", validate=max_words(8)),
        FieldSpec("DOES", validate=contains_all("takes", "produces")),
        # Not required: a missing quote is a signal, not a reason to retry the
        # whole skeleton. Verified against the document in code below.
        FieldSpec("SEEN", required=False),
    ],
)


def _quote_is_real(quote: str, haystack: str, min_overlap: float = 0.6) -> bool:
    """Does this quote actually appear in the document?

    Exact substring first, then a bag-of-words fallback, because the model
    paraphrases lightly even when told to copy. A stage whose evidence cannot
    be found is one the model invented -- which is exactly what happened to the
    'Detection decision stage' that produced a 'threat alert signal' nowhere
    described in the disclosure.
    """
    q = " ".join((quote or "").lower().split())
    if len(q) < 8:
        return False
    hay = " ".join(haystack.lower().split())
    if q in hay:
        return True
    words = _content_words(q)
    if not words:
        return False
    hay_words = _content_words(hay)
    return len(words & hay_words) / len(words) >= min_overlap


def _a1_skeleton(disclosure: Disclosure, llm: LLMClient, *,
                 temperature: float | None = None) -> KVResult:
    text = disclosure.joined(*SKELETON_SECTIONS)[:4000]
    user = f"{text}\n\nWrite the stages now."
    if temperature is None:
        return llm.ask(A1_SYSTEM, user, A1_SPEC, max_tokens=900,
                       prompt_version=PROMPT_VERSION)
    # A4 uses a fixed off-ladder temperature for the comparison run.
    raw = llm.chat(A1_SYSTEM, user, temperature=temperature, max_tokens=900,
                   seed=7)
    from .llm import parse_kv
    return parse_kv(raw, A1_SPEC)


# ------------------------------------------------------------------ A2 prompt


def _a2_system(job: str, pillars: dict) -> str:
    menu = "\n".join(f"           {k} {v['label']}" for k, v in pillars.items())
    return f"""You are filling in a reference card about ONE stage of a technical system.

You are given the stage, the job it does, and a document describing one
particular system.

OURS     - the method THIS document uses to do that job. Name it in up to 25
           words. If the document does not say, write: not stated
SETTING1 - a number, parameter or setting this document gives for that method.
SETTING2 - another one.
SETTING3 - another one.
SETTING4 - another one.
           If there are fewer than four, write: none
FIELD1   - the method THIS document uses, named in 1 to 4 words.
FIELD2   - a method that OTHER systems commonly use for this same job.
FIELD3   - another method that OTHER systems commonly use for this same job.
FIELD4   - another one.
FIELD5   - another one.
FIELD6   - another one.
           Write FIELD2 to FIELD6 the way a textbook would list them in a
           survey section titled: Methods for {job}
           Name each one the way the field normally names it, in 1 to 4 words.
NEWNESS  - one line in exactly this shape:
           instead of <a method from FIELD2 to FIELD6> this system <what it does differently>
           The words "instead of" and "this system" must both appear.
PILLAR   - copy exactly one whole line from this menu, number included:
{menu}

{_FORMAT_BLOCK}

Write the fields once each, in the order listed."""


A2_SPEC = RecordSpec(fields=[
    FieldSpec("OURS", validate=max_words(30)),
    FieldSpec("SETTING1", required=False),
    FieldSpec("SETTING2", required=False),
    FieldSpec("SETTING3", required=False),
    FieldSpec("SETTING4", required=False),
    FieldSpec("FIELD1", required=False, validate=max_words(6)),
    FieldSpec("FIELD2", required=False, validate=max_words(6)),
    FieldSpec("FIELD3", required=False, validate=max_words(6)),
    FieldSpec("FIELD4", required=False, validate=max_words(6)),
    FieldSpec("FIELD5", required=False, validate=max_words(6)),
    FieldSpec("FIELD6", required=False, validate=max_words(6)),
    FieldSpec("NEWNESS", required=False,
              validate=contains_all("instead of", "this system")),
    FieldSpec("PILLAR"),
])

_NONE_TOKENS = {"none", "not stated", "n/a", "na", "nil", "-", ""}


def _clean(value: str) -> str:
    v = (value or "").strip().strip('"').strip()
    return "" if v.lower() in _NONE_TOKENS else v


# ----------------------------------------------------------------- A2b prompt

A2B_SYSTEM = """The text below is the part of a patent disclosure that describes what OTHER
people have already built before this invention.

Read it and find the methods it says other people use for one particular job.

Write one line for each method you find:
PRIOR: <the method name, copied using the text's own words>

Copy the names from the text. Do not add methods that are not in the text.
If the text names no method for this job, write: PRIOR: none

Write plain text lines and nothing else. Write END on the last line."""

A2B_SPEC = RecordSpec(fields=[FieldSpec("PRIOR", required=False, repeat="multi")])


_CONNECTIVES = re.compile(
    r"\bsuch as\b|\bincluding\b|\blike\b|\bas well as\b|\bwith\b|\bfor\b|"
    r"\bfollowed by\b|\bemploys?\b|;|,\s*and\b|,", re.I)
# Generic head-nouns that are not a method name on their own. Without this,
# "DL approaches including CNNs..." yields "DL approaches" as a search term.
_GENERIC_HEADS = {"approach", "approaches", "method", "methods", "technique",
                  "techniques", "system", "systems", "model", "models",
                  "algorithm", "algorithms", "representation", "representations",
                  "feature", "features", "spectra", "implementation"}
_NAMED_ACRONYM = re.compile(r"([A-Z][A-Za-z\-]*(?:\s+[A-Za-z\-]+){0,4})\s*\(([A-Z]{2,8})s?\)")
_LEADING_JUNK = re.compile(
    r"^(?:use of|the use of|an? |use|using|employs?|employing|based on|"
    r"an existing|existing|prior[- ]art)\s+", re.I)


def _extract_method_names(clause: str) -> list[str]:
    """Pull short method names out of a long copied sentence.

    A2b asks the model to copy method names using the document's own words,
    and it obliges by copying whole 20-word clauses like

        "Support Vector Machines (SVMs) with handcrafted acoustic features
         such as Mel-Frequency Cepstral Coefficients (MFCCs) and Linear
         Predictive Cepstral Coefficients (LPCCs)"

    which is useless as a search term. Academic and patent prose names methods
    as "Full Name (ACRONYM)", so that pattern is a reliable deterministic
    extractor -- and it yields both spellings, which matters because the portal
    matches whole words only and a document may use either.
    """
    names: list[str] = []

    for full, acro in _NAMED_ACRONYM.findall(clause):
        full = _LEADING_JUNK.sub("", full).strip(" ,.-")
        if 0 < len(full.split()) <= 6:
            names.append(full)
        names.append(acro)

    # Whatever is left after removing the "Name (ACRO)" pairs, split on
    # connectives -- this catches methods stated without an acronym.
    remainder = _NAMED_ACRONYM.sub(" ", clause)
    for piece in _CONNECTIVES.split(remainder):
        piece = _LEADING_JUNK.sub("", (piece or "").strip(" ,.;-"))
        words = piece.split()
        # Keep only short, contentful fragments that look like a method name.
        if 1 <= len(words) <= 5 and any(w[:1].isupper() or w.isupper() for w in words):
            if any(w.lower() in {"from", "this", "that", "which", "have", "been",
                                 "used", "uses", "also"} for w in words):
                continue
            # Reject a fragment that is only a generic head-noun ("DL approaches").
            contentful = [w for w in words if w.lower().strip(".,") not in _GENERIC_HEADS]
            if not contentful:
                continue
            names.append(piece)

    # Dedupe case-insensitively, keep order.
    seen, out = set(), []
    for n in names:
        k = n.lower().strip()
        if k and k not in seen and len(k) > 1:
            seen.add(k)
            out.append(n.strip())

    if not out:
        trimmed = _LEADING_JUNK.sub("", clause).strip(" ,.;-")
        if trimmed:
            out.append(" ".join(trimmed.split()[:6]))
    return out


def extract_prior_art_methods(disclosure: Disclosure,
                              llm: LLMClient) -> list[str]:
    """Extract the methods the disclosure itself says others already use.

    The highest-value call in Stage A: pure extraction, the one thing a 4B
    model does reliably, over the disclosure's own cited prior art -- which IS
    the real competitor set for its field. This disclosure cites 18 academic
    references; they beat anything the model recalls from memory.

    Called ONCE for the whole document, not once per component. Measured: when
    called per component with a differing THE JOB, the model returned the
    identical four clauses every time -- it was not conditioning on the job at
    all, so N calls bought nothing over one. Distribution to components is done
    deterministically afterwards by `_distribute_alternatives`.
    """
    text = disclosure.joined(*PRIOR_SECTIONS)[:6000]
    if not text.strip():
        return []
    user = f"TEXT\n{text}\n\nList every method the text says others use."
    result = llm.ask(A2B_SYSTEM, user, A2B_SPEC, max_tokens=600,
                     prompt_version=PROMPT_VERSION)
    raw = result.one.get("PRIOR", [])
    if isinstance(raw, str):
        raw = [raw]

    names: list[str] = []
    for clause in (_clean(x) for x in raw):
        if clause:
            names.extend(_extract_method_names(clause))
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _distribute_alternatives(methods: list[str],
                             components: list[Component]) -> dict[str, list[str]]:
    """Attach a method to a component ONLY where there is real lexical overlap.

    Deliberately high-precision and low-recall, because lexical overlap is a
    weak matcher for this: "MFCC" and "takes raw audio and produces a
    time-frequency representation" are deeply related but share no words.

    An earlier version dumped every unmatched method onto the components of
    whichever pillar happened to be largest. Measured on the real disclosure
    that put ~20 methods -- including Support Vector Machines and LPCC -- onto
    an audio *acquisition* stage, which is fabricated precision.

    Methods that match nothing stay at document level (see
    InventionModel.prior_art_methods) where the probe builder can use the
    whole set. For the `alternative` probe -- "who is using any of the known
    competing methods" -- the full list is the right input anyway.
    """
    if not components:
        return {}
    comp_words = {c.id: _content_words(f"{c.function} {c.our_approach} {c.name}")
                  for c in components}
    assigned: dict[str, list[str]] = {c.id: [] for c in components}

    for method in methods:
        mw = _content_words(method)
        scored = [(len(mw & comp_words[c.id]), c.id) for c in components]
        best = max(scored)[0] if scored else 0
        if best > 0:
            for score, cid in scored:
                if score == best:
                    assigned[cid].append(method)
    return assigned


def _merge_alternatives(from_disclosure: list[str],
                        from_model: list[str]) -> list[Alternative]:
    """Disclosure-sourced wins on collision.

    The `source` tag is carried all the way to the report on purpose: a later
    KNOWN match against a disclosure-sourced alternative means something,
    while the same match against a model-recalled one is only a hint.
    """
    merged: list[Alternative] = []
    seen: set[str] = set()
    for name in from_disclosure:
        if name.lower() not in seen:
            seen.add(name.lower())
            merged.append(Alternative(name, "disclosure"))
    for name in from_model:
        if name.lower() not in seen:
            seen.add(name.lower())
            merged.append(Alternative(name, "model_memory"))
    return merged


# Sections to retry the first stage against, in priority order, when the full
# CARD_SECTIONS blend produced a bled-in answer. `hardware` is tried alone
# first: it's short, and the theory is the real answer is drowned out by the
# much larger WST-heavy `inventive_features` text when both are given
# together. If `hardware` is empty or the retry still bleeds, fall back to
# `nature` (the broad invention summary), which mentions sensing more briefly
# but is still uncontaminated by the deep feature-extraction detail.
_NARROW_RETRY_SECTION_SETS: tuple[tuple[str, ...], ...] = (
    ("hardware",),
    ("nature",),
)


def _retry_first_stage_narrow(disclosure: Disclosure, first: Component,
                              llm: LLMClient, pillars: dict) -> None:
    """One more attempt at the first stage, with a much smaller, targeted
    slice of the document, instead of the full multi-section blend.

    Mutates `first` in place only if the retry produces an answer that is
    actually grounded in its own evidence quote (checked the same way
    `detect_job_bleed` checks it) -- never overwrites a working answer with a
    worse one, and never keeps a retry that bled again.
    """
    before = {"our_approach": first.our_approach, "settings": list(first.settings),
             "novelty_claim": first.novelty_claim}
    first.provenance["narrow_retry_attempted"] = True

    for sections in _NARROW_RETRY_SECTION_SETS:
        if not disclosure.joined(*sections).strip():
            continue                                    # nothing to retry against

        retried = _a2_card(disclosure, first.id, first.name, first.function,
                           llm, pillars, evidence=first.evidence,
                           evidence_verified=first.evidence_verified,
                           sections=sections)
        if not retried.our_approach:
            continue                                    # retry itself failed

        evidence_words = _content_words(first.evidence) if first.evidence else set()
        approach_words = _content_words(retried.our_approach)
        grounded = bool(evidence_words & approach_words) if evidence_words else True

        if grounded:
            first.our_approach = retried.our_approach
            first.settings = retried.settings
            first.novelty_claim = retried.novelty_claim
            first.unsupported = False
            first.misplaced = False
            first.provenance["narrow_retry_from"] = before
            first.provenance["narrow_retry_sections"] = sections
            first.provenance["narrow_retry_succeeded"] = True
            return

    first.provenance["narrow_retry_succeeded"] = False


def _a2_card(disclosure: Disclosure, comp_id: str, name: str, job: str,
             llm: LLMClient, pillars: dict, evidence: str = "",
             evidence_verified: bool = False,
             sections: tuple = CARD_SECTIONS) -> Component:
    text = disclosure.joined(*sections)[:5000]

    # Ground A2 in the SPECIFIC text A1 already pointed to for this stage.
    # Without this, a stage whose job is genuinely minor (raw audio capture,
    # in a document that is 90% about the feature-extraction method) gets
    # answered with the document's headline technique instead -- measured: a
    # card asked to fill in "acquisition" answered "Wavelet Scattering
    # Transform" because that's what the rest of the document talks about.
    # An anchor quote the model itself already produced is a much stronger
    # pull toward the right paragraph than the job description alone.
    anchor = (f"\n\nA DIFFERENT STEP ALREADY FOUND THIS STAGE HERE IN THE DOCUMENT\n"
             f'"{evidence}"\n'
             f"Base OURS on what is happening in the document near those words, "
             f"not on the document's overall headline technique if that is a "
             f"different stage's job."
             if evidence else "")
    user = (f"THE STAGE\n{name}\n\nTHE JOB\n{job}{anchor}\n\n"
            f"THE DOCUMENT\n{text}\n\nFill in the card now.")

    def reducer(failed: list[str]) -> tuple[str, str]:
        # Attempt 3: no worked example, only the fields that failed. Some
        # failures are CAUSED by the example when the real input does not fit
        # its shape, so shrinking the prompt is a different ask, not a repeat.
        wanted = "\n".join(f"{f}: <value>" for f in failed)
        return (f"Fill in only these fields about the stage '{name}'.\n"
                f"The job is: {job}\n\n{wanted}\n\n{_FORMAT_BLOCK}", user)

    result = llm.ask(_a2_system(job, pillars), user, A2_SPEC, max_tokens=700,
                     reduce_prompt=reducer, prompt_version=PROMPT_VERSION)
    rec = result.one

    settings = [s for s in (_clean(rec.get(f"SETTING{i}", "")) for i in range(1, 5)) if s]

    # FIELD1 is the DECOY SLOT: it holds OUR method, so the model is never
    # asked to exclude anything -- it just gets a place to put the item we
    # don't want, and code drops it. Telling it "list alternatives EXCEPT
    # ours" is exactly the negative instruction this model ignores.
    model_alts = [a for a in (_clean(rec.get(f"FIELD{i}", "")) for i in range(2, 7)) if a]

    ours = _clean(rec.get("OURS", ""))
    # Backstop for the decoy slot: drop any "alternative" that is really ours
    # restated, which happens when the model fills FIELD2+ from FIELD1.
    ours_tokens = {t for t in re.findall(r"[a-z]{4,}", ours.lower())}
    model_alts = [a for a in model_alts
                  if not (ours_tokens and
                          ours_tokens & set(re.findall(r"[a-z]{4,}", a.lower())))]

    # Disclosure-sourced alternatives are attached later, once for the whole
    # document (see extract_prior_art_methods / _distribute_alternatives).
    alternatives = [Alternative(a, "model_memory") for a in model_alts]

    pillar_digit = menu_key(pillars, rec.get("PILLAR", ""))
    pillar_key = pillars.get(pillar_digit, {}).get("key", "") if pillar_digit else ""

    # Nothing at all came back about this stage: no method, no parameters, no
    # novelty statement. The stage almost certainly is not in the document.
    unsupported = not ours and not settings and not _clean(rec.get("NEWNESS", ""))

    return Component(
        id=comp_id, name=name, function=job,
        our_approach=ours,
        settings=settings,
        unsupported=unsupported,
        evidence=evidence,
        evidence_verified=evidence_verified,
        known_alternatives=alternatives,
        novelty_claim=_clean(rec.get("NEWNESS", "")),
        pillar=pillar_key,
        alternatives_weak=False,        # decided after distribution
        provenance={"attempts": result.attempts, "temperature": result.temperature,
                    "seed": result.seed, "prompt_version": result.prompt_version,
                    "errors": result.errors,
                    # Only kept on failure -- this is what lets a failed card
                    # be diagnosed from the saved JSON instead of needing to
                    # reproduce it live. Truncated: it's for a human to read,
                    # not to feed back into another prompt.
                    "raw_response": result.raw[:500] if not result.ok else "",
                    "alternatives_from_model": len(model_alts)},
    )


# --------------------------------------------------------------- A4 stability

_STOPWORDS = {"takes", "produces", "and", "the", "a", "an", "of", "for", "to",
              "from", "into", "with", "each", "one", "that", "this", "its"}


def _content_words(text: str) -> set[str]:
    """Content words, crudely singularised.

    Singular/plural is not disagreement. Without this, one run saying
    "produces convolutional features" and another saying "produces
    convolutional feature maps" scored 0.17 Jaccard and counted as the model
    contradicting itself, which is a false alarm on the one signal we have.
    """
    words = set()
    for w in re.findall(r"[a-z]{3,}", (text or "").lower()):
        if w in _STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("es"):
            # Only strip BOTH letters for a true "-es" plural, where the base
            # word ends in a sibilant sound (box->boxes, wish->wishes,
            # church->churches). Otherwise the base word already ends in "e"
            # and adds just "s" -- microphone->microphones, feature->features,
            # phase->phases, image->images -- all common technical vocabulary
            # a blanket "-es" strip mangles into a non-word ("microphon"),
            # which silently breaks every word-overlap check in this module.
            if w[:-2].endswith(("s", "x", "z", "ch", "sh")):
                w = w[:-2]
            else:
                w = w[:-1]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def check_stability(run_a: list[dict], run_b: list[dict],
                    threshold: float = 0.34) -> tuple[str, int]:
    """Does the model agree with itself about the architecture?

    At 4B you cannot measure correctness, but you CAN measure self-agreement,
    and an unstable Stage A poisons every stage downstream. Components are
    aligned by content-word overlap of their DOES lines, greedily and without
    replacement, so one component in run A cannot match two in run B.
    """
    b_pool = [( i, _content_words(r.get("DOES", ""))) for i, r in enumerate(run_b)]
    matched = 0
    used: set[int] = set()
    for rec in run_a:
        a_words = _content_words(rec.get("DOES", ""))
        best_i, best_score = None, 0.0
        for i, b_words in b_pool:
            if i in used:
                continue
            score = _jaccard(a_words, b_words)
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= threshold:
            used.add(best_i)
            matched += 1
    # Require most stages to agree, not all of them. One reworded stage out of
    # six is normal variation; three out of six is the model not having a
    # consistent reading of the document.
    import math
    needed = min(STABILITY_MIN_MATCH, math.ceil(0.7 * len(run_a))) if run_a else 1
    level = "stable" if matched >= needed else "unstable"
    return level, matched


def detect_job_bleed(components: list) -> None:
    """Catch the FIRST stage answered with a LATER stage's computational method.

    Deliberately narrow, not a general pairwise checker -- a general version
    was tried and false-positived on C4, because this document genuinely
    reuses wavelet scattering across several real, later stages (feature
    extraction, fusion, classification all correctly build on the same core
    technique). "Shares vocabulary with another stage" cannot mean "wrong"
    once technique reuse across a pipeline is legitimate, which it usually is.

    The one structurally reliable case: the FIRST stage in a physical pipeline
    is where data enters from the outside world (a sensor, a camera, a
    microphone). By construction that stage cannot legitimately be answered
    with a computational technique that operates on already-digitized data --
    a transform cannot run before the signal exists to transform. Measured
    case: C1's job was raw audio capture, its own evidence was "highly
    sensitive microphones" (correct), but its `our_approach` came back
    "Wavelet Scattering Transform" -- verbatim the LATER stage's technique.

    Sets `.misplaced` on the first component only. Does not attempt to fix it
    -- the correct content (the hardware section) is a separate extraction
    problem the caller should surface, not guess at.
    """
    if len(components) < 2:
        return
    first, rest = components[0], components[1:]
    if not first.our_approach:
        return

    own_evidence_words = _content_words(first.evidence) if first.evidence else set()
    approach_words = _content_words(first.our_approach)
    grounded_in_own_evidence = bool(own_evidence_words & approach_words)
    if grounded_in_own_evidence:
        return

    # Confirm it is specifically a LATER, computational stage's vocabulary --
    # not just "not grounded in a possibly-thin evidence quote", which alone
    # is too weak a signal (see the C4 case in the docstring).
    for later in rest:
        later_words = (_content_words(later.our_approach or "") |
                       _content_words(later.evidence or ""))
        if len(approach_words & later_words) >= 2:
            first.misplaced = True
            return


# ------------------------------------------------------------------ top level


def build_invention_model(disclosure: Disclosure, llm: LLMClient | None = None,
                          pillars: dict | None = None,
                          check_consistency: bool = True) -> InventionModel:
    """Run Stage A end to end.

    Without a reachable LLM this returns an EMPTY model with an explicit note,
    not a fabricated one. There is no honest keyword-only way to decompose an
    architecture, and a plausible-looking fake decomposition would poison every
    downstream stage silently.
    """
    pillars = pillars or DEFAULT_PILLARS
    model = InventionModel(
        title=(disclosure.get("title") or "").strip().strip('"“”'),
        pillars=pillars,
        pillar_coverage=measure_pillar_coverage(disclosure, pillars),
    )

    if llm is None or not llm.enabled:
        model.method = "rules"
        model.notes.append(
            "No LLM endpoint: architectural decomposition was NOT attempted. "
            "Keyword rules cannot decompose an architecture, and a guessed "
            "decomposition would be worse than none. Pillar coverage below is "
            "measured deterministically and is still valid.")
        return model

    skeleton = _a1_skeleton(disclosure, llm)
    if not skeleton.records:
        model.method = "rules"
        model.notes.append(
            f"Stage A1 produced no usable stages: {skeleton.errors[:3]}")
        return model
    if not skeleton.ok:
        model.notes.append(f"A1 validation warnings: {skeleton.errors[:5]}")

    if check_consistency:
        second = _a1_skeleton(disclosure, llm, temperature=0.6)
        level, matched = check_stability(skeleton.records, second.records)
        model.stability = level
        if level == "unstable":
            model.notes.append(
                f"UNSTABLE: two runs of the architecture decomposition agreed on "
                f"only {matched} of {len(skeleton.records)} stages. Review before "
                f"trusting anything downstream.")

    full_text = disclosure.joined(*CONTENT_SECTIONS)
    for i, rec in enumerate(skeleton.records, start=1):
        comp_id = (rec.get("COMPONENT") or f"C{i}").strip()
        if not re.fullmatch(r"C\d+", comp_id):
            comp_id = f"C{i}"                       # renumber in code
        quote = (rec.get("SEEN") or "").strip()
        verified = _quote_is_real(quote, full_text)
        model.components.append(
            _a2_card(disclosure, comp_id, rec.get("NAME", ""),
                     rec.get("DOES", ""), llm, pillars,
                     evidence=quote, evidence_verified=verified))

    detect_job_bleed(model.components)
    if model.components and model.components[0].misplaced:
        _retry_first_stage_narrow(disclosure, model.components[0], llm, pillars)

    # One document-level prior-art extraction, distributed deterministically.
    # Run AFTER the retry so alternatives reflect the corrected answer, not
    # the bled-in one.
    prior_methods = extract_prior_art_methods(disclosure, llm)
    placement = _distribute_alternatives(prior_methods, model.components)
    for comp in model.components:
        mine = placement.get(comp.id, [])
        comp.known_alternatives = _merge_alternatives(
            mine, [a.name for a in comp.known_alternatives])
        comp.alternatives_weak = len(comp.known_alternatives) < MIN_ALTERNATIVES
        comp.provenance["alternatives_from_disclosure"] = len(mine)
    model.prior_art_methods = prior_methods
    if prior_methods:
        placed = sum(1 for c in model.components
                     if c.provenance.get("alternatives_from_disclosure"))
        model.notes.append(
            f"Extracted {len(prior_methods)} method names from the disclosure's "
            f"own prior-art section -- the field's real competitor set. "
            f"{placed} stage(s) matched one lexically; the rest are kept at "
            f"document level for the competitor probe.")

    model.method = "llm"

    detect_job_bleed(model.components)
    if model.components and model.components[0].misplaced:
        c = model.components[0]
        retried = c.provenance.get("narrow_retry_attempted", False)
        outcome = "even after a narrower retry" if retried else "not yet retried"
        model.notes.append(
            f"JOB BLEED: {c.id} ({c.name}) is the first (sensing) stage, but "
            f"its answer -- {c.our_approach!r} -- is a computational technique "
            f"from a LATER stage, not a sensing method ({outcome}). It should "
            f"describe hardware/sensor capture instead. Do not trust its "
            f"alternatives list for search until corrected.")
    elif model.components and model.components[0].provenance.get("narrow_retry_attempted"):
        model.notes.append(
            f"C1's job-bleed was CORRECTED by a narrower retry -- see "
            f"provenance.narrow_retry_from for what it said before.")

    # Templating detector: the same failure seen in the old comparison step,
    # where the model answered once and paraphrased that answer into every
    # remaining slot instead of re-reading per component.
    approaches = [c.our_approach.strip().lower() for c in model.components
                  if c.our_approach.strip()]
    if approaches and len(set(approaches)) <= max(1, len(approaches) // 2):
        model.notes.append(
            f"TEMPLATING: {len(approaches)} stages reported only "
            f"{len(set(approaches))} distinct method(s) between them. The model "
            f"likely answered once and repeated itself rather than reading each "
            f"stage. Treat per-stage detail as unreliable.")
    settings_sigs = ["|".join(c.settings).lower() for c in model.components if c.settings]
    if len(settings_sigs) > 2 and len(set(settings_sigs)) == 1:
        model.notes.append(
            "TEMPLATING: every stage returned an identical settings list.")

    # Surface the gaps rather than letting them pass silently.
    stated = sum(1 for c in model.components if c.our_approach)
    if stated < max(3, len(model.components) // 2):
        model.notes.append(
            f"Only {stated} of {len(model.components)} stages had a stated method "
            f"-- the model may not have read the document properly.")
    for key, cov in model.pillar_coverage.items():
        if cov["status"] == "claiming" and cov["level"] == "empty":
            model.notes.append(
                f"PILLAR GAP: '{key}' is marked as a pillar you intend to claim, "
                f"but this document describes no mechanism for it. Nothing to "
                f"search. If you intend to claim it, the disclosure needs that "
                f"section written.")
    unverified = [c for c in model.components if not c.evidence_verified]
    if unverified:
        model.notes.append(
            f"UNVERIFIED STAGE(S): "
            f"{', '.join(f'{c.id} ({c.name})' for c in unverified)} -- the "
            f"decomposition could not point to text in the document showing "
            f"these exist. Check them before they drive any search.")

    phantom = [c.id for c in model.components if c.unsupported]
    if phantom:
        names = ", ".join(f"{c.id} ({c.name})" for c in model.components if c.unsupported)
        model.notes.append(
            f"UNSUPPORTED STAGE(S): {names}. The card step found no method, no "
            f"parameters and no novelty claim for these -- the decomposition "
            f"step probably invented a stage the document does not describe. "
            f"They will be excluded from search probe generation.")

    weak = [c.id for c in model.components if c.alternatives_weak]
    if weak:
        model.notes.append(
            f"Weak alternatives for {weak} -- competitor search for those stages "
            f"will be thin.")
    return model
