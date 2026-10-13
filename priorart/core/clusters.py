"""Step 1: seed -> broad search -> deterministic clustering -> LLM labeling.

Grounds the chatbot's first guesses in real filed documents, not the model's
imagination. Grouping itself is cheap, deterministic, whole-word keyword
matching -- no LLM involved, so it can't drift into vague near-duplicate
categories the way asking a small model to summarise 200+ abstracts would
(that exact failure -- identical, templated answers -- already showed up
earlier today in a different part of this build). The LLM's only job per
cluster is to turn a handful of real snippets into one short label, anchored
by a requirement that the label's key terms are words that actually fired in
that cluster -- so it can't paraphrase into a generic buzzword untethered
from what's actually in the documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import FieldSpec, LLMClient, RecordSpec, max_words, menu_key, one_of_menu

# Generic, public, category-level vocabulary -- NOT the proprietary invention.
# These are the "camps" a real acoustic/UAV-detection filing might fall into.
# Purely a starting lens for grouping; a domain that doesn't match any of
# these still gets its documents kept in the "other" bucket rather than
# silently dropped.
APPROACH_FACETS: dict[str, list[str]] = {
    "handcrafted_features": ["MFCC", "MFCCS", "SPECTRAL", "SPECTROGRAM",
                             "CEPSTRAL", "FILTERBANK", "GMM", "SVM"],
    "deep_raw_audio": ["NEURAL", "CNN", "DEEP", "WAVEFORM", "ENDTOEND"],
    "fixed_transform_frontend": ["TRANSFORM", "WAVELET", "SCATTERING",
                                 "MULTIRESOLUTION", "GAMMATONE"],
    "sequence_model": ["LSTM", "RNN", "GRU", "SEQUENCE", "TEMPORAL",
                       "TRANSFORMER", "ATTENTION"],
    "vision_based": ["CAMERA", "VIDEO", "OPTICAL", "IMAGE", "VISUAL"],
    "thermal_based": ["THERMAL", "INFRARED"],
    "rf_based": ["RADIOFREQUENCY", "JAMMING", "TRANSMITTER", "GNSS", "SPOOFING"],
    "radar_based": ["RADAR", "DOPPLER"],
    "multi_sensor_fusion": ["FUSION", "MULTIMODAL", "MULTISENSOR", "HYBRID"],
}

MIN_CLUSTER_SIZE = 5   # below this, folded into "other" rather than kept alone

# Sensor-modality subset of the facets above, used specifically to drill a
# leading branch (like "multi sensor fusion") into which REAL combination of
# sensors its own documents actually use -- vision+thermal vs vision+rf vs
# all three, etc -- rather than one flat label hiding which combination it
# is. Acoustic is the seed domain itself, so it isn't listed separately
# here; these are the "beyond acoustic" sensors a fusion approach adds.
MODALITY_FACETS: dict[str, list[str]] = {
    k: v for k, v in APPROACH_FACETS.items()
    if k in ("vision_based", "thermal_based", "rf_based", "radar_based")
}


@dataclass
class Cluster:
    key: str                    # facet key, e.g. "handcrafted_features"
    doc_ids: list[str] = field(default_factory=list)
    label: str = ""              # filled in by the LLM labeling step
    key_terms: list[str] = field(default_factory=list)


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9]*", (text or "").upper()))


def match_facets(text: str) -> list[str]:
    """Which approach-facets does this text whole-word match?

    Deliberately the same semantics as the portal itself (whole word, no
    stemming) so a cluster's membership is consistent with what a query box
    built from its terms would actually find.
    """
    words = _word_set(text)
    return [key for key, terms in APPROACH_FACETS.items()
           if words & set(terms)]


def group_documents(docs: list[dict], min_cluster_size: int = MIN_CLUSTER_SIZE
                    ) -> dict[str, Cluster]:
    """docs: [{application_number, title, abstract}, ...] -> {facet_key: Cluster}

    A document can belong to more than one facet (e.g. both `deep_raw_audio`
    and `sequence_model`) -- clusters overlap by design, since a single real
    filing often combines more than one approach.

    `min_cluster_size` is exposed (rather than always using the module
    constant) so a caller can re-run this over a SMALLER, already-narrowed
    pool of documents with a relaxed floor -- e.g. re-clustering what fell
    into "other" the first time, where a facet that missed the floor by one
    real document (observed live: handcrafted_features at 4 documents,
    MIN_CLUSTER_SIZE=5) is worth recovering rather than staying invisible
    forever just because the broad top-level harvest happened to be one
    document short.
    """
    clusters: dict[str, Cluster] = {}
    unmatched: list[str] = []

    for d in docs:
        text = f"{d.get('title', '')} {d.get('abstract', '')}"
        hits = match_facets(text)
        if not hits:
            unmatched.append(d["application_number"])
            continue
        for key in hits:
            clusters.setdefault(key, Cluster(key=key)).doc_ids.append(
                d["application_number"])

    # Fold anything below the size floor into "other" rather than keep a
    # cluster too small to be a meaningful branch, or silently drop it.
    small = [k for k, c in clusters.items() if len(c.doc_ids) < min_cluster_size]
    if small or unmatched:
        other = Cluster(key="other")
        for k in small:
            other.doc_ids.extend(clusters.pop(k).doc_ids)
        other.doc_ids.extend(unmatched)
        if other.doc_ids:
            clusters["other"] = other

    return clusters


# Generic patent-document boilerplate that would otherwise dominate a
# frequency count without being real technical signal -- excluded so a
# "recurring word" candidate from discover_facets_from_leftovers is actually
# about domain content, not about how patents are conventionally written.
_BOILERPLATE_STOPWORDS = {
    "SYSTEM", "SYSTEMS", "METHOD", "METHODS", "APPARATUS", "DEVICE", "DEVICES",
    "MODULE", "MODULES", "UNIT", "UNITS", "MEANS", "COMPRISING", "COMPRISES",
    "COMPRISE", "INVENTION", "EMBODIMENT", "EMBODIMENTS", "PRESENT", "DISCLOSED",
    "DISCLOSURE", "PROVIDED", "PROVIDING", "PROVIDES", "CONFIGURED", "PLURALITY",
    "RESPECTIVE", "SAID", "WHEREIN", "THEREOF", "ACCORDING", "RELATES",
    "RELATING", "FIELD", "BACKGROUND", "SUMMARY", "DESCRIPTION", "CLAIM",
    "CLAIMS", "PATENT", "APPLICATION", "FIGURE", "FIGURES", "PROCESSOR",
    "PROCESSORS", "DATA", "INFORMATION", "BASED", "USING", "ANOTHER", "FIRST",
    "SECOND", "THIRD", "ALSO", "THEREBY", "THUS", "THEREFORE", "HOWEVER",
    "WHICH", "WITH", "FROM", "THAT", "THIS", "THESE", "THOSE", "WHAT", "WHEN",
    "WHERE", "THEIR", "THERE", "HAVE", "HAS", "HAD", "WILL", "ALSO",
    "INCLUDE", "INCLUDES", "INCLUDING", "LIKE", "SUCH", "USED", "ANOTHER",
    "MORE", "OTHER", "OTHERS", "NUMBER", "TYPE", "TYPES", "FORM", "FORMS",
    "PART", "PARTS", "STEP", "STEPS", "PROCESS", "OUTPUT", "INPUT", "SIGNAL",
    "SIGNALS", "VALUE", "VALUES", "TERM", "TERMS", "ANALYSIS", "TECHNIQUE",
    "TECHNIQUES", "APPROACH", "APPROACHES", "MODEL", "MODELS", "PERFORMANCE",
    "RESULT", "RESULTS", "APPLICATIONS", "COMBINATION", "COMBINATIONS",
    "SENSOR", "SENSORS", "SENSING", "MONITOR", "MONITORING", "CONTROLLER",
    "CONTROLLERS", "MICROCONTROLLER", "HARDWARE", "SOFTWARE", "COMPONENT",
    "COMPONENTS", "NETWORK", "NETWORKS", "REAL", "TIME", "REALTIME",
    # Indian patent-filing boilerplate -- "COMPLETE SPECIFICATION" is the
    # standard heading nearly every filing uses (as opposed to a provisional
    # specification), so it recurs across almost any two unrelated
    # documents and is not a real technical pattern. Caught live: these two
    # words were independently "discovered" as fake branches from 120
    # genuinely unrelated real documents that only shared this filing
    # boilerplate, and a person confidently answered a question about one of
    # them ("more like complete...") having no way to know it wasn't real.
    "COMPLETE", "SPECIFICATION", "SPECIFICATIONS", "PROVISIONAL", "PRIORITY",
    "FILED", "FILING", "CONDITIONS", "CONDITION", "NOVELTY", "PRIOR",
    # Generic prose common across almost any invention abstract regardless
    # of domain, confirmed live once abstract extraction was actually clean:
    # these still topped a from-scratch frequency scan without representing
    # any real technical distinction.
    "LEARNING", "SOLUTION", "SOLUTIONS", "PROCESSING", "COMMUNICATION",
    "COMMUNICATIONS", "THROUGH", "RESPONSE", "HIGH", "ABSTRACT",
}

# The seed/domain words themselves -- present in nearly every document by
# construction of the broad search that harvested them, so they carry no
# discriminating signal between documents.
_DOMAIN_STOPWORDS = {
    "ACOUSTIC", "ACOUSTICS", "AUDIO", "SOUND", "SOUNDS", "MICROPHONE",
    "MICROPHONES", "DRONE", "DRONES", "UAV", "UAVS", "UAS", "QUADCOPTER",
    "QUADCOPTERS", "DETECTION", "DETECTING", "DETECT", "DETECTS",
    "CLASSIFICATION", "CLASSIFYING", "CLASSIFY",
    # Synonyms for the domain itself, not a distinguishing sub-approach --
    # "AERIAL" recurs across virtually every drone-related document (it's
    # just "UAV" spelled differently) and was caught live being discovered
    # as a nonsense sub-branch of an audio-only cluster.
    "AERIAL", "UNMANNED",
    # Spelled-out synonyms of a facet term already covered elsewhere in
    # APPROACH_FACETS, where an exact/stem match against the acronym form
    # can't catch the substitution -- caught live: "CONVOLUTIONAL" was
    # discovered as a "new" sub-branch of deep_raw_audio even though "CNN"
    # is already that branch's own defining term for the same concept.
    "CONVOLUTIONAL",
}


_FACET_SANITY_MENU = {
    "1": "names a specific SENSING or DETECTION approach, technique, sensor type, or signal-processing method",
    "2": "is either generic descriptive language, or a real specific technique about something OTHER than sensing/detection",
}

FACET_SANITY_SYSTEM = f"""You read a single word that recurred across several
real patent abstracts describing inventions in one technical domain. Decide
whether this word itself names a SPECIFIC SENSING or DETECTION approach,
technique, sensor type, or signal-processing method
(like "LIDAR", "WAVELET", "THERMAL", "DOPPLER") -- as opposed to answer 2,
which covers BOTH of the following:
- generic descriptive language that could apply to almost any invention
  regardless of domain (like "EFFICIENT", "DESIGNED", "ROBUST", "ADVANCED")
- a real, specific, genuinely technical word that is NOT about how the
  system senses or detects anything -- it's about a DIFFERENT part of the
  invention entirely, such as power supply ("SOLAR"), data security or
  storage ("BLOCKCHAIN"), multi-vehicle coordination ("SWARM"), or
  communications. These are real techniques, just not a detection-approach
  distinction, and must still be answer 2 -- caught live: a word being
  genuinely technical is NECESSARY but not SUFFICIENT; it must also be
  about sensing/detection specifically.

Menu:
1 = {_FACET_SANITY_MENU['1']}
2 = {_FACET_SANITY_MENU['2']}

Do NOT explain your reasoning. Do NOT write a sentence. Your entire output
for ANSWER is a SINGLE DIGIT (1 or 2) and nothing else.

Wrong:  ANSWER: This word appears to describe a specific method, so...
Right:  ANSWER: 1

Write plain text and nothing else. Write END on the last line."""

_FACET_SANITY_SPEC = RecordSpec(fields=[
    FieldSpec("ANSWER", validate=one_of_menu(_FACET_SANITY_MENU))])


def _word_is_technical(word: str, llm) -> bool:
    """Conservative gate before trusting a discovered word as a real branch
    -- one call per candidate word, never batched (same discipline as
    interpret_hint: batching distinct judgments produces templated answers
    at this model size).

    "Genuinely technical" is necessary but not sufficient: a from-scratch
    discovery run against a real 110-document harvest turned up "SOLAR",
    "BLOCKCHAIN", and "SWARM" as accepted branches -- all real, specific,
    technical words, and all irrelevant to how the system senses or detects
    anything (they're about power supply, data security, and multi-vehicle
    coordination respectively). The gate now explicitly checks relevance to
    sensing/detection, not just genericness.

    Defaults to REJECT if the LLM is unavailable or the call fails. Showing
    the person a fake or irrelevant branch and letting them honestly answer
    a question about it happened live more than once -- that's worse than
    leaving a real pattern sitting in "other" until the endpoint is
    reachable again.
    """
    if not llm or not llm.enabled:
        return False
    user = f"WORD: {word}\nANSWER:"
    result = llm.ask(FACET_SANITY_SYSTEM, user, _FACET_SANITY_SPEC, max_tokens=40,
                     prompt_version="facet-sanity-v2")
    if not result.ok:
        return False
    return menu_key(_FACET_SANITY_MENU, result.one["ANSWER"]) == "1"


def discover_facets_from_leftovers(docs: list[dict], existing_vocab: dict[str, list[str]],
                                   min_doc_support: int = MIN_CLUSTER_SIZE,
                                   max_new_facets: int = 3, llm=None,
                                   require_llm_gate: bool = True) -> dict[str, list[str]]:
    """Finds words that recur across enough of the leftover ("other")
    documents to be a real pattern -- grounded in a frequency count over
    real document text, never the model inventing a category from nothing.

    A word clearing the frequency bar is necessary but not sufficient: a
    generic adjective like "EFFICIENT" or "DESIGNED" recurs across almost
    any patent corpus without describing a real technical distinction
    (observed live twice). `require_llm_gate` (on by default) runs each
    surviving candidate through `_word_is_technical` before accepting it;
    pass `require_llm_gate=False` only to inspect the raw frequency
    candidates directly (e.g. in a test of the frequency logic itself).

    Returns {candidate_key: [seed_term]} for at most `max_new_facets` words.
    Empty if nothing recurs enough to be a real pattern -- "other" staying
    "other" is an honest outcome, not a failure to paper over.
    """
    existing_terms = {t.upper() for terms in existing_vocab.values() for t in terms}
    excluded = existing_terms | _BOILERPLATE_STOPWORDS | _DOMAIN_STOPWORDS
    excluded_stems = {w[:5] for w in excluded if len(w) >= 5}

    def _already_covered(word: str) -> bool:
        # Exact match catches short excluded words (e.g. "AERIAL" against
        # itself); the stem check catches word-form variants of a LONGER
        # excluded term that exact matching misses -- observed live,
        # "CAMERAS" (plural) was treated as a brand new discovery even
        # though "CAMERA" (singular) is already vision_based's own term,
        # producing a nonsense "cameras" sub-branch inside an audio-only
        # branch. Same 5-letter-stem convention already used elsewhere in
        # this codebase (e.g. _words_grounded_in_seed).
        return word in excluded or word[:5] in excluded_stems

    doc_freq: dict[str, int] = {}
    for d in docs:
        text = f"{d.get('title', '')} {d.get('abstract', '')}"
        words = {w for w in _word_set(text) if len(w) >= 5 and not _already_covered(w)}
        for w in words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    candidates = sorted(
        ((w, n) for w, n in doc_freq.items() if n >= min_doc_support),
        key=lambda kv: -kv[1])

    if not require_llm_gate:
        return {w.lower(): [w] for w, _ in candidates[:max_new_facets]}

    accepted: list[str] = []
    for w, _n in candidates:
        if len(accepted) >= max_new_facets:
            break
        if _word_is_technical(w, llm):
            accepted.append(w)
    return {w.lower(): [w] for w in accepted}


SEMANTIC_GROUP_SYSTEM = """You read several short excerpts from REAL,
different, already-published patent filings that did NOT fit any of the
existing known technical categories for this domain. Find any GENUINE
shared TECHNICAL APPROACH among them -- what SENSOR or SIGNAL TYPE they use
and HOW they process it -- not just a shared application area.

A real group means the documents use the SAME kind of underlying technique
(e.g. several documents that all sense via sound/acoustic signals, even
though they're used for completely different purposes -- pest control,
noise monitoring, pipeline leaks -- still share a real technical group).

REJECT these as too broad to be a real group, even if they feel related:
- "surveillance", "monitoring", "threat detection", "specialized tasks",
  or any label that's just restating "this is a drone patent" -- true of
  nearly every document, so it distinguishes nothing
- grouping documents that use DIFFERENT sensor types together just because
  both detect or classify something (e.g. a RADAR/Doppler-based document
  and an ACOUSTIC-based document are NOT the same group even if both are
  "target classification" -- the sensing approach is what must match, not
  the end goal)

For each group of 2 or more documents that clearly share the SAME
underlying sensing/technical approach, give it a short label naming that
approach. List which document numbers belong (given with each excerpt
below). A document that does not clearly share a real technical approach
with another must be left out of every group -- do not force it in, and do
not invent a broad umbrella label just to have somewhere to put it.

GROUP: <a short label for the real shared TECHNICAL approach>
DOCS: <space separated document numbers that belong, e.g. 1 3 7>
(repeat GROUP/DOCS once per real group found. If nothing shares a real
technical approach, write nothing at all -- do not force a group.)

Write plain text lines and nothing else. Write END on the last line."""

SEMANTIC_GROUP_SPEC = RecordSpec(
    record_key="GROUP", min_records=0, max_records=10,
    fields=[FieldSpec("GROUP", validate=max_words(10)), FieldSpec("DOCS")])


def discover_semantic_groups(docs: list[dict], llm, min_group_size: int = 2
                             ) -> dict[str, list[str]]:
    """EXPERIMENTAL: reads full abstracts TOGETHER in one call (not one
    word or one document at a time) and asks the model to find genuine
    THEMATIC groupings among documents that keyword-frequency discovery
    found nothing in common for. Unlike discover_facets_from_leftovers,
    this can find a shared MEANING across documents that share no literal
    recurring word at all -- e.g. two documents about acoustic monitoring
    of different wildlife, worded completely differently.

    This is a genuine departure from the "never batch distinct judgments
    into one call" discipline used everywhere else in this codebase (that
    discipline was established against a ~4B model that produced templated,
    near-duplicate answers when asked to judge several things at once) --
    worth re-testing specifically because a materially more capable model
    is now available, not assumed safe just because it is one call.

    Returns {label: [application_number, ...]}. Empty if the LLM is
    unavailable, the call fails, or it genuinely found nothing -- never a
    forced grouping. Document membership IS grounded against the real input
    list (a hallucinated document number is silently dropped, never
    trusted) even though the LABEL text itself isn't word-grounded the way
    facet discovery is, since a genuine thematic label often won't reuse
    any single document's own words. Any group below min_group_size is
    discarded as noise, matching the size-floor discipline used everywhere
    else in this codebase.
    """
    if not llm or not llm.enabled or not docs:
        return {}

    numbered = list(enumerate(docs, start=1))
    excerpts = []
    for i, d in numbered:
        title = (d.get("title") or "")[:100]
        abstract = (d.get("abstract") or "")[:220]
        excerpts.append(f"[{i}] {title}: {abstract}")

    user = "EXCERPTS\n" + "\n".join(excerpts) + "\n\nWrite the groups now."
    result = llm.ask(SEMANTIC_GROUP_SYSTEM, user, SEMANTIC_GROUP_SPEC, max_tokens=600,
                     prompt_version="semantic-group-v1")
    if not result.ok:
        return {}

    by_index = {i: d for i, d in numbered}
    groups: dict[str, list[str]] = {}
    for rec in result.records:
        label = (rec.get("GROUP") or "").strip()
        if not label:
            continue
        try:
            indices = [int(x) for x in (rec.get("DOCS") or "").split()]
        except ValueError:
            continue
        appnos = [by_index[i]["application_number"] for i in indices if i in by_index]
        appnos = list(dict.fromkeys(appnos))  # de-dupe, keep order
        if len(appnos) >= min_group_size:
            groups[label] = appnos
    return groups


def expand_with_discovered_facets(clusters: dict[str, Cluster], docs_by_id: dict[str, dict],
                                  existing_vocab: dict[str, list[str]],
                                  min_doc_support: int = MIN_CLUSTER_SIZE,
                                  max_new_facets: int = 3, llm=None,
                                  require_llm_gate: bool = True
                                  ) -> tuple[dict[str, Cluster], dict[str, list[str]]]:
    """If the "other" bucket has enough real documents, look for recurring
    real vocabulary within it and split off new branches for whatever
    actually recurs -- rather than leaving a large "other" bucket as a dead
    end, or asking the model to invent a category from nothing. This is what
    lets the branch list actually grow over time instead of being a fixed
    set list capped at whatever facets were hand-written in advance.

    Each candidate is gated through `_word_is_technical` (see
    discover_facets_from_leftovers) before being trusted as a real branch --
    `llm` needs to be passed through for that gate to actually run.

    Returns the updated clusters dict and the newly discovered facet terms
    (key -> seed terms), so the caller can merge them into the vocabulary
    used for labeling.
    """
    other = clusters.get("other")
    if not other or len(other.doc_ids) < min_doc_support:
        return clusters, {}

    other_docs = [docs_by_id[i] for i in other.doc_ids if i in docs_by_id]
    new_vocab = discover_facets_from_leftovers(other_docs, existing_vocab,
                                               min_doc_support, max_new_facets,
                                               llm=llm, require_llm_gate=require_llm_gate)
    if not new_vocab:
        return clusters, {}

    remaining_other: list[str] = []
    for appno in other.doc_ids:
        d = docs_by_id.get(appno)
        words = _word_set(f"{d.get('title', '')} {d.get('abstract', '')}") if d else set()
        hit_keys = [key for key, terms in new_vocab.items() if words & set(terms)]
        if not hit_keys:
            remaining_other.append(appno)
        for key in hit_keys:
            clusters.setdefault(key, Cluster(key=key)).doc_ids.append(appno)

    other.doc_ids = remaining_other
    if not other.doc_ids:
        clusters.pop("other", None)
    return clusters, new_vocab


def expand_other_with_semantic_groups(clusters: dict[str, Cluster], docs_by_id: dict[str, dict],
                                      llm, min_doc_support: int = MIN_CLUSTER_SIZE
                                      ) -> dict[str, Cluster]:
    """Second pass over whatever expand_with_discovered_facets couldn't
    place -- reads the real leftover abstracts TOGETHER and looks for a
    genuine shared TECHNIQUE among them (discover_semantic_groups), rather
    than requiring one literal recurring word the way the first pass does.
    Splits off one new top-level branch per real group found.

    Run at bootstrap time (here) as well as later in the live conversation
    (dialogue.py's drill_other_bucket does the identical second pass on a
    branch that's still an "other" bucket by the time it's drilled into) so
    the FIRST branch list already benefits from it, not just a bucket that
    happens to get revisited later.

    A semantic group's label comes directly from the LLM reading the real
    documents together, so these new clusters are given that label directly
    and are NOT re-labeled by label_cluster/label_all_clusters afterward --
    label_cluster requires reusing one of a facet_vocab seed term for its
    own key, which a genuinely new semantic key has none of, so it would
    just fail validation and fall back to the key rewritten as words,
    wasting a call and quietly discarding the richer label this pass found.
    """
    other = clusters.get("other")
    if not other or len(other.doc_ids) < min_doc_support:
        return clusters

    other_docs = [docs_by_id[i] for i in other.doc_ids if i in docs_by_id]
    groups = discover_semantic_groups(other_docs, llm)
    if not groups:
        return clusters

    grouped_appnos: set[str] = set()
    for label, appnos in groups.items():
        base_key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "semantic_group"
        key, suffix = base_key, 2
        while key in clusters:
            key = f"{base_key}_{suffix}"
            suffix += 1
        key_terms = [w.upper() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", label) if len(w) >= 4]
        clusters[key] = Cluster(key=key, doc_ids=list(appnos), label=label,
                                key_terms=key_terms or [key.upper()])
        grouped_appnos.update(appnos)

    other.doc_ids = [i for i in other.doc_ids if i not in grouped_appnos]
    if not other.doc_ids:
        clusters.pop("other", None)
    return clusters


def sub_cluster_by_combination(docs: list[dict], facet_vocab: dict[str, list[str]],
                               min_cluster_size: int = MIN_CLUSTER_SIZE
                               ) -> dict[str, Cluster]:
    """Group documents by the exact COMBINATION of facets they match, not by
    single membership -- used to drill a branch into its real internal
    variation (e.g. "multi sensor fusion" is actually several different
    real combinations of which extra sensor is fused with acoustic). Only
    combinations that genuinely occur in these documents are ever produced;
    nothing is invented from a theoretical combinatorial list of every
    possible subset.

    `min_cluster_size` is exposed the same way group_documents exposes it --
    a caller drilling into an already-narrowed pool (fewer documents than
    the original broad harvest) can relax the floor, since a real pattern
    there is worth surfacing from fewer supporting documents (see
    dialogue.py's _DRILL_MIN_SUPPORT).
    """
    buckets: dict[frozenset, list[str]] = {}
    unmatched: list[str] = []
    for d in docs:
        text = f"{d.get('title', '')} {d.get('abstract', '')}"
        words = _word_set(text)
        hits = frozenset(key for key, terms in facet_vocab.items() if words & set(terms))
        if not hits:
            unmatched.append(d["application_number"])
            continue
        buckets.setdefault(hits, []).append(d["application_number"])

    clusters: dict[str, Cluster] = {}
    leftover: list[str] = list(unmatched)
    for combo, doc_ids in buckets.items():
        if len(doc_ids) < min_cluster_size:
            leftover.extend(doc_ids)
            continue
        key = "+".join(sorted(combo))
        clusters[key] = Cluster(key=key, doc_ids=doc_ids)
    if leftover:
        clusters.setdefault("other", Cluster(key="other")).doc_ids.extend(leftover)
    return clusters


LABEL_PROMPT_SYSTEM = """You read a few short excerpts from real, published
patent filings that all touch on one technical approach. You write ONE short
label describing that shared approach, in the words a search engine category
would use -- not a summary of any one filing.

LABEL - a phrase of 4 to 12 words naming the shared technical approach.
KEY_TERMS - 2 to 4 of the single words given to you below (copy them exactly,
            comma-separated) that best represent this approach.

Write plain text lines and nothing else. Write END on the last line."""


def _label_valid(seed_terms: list[str]):
    seed_upper = {t.upper() for t in seed_terms}

    def check(value: str) -> str | None:
        given = {t.strip().upper() for t in value.split(",") if t.strip()}
        if len(given & seed_upper) < 1:
            return f"must reuse at least one of the given terms {sorted(seed_upper)}"
        return None
    return check


LABEL_SPEC_TEMPLATE = lambda seed_terms: RecordSpec(fields=[
    FieldSpec("LABEL", validate=max_words(14)),
    FieldSpec("KEY_TERMS", validate=_label_valid(seed_terms)),
])


def label_cluster(cluster: Cluster, docs_by_id: dict[str, dict],
                  llm: LLMClient, sample_size: int = 5,
                  facet_vocab: dict[str, list[str]] | None = None) -> Cluster:
    """One LLM call, grounded in real snippets and the facet's own seed words.

    Falls back to a generic but honest label if the LLM is unavailable or
    fails validation -- never silently invents a confident-sounding label
    with no basis.

    `facet_vocab` defaults to the top-level APPROACH_FACETS; a drilled-down
    combination cluster (key like "rf_based+vision_based") passes the finer
    vocabulary it was built from instead, and seed terms are the union of
    every facet named in the combination.
    """
    facet_vocab = facet_vocab or APPROACH_FACETS
    seed_terms: list[str] = []
    for part in cluster.key.split("+"):
        seed_terms.extend(facet_vocab.get(part, []))
    sample_ids = cluster.doc_ids[:sample_size]
    snippets = []
    for appno in sample_ids:
        d = docs_by_id.get(appno, {})
        title = (d.get("title") or "")[:100]
        abstract = (d.get("abstract") or "")[:220]
        snippets.append(f"- {title}: {abstract}")

    if not llm or not llm.enabled or not snippets:
        cluster.label = cluster.key.replace("_", " ")
        cluster.key_terms = seed_terms[:3] or [cluster.key]
        return cluster

    user = (f"SEED TERMS: {', '.join(seed_terms)}\n\nEXCERPTS\n" +
           "\n".join(snippets) + "\n\nWrite the label now.")
    result = llm.ask(LABEL_PROMPT_SYSTEM, user, LABEL_SPEC_TEMPLATE(seed_terms),
                     max_tokens=200, prompt_version="cluster-label-v1")

    if result.ok:
        cluster.label = result.one["LABEL"].strip()
        cluster.key_terms = [t.strip().upper()
                             for t in result.one["KEY_TERMS"].split(",") if t.strip()]
    else:
        # Honest fallback, not a fabricated label.
        cluster.label = cluster.key.replace("_", " ")
        cluster.key_terms = seed_terms[:3] or [cluster.key]
    return cluster


def label_all_clusters(clusters: dict[str, Cluster], docs_by_id: dict[str, dict],
                       llm: LLMClient | None,
                       facet_vocab: dict[str, list[str]] | None = None) -> dict[str, Cluster]:
    for cluster in clusters.values():
        label_cluster(cluster, docs_by_id, llm, facet_vocab=facet_vocab)
    return clusters
