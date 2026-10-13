"""Step 2: the question loop.

Each turn: pick the two most plausible remaining branches (code, deterministic
-- never the model's call), ask a contrasting question (one LLM call, judgment
-free -- it only phrases two known labels into a sentence), read the human's
free-text hint, and interpret it against EACH named branch separately (one
narrow LLM call per branch, never batched -- batching this exact shape is what
produced identical answers across supposedly-distinct slots earlier in this
build).

Scores are soft and capped, so a single vague hint can never fully commit or
rule out a branch on its own -- confirming takes either one clear "yes" or
several consistent "maybe"s; the same is true in reverse for ruling something
out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .clusters import (
    APPROACH_FACETS, MIN_CLUSTER_SIZE, _BOILERPLATE_STOPWORDS, _DOMAIN_STOPWORDS, _word_set,
    discover_facets_from_leftovers, discover_semantic_groups, group_documents, label_cluster,
    match_facets, sub_cluster_by_combination,
)
from .llm import FieldSpec, LLMClient, RecordSpec, max_words, menu_key, one_of_menu

# menu digit -> score delta. A single strong confirm (1) crosses the confirm
# threshold alone; a single vague "maybe" (2) does not -- it takes about three
# in a row. Symmetric in the other direction for ruling a branch out.
SENTIMENT_MENU = {
    "1": "clearly confirms this approach",
    "2": "tentatively or vaguely suggests this approach",
    "3": "does not address this approach either way",
    "4": "tentatively suggests this is NOT the approach",
    "5": "clearly rules this approach out",
}
SENTIMENT_DELTAS = {"1": 2.0, "2": 0.7, "3": 0.0, "4": -0.7, "5": -2.0}

SCORE_CAP = 5.0
CONFIRM_THRESHOLD = 2.0
PRUNE_THRESHOLD = -2.0


# -------------------------------------------------------- pick candidates


def pick_open_candidates(live_clusters: list[dict],
                         doc_counts: dict[str, int] | None = None,
                         k: int = 2, exclude_ids: set[str] | None = None) -> list[dict]:
    """The top-K live clusters by score. Empty signals convergence -- the
    caller should treat that as done, not an error. Exactly ONE remaining
    live candidate is still returned (not treated as convergence) -- it
    still deserves a genuine open question grounded in its own documents
    before the conversation ends, rather than being silently left
    unresolved just because nothing remains to explicitly contrast it
    against (observed live: the last live branch -- genuinely relevant --
    never got asked about at all once everything else confirmed or was
    pruned, because the loop required at least 2 live candidates to
    proceed).

    `k` defaults to 2, not more: REAL FAILURE, observed live with 3 tied
    live branches -- pick_open_candidates returned all 3, but the BABY-tier
    question prompt is a strict two-way "closer to A or B?" contrast by
    design, so the model silently phrased the question around only 2 of
    them while the 3rd (a 57-document "other" bucket, itself covering many
    real sub-approaches) was still scored and PRUNED OUTRIGHT in the same
    turn -- from a reply to a question it was never actually shown as part
    of. The person never got a chance to say anything about it at all. A
    strict pairwise batch, every turn, at every tier, means what gets
    SCORED always matches what the person actually saw and answered -- the
    same root cause was already softening a related failure elsewhere (one
    "yes" confirming four simultaneously-bundled, substantively unrelated
    "other"-bucket branches at once, each only weakly grounded).

    Every cluster starts at score 0.0, so the very first question in any
    conversation is a tie across the board -- `doc_counts` breaks that tie by
    how many real filed documents actually support each cluster, so the
    opening question targets the best-evidenced branches rather than an
    arbitrary sort order.

    `exclude_ids` (e.g. clusters the person just declined to discuss) is
    skipped when there's still enough left to ask about instead -- but never
    at the cost of returning nothing: if every live cluster was recently
    declined, they're used anyway, since re-asking eventually beats the
    conversation silently stalling forever.
    """
    doc_counts = doc_counts or {}
    ranked = sorted(live_clusters,
                    key=lambda c: (-c["score"], -doc_counts.get(c["cluster_id"], 0)))
    if exclude_ids:
        filtered = [c for c in ranked if c["cluster_id"] not in exclude_ids]
        if filtered:
            ranked = filtered
    return ranked[:k]


# --------------------------------------------------------- question phrasing


# Four tiers, not two -- named directly: "ask me slow and steady... baby
# level... toddler... kid... teenager... start slow". The earlier 2-tier
# design (SIMPLE/DEEP) still let the very first question ("give me a
# high-level overview of your general approach") invite a full free-text
# paragraph of real technical detail -- exactly the over-disclosure this
# whole tool exists to avoid. Every tier below is CLOSED and reactive
# instead: the fair answer is a short pick, yes/no, or "sort of, but X" --
# never an invitation to describe your own approach unprompted. This also
# restores the ORIGINAL "this or that" contrast style at the start of a
# conversation (deliberately moved away from once, per an earlier request
# for open-ended questions -- reversed again here because open-ended
# questions were, in practice, exactly what caused the over-disclosure).
#
# BABY   -- the very first question of the whole conversation. Category
#           labels only, no document text at all: "closer to A or B?"
# TODDLER -- still closed, but may reference the GENERAL shape of what real
#           filings do (not named techniques yet).
# KID    -- describes ONE real document's actual technique at a moderate
#           level of detail, asks "do you do something like this?"
# TEENAGER -- describes a real document's technique with full specific
#           named methods (only reached once a branch has actually been
#           drilled into finer real sub-approaches, i.e. real context has
#           accumulated) -- still closed ("does yours do something
#           resembling this, or is it different?"), never "describe yours".
OPEN_QUESTION_SYSTEM_BABY = """You write ONE extremely simple, closed OPENING
question for a conversation where a person is describing their own
technical project, without naming their exact proprietary technique. You
are given a couple of short CATEGORY LABELS naming different real approaches
used in this general domain (not document text).

This is the VERY FIRST question of the whole conversation -- keep it as
easy and low-effort as possible. Phrase it as a plain "is your approach
closer to [label A], or [label B]?" contrast, in everyday words, not jargon.
The only fair answers are something like "the first", "the second",
"neither", or "not sure yet". Do NOT invite the person to describe their
own approach, list steps, or explain anything -- if a fair answer would be
longer than a few words, the question is too open.

QUESTION: <the question, ONE short sentence, ending with the two closed options named plainly>

Write plain text and nothing else. Write END on the last line."""

OPEN_QUESTION_SYSTEM_TODDLER = """You write ONE simple, CLOSED question for
a conversation where a person is describing their own technical project,
without naming their exact proprietary technique. You are given a few short
excerpts from REAL, already-published, DIFFERENT patent filings -- each
pursuing a different real approach in the same general domain.

Phrase it as "is it closer to [the rough general shape of one real filing],
or [the rough general shape of another]?" -- using only the GENERAL shape of
what the real filings do (e.g. "extracts specific features first" vs "feeds
the raw signal straight into a model"), not named transforms or model types
yet. The fair answer is a short pick or "neither" -- do NOT invite the
person to describe their own pipeline in their own words at length.

QUESTION: <the question, at most 2 sentences>

Write plain text and nothing else. Write END on the last line."""

OPEN_QUESTION_SYSTEM_KID = """You write ONE CLOSED, comparative question for
a conversation where a person is describing their own technical project,
without asking them to reveal their exact proprietary technique. You are
given a few short excerpts from REAL, already-published, DIFFERENT patent
filings.

Pick ONE of the given real filings and describe, in plain words, roughly
WHAT it actually does -- a moderate level of detail (e.g. "extracts specific
engineered features first" or "feeds the raw signal into a neural network
directly"); you do not need to quote it verbatim. Then ask the person
whether their own approach does something LIKE that, or something
different -- phrased so a short reaction (yes / no / sort of / different
how) is a complete, sufficient answer. Do NOT ask them to describe their own
approach unprompted or in detail.

QUESTION: <the question, at most 3 sentences>

Write plain text and nothing else. Write END on the last line."""

OPEN_QUESTION_SYSTEM_TEENAGER = """You write ONE CLOSED, comparative
question for a conversation that has already built real confidence about
the general shape of a person's approach, and is now narrowing into
specifics -- real technical detail is earned at this point. You are given a
few short excerpts from REAL, already-published, DIFFERENT patent filings,
all within the SAME narrower branch already established.

Pick ONE of the given real filings and describe, in plain words, its ACTUAL
specific technique in real detail -- name the actual kind of transform,
model, or processing step it uses if the excerpt says so. Then ask the
person whether their own approach does something resembling THAT specific
technique, or is different -- phrased so a short reaction (yes / no / close
but different in some way) is a complete, sufficient answer. Do NOT ask them
to freely describe or explain their own pipeline unprompted.

QUESTION: <the question, at most 3 sentences>

Write plain text and nothing else. Write END on the last line."""

OPEN_QUESTION_SPEC = RecordSpec(fields=[FieldSpec("QUESTION", validate=max_words(90))])

TIER_BABY, TIER_TODDLER, TIER_KID, TIER_TEENAGER = "baby", "toddler", "kid", "teenager"


def _candidate_depth(candidates: list[dict]) -> int:
    """Max nesting depth among the given candidates: 0 = top-level,
    1 = one level drilled, 2+ = deeper. cluster_id encodes depth directly
    via its parent__child construction (e.g. "deep_raw_audio__xgboost" is
    depth 1), so this needs no extra lookup."""
    return max((c["cluster_id"].count("__") for c in candidates), default=0)


def _question_tier(is_first_question: bool, candidates: list[dict]) -> str:
    """baby -> toddler -> kid -> teenager, matching how much real document
    detail the question may reference and how closed the invited answer
    must stay. The very first question of the whole conversation is ALWAYS
    baby, regardless of anything else -- it used to ask "give me a
    high-level overview of your general approach", which alone produced a
    full paragraph of real technical detail (named directly: "i shouldnt be
    disclosing as much info as i did... the questions were framed like
    that"). After that, tier tracks how deep the CURRENT candidates are
    drilled -- a branch already split into finer real sub-approaches has
    earned more specific questions; one that hasn't stays generic no matter
    how many turns have passed.
    """
    if is_first_question:
        return TIER_BABY
    depth = _candidate_depth(candidates)
    if depth == 0:
        return TIER_TODDLER
    if depth == 1:
        return TIER_KID
    return TIER_TEENAGER


def _flat_open_question(labels: list[str]) -> str:
    """Closed, "this or that" fallback for every tier -- the safest
    possible question when the LLM is unavailable, regardless of how deep
    the real one would have gone: naming the real candidates plainly and
    asking for a pick is never an over-disclosure risk the way an open
    "describe your approach" fallback would be."""
    if len(labels) == 1:
        return f"Just a quick check -- is your approach closer to {labels[0]}, or something else entirely?"
    joined = ", ".join(labels[:-1]) + " or " + labels[-1]
    return f"Just a quick pick to start -- is your approach closer to {joined}, or none of those?"


def generate_open_question(store, conversation_id: str, candidates: list[dict],
                           llm: LLMClient | None, is_first_question: bool = False,
                           sample_size: int = 2) -> tuple[str, str]:
    """Builds ONE closed, reactive question, grounded in real document
    snippets from the current top candidate branches -- the fair answer at
    every tier is a short pick or reaction, never an invitation to describe
    the person's own approach unprompted (see _question_tier).

    WHICH clusters to draw from was already decided in code by
    pick_open_candidates; the model's only job is phrasing. Falls back to a
    flat, still-closed prompt (naming the real candidates, never inventing
    detail) if the LLM is unavailable or fails.
    """
    tier = _question_tier(is_first_question, candidates)
    labels = [c["label"] for c in candidates]
    if not llm or not llm.enabled:
        return _flat_open_question(labels), "template"

    if tier == TIER_BABY:
        # No document excerpts at all -- pure category-label contrast, kept
        # as easy and low-effort as the very first question can be.
        system = OPEN_QUESTION_SYSTEM_BABY
        user = "CATEGORY LABELS\n" + "\n".join(f"- {l}" for l in labels) + "\n\nWrite the question now."
    else:
        snippets = []
        for c in candidates:
            docs = store.docs_for_cluster(conversation_id, c["cluster_id"])[:sample_size]
            for d in docs:
                title = (d.get("title") or "")[:100]
                abstract = (d.get("abstract") or "")[:200]
                snippets.append(f"- ({c['label']}) {title}: {abstract}")
        if not snippets:
            return _flat_open_question(labels), "template"
        system = {TIER_TODDLER: OPEN_QUESTION_SYSTEM_TODDLER,
                 TIER_KID: OPEN_QUESTION_SYSTEM_KID,
                 TIER_TEENAGER: OPEN_QUESTION_SYSTEM_TEENAGER}[tier]
        user = "EXCERPTS\n" + "\n".join(snippets) + "\n\nWrite the question now."

    result = llm.ask(system, user, OPEN_QUESTION_SPEC, max_tokens=220,
                     prompt_version=f"open-question-{tier}-v1")
    if not result.ok:
        reason = "; ".join(result.errors[:1]) or "unknown"
        if "parsed 0" in reason and result.raw:
            reason += f" | raw: {result.raw[:200]!r}"
        return _flat_open_question(labels), f"template ({reason})"

    return result.one["QUESTION"].strip(), "llm"


# ------------------------------------------------------ hint interpretation


HINT_SYSTEM = f"""You read one reply from a person describing a technical
project, in a conversation where they are being deliberately vague. They were
just asked to contrast TWO candidate approaches -- given to you below as THIS
APPROACH (with its position in that question, first or second) and THE OTHER
APPROACH. Decide how the reply relates to THIS APPROACH SPECIFICALLY,
compared to the other one -- not how it relates to the general topic.

If the reply uses an ordinal like "the first", "the second", "the latter", or
names one approach directly, match it against THIS APPROACH's own position or
label, given below.

A reply that clearly points at THE OTHER approach instead is evidence AGAINST
this one (4 or 5) -- picking one specific alternative is itself a signal that
it is NOT the other one, not just silence about this one (3).

1/5 vs 2/4 is about how SPECIFIC and CONCRETE the reply is, not about
whether it uses words like "yes" or "definitely". A reply that names actual
technical specifics matching this approach (specific processing steps,
named techniques, named model types) is a CLEAR confirm (1) even if it never
says "yes" -- reserve 2 (tentative) for replies that are genuinely vague or
hedged, not for replies that are detailed but phrased casually. The same
applies in reverse for ruling something out (5 vs 4).

The REPLY is the person's own words -- it may itself be short, a single word,
or even a bare number. That number is NOT your answer; it is only something
they said. Judge what it means about THIS approach, then pick your own line
from the menu below independently.

Menu:
1 = {SENTIMENT_MENU['1']}
2 = {SENTIMENT_MENU['2']}
3 = {SENTIMENT_MENU['3']}
4 = {SENTIMENT_MENU['4']}
5 = {SENTIMENT_MENU['5']}

Do NOT explain your reasoning. Do NOT write a sentence describing the reply.
Do NOT restate what the reply said. Your entire output for SENTIMENT is a
SINGLE DIGIT (1, 2, 3, 4, or 5) and nothing else -- any words before or
after the digit are wrong, even if your reasoning is correct.

Wrong:  SENTIMENT: The reply describes a system based on training data,
        which suggests...
Right:  SENTIMENT: 1

Write plain text and nothing else. Write END on the last line."""

HINT_SPEC = RecordSpec(fields=[FieldSpec("SENTIMENT", validate=one_of_menu(SENTIMENT_MENU))])


@dataclass
class HintResult:
    cluster_id: str
    sentiment_digit: str        # "1".."5"
    delta: float
    method: str                 # llm | fallback


_UNAMBIGUOUS_NEITHER = {
    "neither", "none", "none of these", "none of those", "not either",
    "neither one", "neither of them", "neither of those",
}


def _is_unambiguous_neither(hint_text: str) -> bool:
    """A small, EXACT-match set of complete replies that unambiguously
    reject BOTH named options at once -- no interpretation needed. Deciding
    this in code rather than trusting the model's reading of negation:
    observed live, "neither" sometimes read as only weakly negative (needed
    2-3 repeats of the SAME question before both branches finally crossed
    the prune threshold), while a DIFFERENT negation case ("not visual")
    over-generalized to rule out a branch it never actually addressed. This
    model's handling of negation is unreliable in both directions, so this
    one genuinely unambiguous phrase is resolved structurally instead.

    Deliberately narrow: a hedged reply that merely CONTAINS "neither"
    ("neither really, closer to the first I guess") is NOT unambiguous and
    must still go to the model -- only an exact match to the whole reply
    qualifies.
    """
    normalized = re.sub(r"[^a-z ]", "", hint_text.strip().lower())
    return normalized in _UNAMBIGUOUS_NEITHER


# A small, EXACT-match set, same discipline as _is_unambiguous_neither: a
# hedged reply that merely CONTAINS one of these words ("I'll pass on the
# details but it's audio-based") is NOT a decline and must still be scored
# normally -- only a reply that IS one of these, in full, counts. Requested
# directly: "if i wish to not answer a question i will just say i dont want
# to and the llm should understand that it cant help it and has to move on"
# -- a decline is NOT evidence against the branch (must never be scored as
# a rule-out) and must not cause the exact same question to be re-asked
# next turn (see run_dialogue_loop's use of this with the turns log).
# "idk"/"not sure" etc are included here too -- NOT a confidentiality
# refusal the way "i dont want to answer" is, but observed live to need the
# exact same handling: an uncertain non-answer carries no real signal
# either way, and leaving it to the LLM's neutral-sentiment read means it
# still counts as "asked and answered", so the SAME unresolved branch (if
# it's the only one left not recently excluded) just gets re-asked
# verbatim next turn. Treating it as a decline -- skip scoring, exclude
# from the next pick -- gives the conversation somewhere to actually go.
_DECLINE_PHRASES = {
    "pass", "i pass", "skip", "skip that", "skip this", "skip this one",
    "next question", "no comment", "i dont want to answer",
    "i dont want to answer that", "i dont want to say",
    "i dont want to say that", "id rather not", "id rather not say",
    "id rather not answer", "rather not say", "rather not answer",
    "i cant say", "i cant answer that", "i wont answer that",
    "not answering that", "cant tell you that", "cant discuss that",
    "idk", "i dont know", "i do not know", "dont know", "not sure",
    "unsure", "no idea",
}


def _is_decline(hint_text: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", hint_text.strip().lower())
    return normalized in _DECLINE_PHRASES


def _hint_grounds_cluster(hint_text: str, cluster: dict) -> bool:
    """Whether the reply actually uses (whole-word) any of THIS cluster's
    own key terms -- the same grounding discipline used everywhere else in
    this codebase (label validation, seed grounding, facet-discovery
    exclusion). A reply can still genuinely confirm a branch without ever
    using its jargon, so this is not a hard requirement to confirm at all --
    see interpret_hint's use of it, which only caps an UNGROUNDED strongest-
    tier read rather than rejecting it outright."""
    hint_words = _word_set(hint_text)
    term_words = {t.upper() for t in (cluster.get("key_terms") or [])}
    return bool(hint_words & term_words)


def interpret_hint(hint_text: str, cluster: dict, llm: LLMClient | None,
                   other_label: str | None = None, position: str | None = None) -> HintResult:
    """One narrow call per cluster. Never batched -- see module docstring.

    `other_label`/`position` give this call the CONTRASTING approach and this
    cluster's own ordinal slot in the question that was actually asked.
    Without this, a reply like "second I guess" or "not the first one" is
    unresolvable in isolation, and a reply that clearly favors the OTHER
    named approach reads as merely "doesn't mention this one" (neutral)
    instead of "leans away from this one" -- both observed live against a
    real conversation, and both are why a vague-but-clearly-comparative
    reply repeatedly failed to move the score enough to stop the same
    question being re-asked 2-3 times in a row.
    """
    if _is_unambiguous_neither(hint_text):
        return HintResult(cluster["cluster_id"], "5", SENTIMENT_DELTAS["5"], "rule:neither")

    if not llm or not llm.enabled:
        return HintResult(cluster["cluster_id"], "3", 0.0, "fallback")

    this_line = f"THIS APPROACH{f' ({position})' if position else ''}: {cluster['label']}"
    other_line = f"OTHER APPROACH: {other_label}" if other_label else ""
    user = "\n".join(l for l in (this_line, other_line, f'REPLY: "{hint_text}"') if l) + "\nSENTIMENT:"
    result = llm.ask(HINT_SYSTEM, user, HINT_SPEC, max_tokens=60,
                     prompt_version="hint-v2")
    if not result.ok:
        # Honest neutral fallback -- never guess a direction with no basis.
        # Reason is attached so a real endpoint/config problem is visible
        # instead of looking identical to a genuine "model was ambiguous".
        reason = "; ".join(result.errors[:1]) or "unknown"
        if "parsed 0" in reason and result.raw:
            reason += f" | raw: {result.raw[:200]!r}"
        return HintResult(cluster["cluster_id"], "3", 0.0, f"fallback ({reason})")

    digit = menu_key(SENTIMENT_MENU, result.one["SENTIMENT"])
    if digit is None:
        return HintResult(cluster["cluster_id"], "3", 0.0, "fallback (menu_key unresolved)")

    # Structural backstop, not just a prompt instruction: if the person's own
    # reply IS that exact digit, an "answer" of the same digit is
    # indistinguishable from the model echoing the input back rather than
    # reasoning about it -- observed live (hint "1" against two different
    # clusters both came back "1"). A negative instruction in the prompt
    # ("don't just echo it") is not something a 4B model reliably follows, so
    # this is caught here rather than trusted.
    if hint_text.strip() == digit:
        return HintResult(cluster["cluster_id"], "3", 0.0,
                          "fallback (indistinguishable from echoing the reply's own digit)")

    # Structural backstop, not just a prompt instruction: a "clearly
    # confirms" read (1) is enough to fully confirm a branch from ONE turn
    # alone (see CONFIRM_THRESHOLD). Observed live on qwen3-max: a reply
    # describing a plain CNN pipeline -- no LSTM/RNN/GRU/sequence/temporal/
    # attention anywhere in it -- still scored a full "1" confirm for the
    # "sequence_model" branch, single-handedly confirming a branch the
    # reply never actually supported. Capping an UNGROUNDED "1" down to the
    # same delta as a tentative "2" means it still counts as real positive
    # signal, just never enough alone to confirm -- it takes either a
    # grounded confirm, or several ungrounded ones in a row, matching the
    # "no single vague answer locks anything in" design this loop already
    # uses everywhere else. Only "1" is capped this way -- "5" (rule out)
    # is trusted as-is, since a generic rejection ("no, not that") genuinely
    # doesn't need to use the branch's own jargon to be a real signal.
    if digit == "1" and not _hint_grounds_cluster(hint_text, cluster):
        return HintResult(cluster["cluster_id"], digit, SENTIMENT_DELTAS["2"],
                          "llm (ungrounded confirm, capped)")
    return HintResult(cluster["cluster_id"], digit, SENTIMENT_DELTAS[digit], "llm")


# --------------------------------------------------------------- drill-down


_DRILL_MIN_SUPPORT = 3  # relaxed from MIN_CLUSTER_SIZE (5) -- a branch being
# drilled (a confirmed branch, or "other") is already a narrower, more
# specific pool than the original broad harvest, so a real sub-pattern is
# worth surfacing from just a handful of documents. Shared by both
# drill_into_cluster and drill_other_bucket, which now use the same
# discovery mechanism.


def drill_into_cluster(store, conversation_id: str, cluster: dict,
                       llm: LLMClient | None) -> list[dict]:
    """A branch that's crossed the confirm threshold may really bundle
    several distinct real sub-approaches together -- named directly in the
    approved plan: "splits into finer sub-approaches and keeps going,
    narrower each round."

    Rather than forcing the SAME fixed sensor-modality lens onto every
    branch regardless of what it actually is (observed live: an
    audio-only PROCESSING-METHOD branch got drilled by vision/thermal/
    radar vocabulary, producing meaningless children and reopening a
    dimension the seed had already closed), this discovers whatever
    ACTUALLY recurs across the branch's OWN real documents -- the exact
    same frequency-count + LLM technical/generic gate already used to
    recover genuine patterns from the "other" bucket, just applied one
    level deeper. A genuine sensor-fusion branch will still surface real
    vision/thermal/rf/radar sub-splits this way, grounded in its own real
    documents -- it just won't be FORCED onto a branch where it makes no
    sense.

    Returns the new child cluster dicts -- empty if the branch has too few
    documents, or if discovery finds fewer than 2 real sub-patterns worth
    asking about (a branch with no real internal variation stays a leaf,
    not force-split).
    """
    docs = store.docs_for_cluster(conversation_id, cluster["cluster_id"])
    if len(docs) < MIN_CLUSTER_SIZE:
        return []

    docs_by_id = {d["application_number"]: d for d in docs}
    discovered_vocab = discover_facets_from_leftovers(
        docs, APPROACH_FACETS, min_doc_support=_DRILL_MIN_SUPPORT,
        max_new_facets=4, llm=llm)
    if not discovered_vocab:
        return []

    sub = sub_cluster_by_combination(docs, discovered_vocab, min_cluster_size=_DRILL_MIN_SUPPORT)
    real = {k: c for k, c in sub.items() if k != "other"}
    if len(real) < 2:
        return []

    children = []
    for key, c in sub.items():
        child_id = f"{cluster['cluster_id']}__{key}"
        label_cluster(c, docs_by_id, llm, facet_vocab=discovered_vocab)
        store.add_cluster(conversation_id, child_id, c.label, c.key_terms,
                          parent_id=cluster["cluster_id"])
        store.link_docs_to_cluster(conversation_id, child_id, c.doc_ids, source="drilldown")
        children.append(store.get_cluster(conversation_id, child_id))

    store.set_cluster_status(conversation_id, cluster["cluster_id"], "drilled")
    return children


def drill_other_bucket(store, conversation_id: str, other_cluster: dict,
                       llm: LLMClient | None) -> tuple[list[dict], list[dict]]:
    """"Other" isn't one coherent approach -- it's whatever the broad
    harvest's per-facet size floor folded away, plus genuinely unmatched
    documents. A facet that missed the floor by ONE real document (observed
    live: handcrafted_features at 4 documents against MIN_CLUSTER_SIZE=5)
    then has no branch at all to be recognized under -- a person can
    describe it in detail and the conversation just keeps re-asking the
    same handful of unrelated candidates, because the right category was
    never in play to begin with.

    Two passes over "other"'s own documents, each catching a different
    kind of real pattern:

    1. Literal recurring-word recovery (group_documents, relaxed floor) --
       cheap, and recovers any facet that already has a name in
       APPROACH_FACETS but fell just short of the original floor.
    2. Semantic grouping (discover_semantic_groups) over whatever pass 1
       left unaccounted for -- reads full abstracts TOGETHER and can find a
       genuine shared TECHNICAL APPROACH across documents that share no
       literal recurring word at all (verified live: 4 real acoustic-
       sensing patents across completely unrelated applications -- pest
       control, noise pollution, pipeline leaks, general surveillance --
       correctly grouped despite sharing no common keyword). Deliberately
       run SECOND, on the remainder only, so it isn't redoing pass 1's job
       or producing a redundant near-duplicate of a facet pass 1 already
       found.

    Saved as child clusters (parent_id = "other"), same as a normal drill,
    so the loop naturally continues into them. Returns (pass1_children,
    pass2_children) -- kept SEPARATE rather than one flat list because the
    two passes found their branches by genuinely different mechanisms (a
    literal recurring word vs. a shared theme read across abstracts that
    share no common word), and a caller reporting this back to a person
    needs to say which is which rather than captioning both the same way.
    Either or both may be empty (a truly generic leftover stays a leaf, not
    force-split).
    """
    docs = store.docs_for_cluster(conversation_id, other_cluster["cluster_id"])
    if not docs:
        return [], []
    docs_by_id = {d["application_number"]: d for d in docs}
    existing_top_level = {c["cluster_id"] for c in store.get_clusters(conversation_id, parent_id=None)}

    # Pass 1: literal recurring-word recovery.
    recovered = group_documents(docs, min_cluster_size=_DRILL_MIN_SUPPORT)
    real = {k: c for k, c in recovered.items() if k != "other"}

    pass1_children = []
    recovered_appnos: set[str] = set()
    for key, c in real.items():
        recovered_appnos.update(c.doc_ids)
        if key in existing_top_level:
            # This facet already exists as its own top-level branch (it
            # cleared the floor there) -- these are just MORE real
            # documents for that SAME branch, not a new one. Merging avoids
            # a confusing duplicate-labeled child: observed live,
            # "deep raw audio" showing up twice at once -- a real top-level
            # branch and, separately, a same-named child of "other" -- with
            # nothing to tell them apart on screen.
            store.link_docs_to_cluster(conversation_id, key, c.doc_ids, source="other-redrill")
            continue
        child_id = f"{other_cluster['cluster_id']}__{key}"
        label_cluster(c, docs_by_id, llm, facet_vocab=APPROACH_FACETS)
        store.add_cluster(conversation_id, child_id, c.label, c.key_terms,
                          parent_id=other_cluster["cluster_id"])
        store.link_docs_to_cluster(conversation_id, child_id, c.doc_ids, source="other-redrill")
        pass1_children.append(store.get_cluster(conversation_id, child_id))

    # Pass 2: semantic grouping over whatever pass 1 left unaccounted for.
    pass2_children = []
    remainder = [d for d in docs if d["application_number"] not in recovered_appnos]
    if len(remainder) >= 2:
        for label, appnos in discover_semantic_groups(remainder, llm).items():
            key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "semantic_group"
            child_id = f"{other_cluster['cluster_id']}__{key}"
            key_terms = [w.upper() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", label)
                        if len(w) >= 4]
            store.add_cluster(conversation_id, child_id, label, key_terms,
                              parent_id=other_cluster["cluster_id"])
            store.link_docs_to_cluster(conversation_id, child_id, appnos, source="other-semantic")
            pass2_children.append(store.get_cluster(conversation_id, child_id))

    if not pass1_children and not pass2_children:
        return [], []

    store.set_cluster_status(conversation_id, other_cluster["cluster_id"], "drilled")
    return pass1_children, pass2_children


def _is_other_bucket(cluster_id: str) -> bool:
    """True for the top-level "other" bucket OR any NESTED leftover bucket
    created by a drill (e.g. "deep_raw_audio__other") -- these are never a
    coherent technique to confirm or search the real portal for, no matter
    how deep in the tree they are. Checking the cluster_id's own
    construction (parent__key) catches every depth, not just the top level.

    Caught live: only the literal top-level "other" was special-cased
    before. A NESTED other bucket ("deep_raw_audio__other") got normally
    confirmed like a real branch and queued a real targeted search using
    its fallback label -- literally the word "OTHER" -- as search
    vocabulary, which the real portal repeatedly choked on with 500 errors.
    """
    return cluster_id.rsplit("__", 1)[-1] == "other"


def drill_and_report_other(store, conversation_id: str, cluster: dict,
                           llm: LLMClient | None) -> list[dict]:
    """Drill an "other"-style leftover bucket (top-level or nested) and
    print what real signal it recovered, if any -- the "other" counterpart
    to drill_and_report/handle_confirmed_cluster, which must NEVER be
    called on one of these (see _is_other_bucket). Reports the two passes
    separately: a pass-1 branch is a real facet that just missed the
    original size floor; a pass-2 branch was found by reading the leftover
    abstracts together for a shared technique with no recurring keyword at
    all -- conflating the two captions would misdescribe how a given
    branch was actually found."""
    pass1_children, pass2_children = drill_other_bucket(store, conversation_id, cluster, llm)
    if pass1_children:
        print(f"  '{cluster['label']}' had real signal in it -- recovered "
             f"{len(pass1_children)} branch(es) that missed the original size floor:")
        for ch in pass1_children:
            print(f"    - {ch['label']}")
    if pass2_children:
        print(f"  '{cluster['label']}' had a real shared technique in it -- found "
             f"{len(pass2_children)} branch(es) by reading the leftover abstracts together "
             f"(no recurring keyword needed):")
        for ch in pass2_children:
            print(f"    - {ch['label']}")
    return pass1_children + pass2_children


def route_cluster_followup(store, conversation_id: str, cluster: dict,
                           llm: LLMClient | None) -> None:
    """Sends a cluster to the right follow-up once it has enough signal.
    An "other"-style leftover bucket (any nesting depth) is NEVER treated
    as a confirmable branch -- there's no real technique named "other" to
    drill-and-search for -- it gets looked inside instead, whether it's the
    top-level bucket or a nested one. This check runs BEFORE the normal
    "confirmed" check specifically so a nested other bucket that already
    crossed the confirm threshold (set by the generic scoring in
    process_hint, which has no notion of "other"-ness) gets routed here
    instead of into handle_confirmed_cluster/queue_targeted_search.
    """
    if _is_other_bucket(cluster["cluster_id"]):
        if cluster["status"] != "drilled" and cluster["score"] > 0:
            drill_and_report_other(store, conversation_id, cluster, llm)
    elif cluster["status"] == "confirmed":
        handle_confirmed_cluster(store, conversation_id, cluster, llm)


def show_state(store, conversation_id: str) -> None:
    clusters = store.get_clusters(conversation_id)
    print("\n  current branches:")
    for c in sorted(clusters, key=lambda x: -x["score"]):
        bar = "#" * int(max(0, c["score"] + 5))
        indent = "  -> " if c["parent_id"] else ""
        print(f"    [{c['status']:9}] {c['score']:+5.1f} {bar:10} {indent}{c['label']}")

    total = len(store.documents_with_cluster_info(conversation_id))
    if total:
        from .shortlist import build_shortlist
        in_play = len(build_shortlist(store, conversation_id, limit=total))
        eliminated = total - in_play
        print(f"\n  documents still in play: {in_play} / {total}"
             f"{f' ({eliminated} eliminated by ruled-out branches)' if eliminated else ''}")
    print()


def drill_and_report(store, conversation_id: str, cluster: dict,
                     llm: LLMClient | None) -> list[dict]:
    """Drill a confirmed branch and print what it split into, if anything."""
    children = drill_into_cluster(store, conversation_id, cluster, llm)
    if children:
        print(f"  '{cluster['label']}' is really a few different real combinations --"
             f" drilling in:")
        for ch in children:
            print(f"    - {ch['label']}")
    return children


def handle_confirmed_cluster(store, conversation_id: str, cluster: dict,
                             llm: LLMClient | None) -> None:
    """Everything that happens once a branch crosses the confirm threshold:
    drill it into finer real combinations (no network needed, happens
    immediately), and queue a real, narrower InPASS search targeted at it
    (needs internet -- saved as pending, picked up next time this
    conversation runs on that network, per Step 4 of the approved plan)."""
    from .bootstrap import queue_targeted_search

    drill_and_report(store, conversation_id, cluster, llm)
    if queue_targeted_search(store, conversation_id, cluster):
        print(f"  queued a real targeted search for '{cluster['label']}' -- "
             f"run this conversation again on internet to fetch it.")


# Sensor-MODALITY branches only -- the ones the seed's own vocabulary can
# safely settle up front, without touching the actual proprietary technique.
# PROCESSING-METHOD branches (deep_raw_audio, sequence_model,
# fixed_transform_frontend, handcrafted_features) are deliberately excluded:
# the seed blurb genuinely doesn't say which of those is used, so those stay
# open to real discovery through the conversation, same as before.
_SEED_GROUNDABLE_FACETS = ("vision_based", "thermal_based", "radar_based",
                          "rf_based", "multi_sensor_fusion")

# Words that establish acoustic/audio as the sensing modality -- the SAME
# vocabulary already trusted as the real portal search's own domain anchor
# (bootstrap.py's _ANCHOR_ROWS) and excluded from facet discovery as
# domain-baseline (clusters.py's _DOMAIN_STOPWORDS). A seed can establish
# "acoustic" WITHOUT ever using any of the other 5 branches' own words, so
# checking only those 5 branches' vocabulary (as a first version of this
# function did) misses the actual signal entirely -- acoustic isn't one of
# the "alternative" facets, it's the implicit baseline domain.
_ACOUSTIC_ANCHOR_WORDS = {"ACOUSTIC", "ACOUSTICS", "AUDIO", "SOUND", "SOUNDS",
                          "MICROPHONE", "MICROPHONES"}


def apply_seed_grounding(store, conversation_id: str, seed_text: str,
                         llm: LLMClient | None) -> list[str]:
    """If the seed's own words already settle the sensor modality (e.g.
    "audio dataset... highly sensitive microphones"), use that -- there is
    no confidentiality reason to make the conversation re-ask something you
    already told the tool in the seed. The seed is already public-safe (it's
    the exact text used to build the real portal search), so this only ever
    surfaces information you already gave, never a guess beyond it.

    Uses the SAME deterministic whole-word matching as real document
    clustering -- not an LLM judgment call -- so this is exactly as
    grounded as everything else in this system, and never touches the
    processing-method branches the seed doesn't actually disambiguate.

    Only acts if the seed makes a clear enough statement: a branch is
    confirmed if the seed directly names it (e.g. "camera"), and pruned if
    the seed establishes acoustic sensing and does NOT also name it -- each
    of the 5 branches is judged independently, so a seed mentioning BOTH
    audio and camera confirms vision_based while still correctly pruning
    thermal_based/radar_based/etc, since nothing in the seed suggests those
    either. A seed silent about sensing modality altogether must not be
    read as "therefore none of these apply" -- that would be a guess, not a
    fact from what you actually wrote. Returns the cluster_ids it touched.
    """
    hits = set(match_facets(seed_text)) & set(_SEED_GROUNDABLE_FACETS)
    acoustic_established = bool(_word_set(seed_text) & _ACOUSTIC_ANCHOR_WORDS)
    if not hits and not acoustic_established:
        return []

    existing = {c["cluster_id"] for c in store.get_clusters(conversation_id, parent_id=None)}
    touched = []
    for key in _SEED_GROUNDABLE_FACETS:
        if key not in existing:
            continue
        cluster = store.get_cluster(conversation_id, key)
        if key in hits:
            store.update_cluster_score(conversation_id, key, SCORE_CAP, cap=SCORE_CAP)
            store.set_cluster_status(conversation_id, key, "confirmed")
            handle_confirmed_cluster(store, conversation_id, cluster, llm)
        elif acoustic_established:
            store.update_cluster_score(conversation_id, key, -SCORE_CAP, cap=SCORE_CAP)
            store.set_cluster_status(conversation_id, key, "pruned")
        else:
            continue
        touched.append(key)
    return touched


# --------------------------------------------------- two-way: person asks back


def _recently_declined_cluster_ids(store, conversation_id: str, within_last_n_turns: int = 3) -> set[str]:
    """cluster_ids named in a "decline" turn within the last N turns -- used
    to steer pick_open_candidates away from immediately re-asking about
    something the person JUST said they didn't want to discuss. Reuses the
    existing turns log (kind="decline", cluster_refs already recorded there)
    rather than adding a new schema column for this."""
    turns = store.get_turns(conversation_id)
    recent = turns[-within_last_n_turns:] if within_last_n_turns else turns
    declined: set[str] = set()
    for t in recent:
        if t["kind"] == "decline":
            declined.update(t.get("cluster_refs") or [])
    return declined


def _recently_asked_cluster_ids(store, conversation_id: str,
                                within_last_n_questions: int = 3) -> set[str]:
    """cluster_ids asked about in the last N "question" turns (not the last
    N raw turns -- a side user_question or a decline shouldn't shrink this
    window) -- stops pick_open_candidates from re-selecting a set that's
    already had its recent turn.

    REAL FAILURE, observed live, in two stages:
    1. A one-turn-only version of this (remembering just the SINGLE most
       recent question) was tried first. It immediately forced progress off
       set A onto a different set B next turn -- but by the turn AFTER
       THAT, A was no longer "the last one asked" (B was), so A won the
       pick again. Since A was also the highest-scoring set, this produced
       an exact A -> B -> A -> B oscillation forever, confirmed directly
       from the real stored turns log (turn 5 asked set A, turn 6 asked set
       B, turn 11 was back to set A, verbatim). The person correctly
       perceived this as "it keeps asking me the same question" even
       though, turn to turn, SOME exclusion was technically happening.
    2. Several OTHER live branches never scored high enough to ever
       outrank A or B at all (every hint about them read as "doesn't
       address this" -- sentiment 3, delta 0.0 -- since they're
       semantically unrelated topic-labels, not real competitors for what
       the person was actually describing). A short memory window never
       gives those genuinely-untouched branches a real turn before the
       high scorers cycle back.

    A multi-question window (several DISTINCT recent sets excluded at
    once, not just the latest) is what actually breaks the cycle: with N
    live branches and a window covering the last few questions, the
    conversation is forced to work through the other candidates for real
    before anything already asked about recently can return. Falls back to
    re-asking anyway if nothing else is live (same never-stall guarantee
    pick_open_candidates already gives exclude_ids).
    """
    turns = store.get_turns(conversation_id)
    asked: set[str] = set()
    seen_questions = 0
    for t in reversed(turns):
        if t["kind"] == "question":
            asked.update(t.get("cluster_refs") or [])
            seen_questions += 1
            if seen_questions >= within_last_n_questions:
                break
    return asked


_QUESTION_STOPWORDS = {"WHAT", "WHICH", "WHERE", "WHEN", "DOES", "HAVE", "THAT",
                       "THIS", "THERE", "THESE", "THOSE", "WITH", "ABOUT", "SOME",
                       "ANY", "PAPER", "PAPERS", "DONE", "SOMETHING", "LIKE"}


def answer_user_question(store, conversation_id: str, question_text: str,
                         top_k: int = 5) -> list[dict]:
    """The person's OWN question back to the system ("is there any paper
    that's done X?") -- requested directly: "i should be able to ask u
    questions too... like lowkey two way communication".

    Deterministic, LLM-free: whole-word overlap between the question and
    every REAL, already-downloaded document this conversation has ever
    touched. These documents are already public and already fetched, so
    surfacing them carries none of the confidentiality risk answering about
    the person's OWN invention would -- and never inventing an answer (no
    LLM synthesis) means it can never confidently describe a document that
    doesn't actually say what's claimed. Returns [] (never a guess) if
    nothing shares even one real word with the question.
    """
    q_words = _word_set(question_text) - _BOILERPLATE_STOPWORDS - _DOMAIN_STOPWORDS - _QUESTION_STOPWORDS
    if not q_words:
        return []
    scored = []
    for d in store.all_conversation_docs(conversation_id):
        doc_words = _word_set(f"{d.get('title', '')} {d.get('abstract', '')}")
        overlap = len(q_words & doc_words)
        if overlap:
            scored.append((overlap, d))
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:top_k]]


def run_dialogue_loop(store, conversation_id: str, llm: LLMClient,
                      max_turns: int = 20) -> None:
    """The interactive question loop -- shared by the standalone step-2 CLI
    and the single unified chatbot entry point (scripts/run_chat.py), so
    both run the exact same conversation mechanics.
    """
    live_llm = llm if llm.enabled else None

    # Catch-up: a branch confirmed in an earlier run (or before drill-down
    # existed) may not have been split yet -- without this, resuming a
    # conversation where the top level already finished looks like it
    # "won't let you resume": there was nowhere left to go. Every confirmed
    # cluster at ANY depth is checked, not just top-level ones -- a nested
    # "other" bucket needs the same catch-up treatment a top-level one does.
    for c in store.get_clusters(conversation_id, status="confirmed"):
        route_cluster_followup(store, conversation_id, c, live_llm)

    for c in store.get_clusters(conversation_id, status="live"):
        if _is_other_bucket(c["cluster_id"]) and c["score"] > 0:
            route_cluster_followup(store, conversation_id, c, live_llm)

    show_state(store, conversation_id)

    convo = store.get_conversation(conversation_id)
    # `turn` is the conversation's cumulative turn number, used for turn_no
    # numbering in the turns table and never reset -- it must keep growing
    # across every resume so stored turns stay uniquely ordered. `max_turns`
    # is a SAFETY VALVE against one session running forever, not a lifetime
    # cap on the conversation -- confirmed live: resuming a long-lived
    # conversation whose cumulative turn_count had already reached 20 hit
    # "safety cap" immediately, with zero questions asked, because the old
    # code compared the cumulative count directly against max_turns. Tracked
    # separately here so a long conversation can always be resumed.
    turn = convo["turn_count"]
    turns_this_session = 0
    stopped = False
    while turns_this_session < max_turns:
        is_first_question = (turn == 0)
        live = store.get_clusters(conversation_id, status="live")
        doc_counts = store.cluster_doc_counts(conversation_id)
        asked_recently = _recently_asked_cluster_ids(store, conversation_id)
        exclude_ids = _recently_declined_cluster_ids(store, conversation_id) | asked_recently
        candidates = pick_open_candidates(live, doc_counts, exclude_ids=exclude_ids)
        if not candidates:
            confirmed = store.get_clusters(conversation_id, status="confirmed")
            print("Converged -- nothing left still live to ask about.")
            if confirmed:
                print("Leading branch(es):", [c["label"] for c in confirmed])
            break

        # REAL FAILURE, observed live: once every OTHER live branch is
        # either pruned or also in the recently-asked window, this ONE
        # remaining candidate is all pick_open_candidates has left to give
        # (its own "never stall" fallback re-includes recently-asked
        # candidates rather than ever returning nothing) -- so a branch
        # that keeps getting a genuinely uninformative answer ("maybe",
        # "idk") gets asked about again, verbatim, every single turn,
        # forever. Confirmed directly from the real stored turns log: once
        # only "other__uav_ambient_threat_detection_for_ppe" was left, it
        # was asked 3 turns running with identical phrasing. Stopping here
        # instead of repeating is strictly better than hammering the same
        # question: these branches stay LIVE (not pruned), so their
        # documents still appear in the shortlist, and the conversation can
        # always be resumed later -- e.g. once a targeted search brings
        # back fresh documents there's something new to actually ask about.
        if asked_recently and {c["cluster_id"] for c in candidates} <= asked_recently:
            print("Nothing new left to ask about right now -- every remaining branch was "
                 "already covered in the last few turns without a clear answer. Stopping "
                 "here for now rather than repeating the same question; resume any time, "
                 "or dig in again once a targeted search brings back fresh documents.")
            break

        cluster_ids = [c["cluster_id"] for c in candidates]
        question, method = generate_open_question(store, conversation_id, candidates, live_llm,
                                                   is_first_question=is_first_question)
        turn += 1
        question_turn = turn          # anchors this question <-> its eventual answer,
        turns_this_session += 1       # even if side-questions bump `turn` further in between
        store.bump_turn_count(conversation_id)
        store.add_turn(conversation_id, question_turn, "question", question, cluster_refs=cluster_ids)
        print(f"\nbot [{method}]: {question}")

        # Two-way: the person may ask US something back ("is there any paper
        # that's done X?") instead of answering yet -- requested directly:
        # "i should be able to ask u questions too... lowkey two way
        # communication". Detected structurally (ends with "?"), answered
        # from the real, already-downloaded, already-public documents this
        # conversation has touched (never an LLM guess), then the SAME
        # question is re-shown since it still hasn't actually been answered.
        while True:
            hint = input("you: ").strip()
            if hint.lower() in ("stop", "quit", "exit"):
                print("Stopped by you.")
                stopped = True
                break
            if hint.endswith("?"):
                # A fresh turn_no of its own -- add_turn's primary key is
                # (conversation_id, turn_no), so reusing question_turn here
                # would silently OVERWRITE that question's row instead of
                # recording a separate one.
                turn += 1
                store.bump_turn_count(conversation_id)
                store.add_turn(conversation_id, turn, "user_question", hint, cluster_refs=cluster_ids)
                matches = answer_user_question(store, conversation_id, hint)
                if matches:
                    print("  Here's what's already been found that might relate to that:")
                    for d in matches:
                        print(f"    {d['application_number']}: {(d.get('title') or '')[:70]}")
                        print(f"        {(d.get('abstract') or '')[:160]}")
                else:
                    print("  Nothing in what's been found so far shares real vocabulary with "
                         "that -- might need a fresh search once we know more.")
                print(f"\nbot [{method}]: {question}")
                continue
            break
        if stopped:
            break

        if _is_decline(hint):
            store.add_turn(conversation_id, question_turn, "decline", hint, cluster_refs=cluster_ids)
            print("  Okay, skipping that -- moving on.")
            show_state(store, conversation_id)
            continue

        store.add_turn(conversation_id, question_turn, "hint", hint, cluster_refs=cluster_ids)
        results = process_hint(store, conversation_id, question_turn, hint, cluster_ids, live_llm)
        for r in results:
            print(f"    -> {r.cluster_id}: sentiment {r.sentiment_digit} "
                 f"(delta {r.delta:+.1f}, {r.method})")

        for cid in cluster_ids:
            c = store.get_cluster(conversation_id, cid)
            if not c:
                continue
            route_cluster_followup(store, conversation_id, c, live_llm)

        show_state(store, conversation_id)
    else:
        print(f"Hit the {max_turns}-turn safety cap for this session -- "
             f"run again with the same conversation id to keep going.")

    show_shortlist(store, conversation_id)
    print(f"\nSaved. Resume any time with conversation id: {conversation_id}")


def show_shortlist(store, conversation_id: str, limit: int = 50) -> None:
    """Prints the current ranked shortlist -- documents that only ever
    matched a pruned branch are already excluded, which is what actually
    delivers "eliminating abstracts as we go" rather than just narrowing
    which branches get asked about."""
    from .shortlist import build_shortlist

    entries = build_shortlist(store, conversation_id, limit=limit)
    print(f"\n  shortlist ({len(entries)} document(s), highest first):")
    for e in entries:
        tag = f"[{e.verdict}] " if e.verdict else ""
        print(f"    {tag}{e.rank_score:+5.1f}  {e.application_number}  {e.title[:70]}")


# ------------------------------------------------------------------- a turn


def process_hint(store, conversation_id: str, turn_no: int, hint_text: str,
                 cluster_ids: list[str], llm: LLMClient | None) -> list[HintResult]:
    """Interpret one hint against each named cluster, update scores and
    status, and return what happened for display.

    Each cluster's call is told about the OTHER cluster(s) in this same
    contrast and its own ordinal position -- see interpret_hint's docstring
    for why that context is necessary, not optional.
    """
    clusters = [store.get_cluster(conversation_id, cid) for cid in cluster_ids]
    positions = ["first", "second"]
    results = []
    for i, cid in enumerate(cluster_ids):
        cluster = clusters[i]
        if not cluster:
            continue
        others = [c["label"] for j, c in enumerate(clusters) if j != i and c]
        other_label = ", ".join(others) if others else None
        position = positions[i] if i < len(positions) else None
        r = interpret_hint(hint_text, cluster, llm, other_label=other_label, position=position)
        new_score = store.update_cluster_score(conversation_id, cid, r.delta,
                                                cap=SCORE_CAP, turn_no=turn_no)
        if new_score >= CONFIRM_THRESHOLD and cluster["status"] == "live":
            store.set_cluster_status(conversation_id, cid, "confirmed")
        elif new_score <= PRUNE_THRESHOLD and cluster["status"] == "live":
            store.set_cluster_status(conversation_id, cid, "pruned")
        results.append(r)
    return results
