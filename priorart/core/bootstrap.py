"""Step 1 as a reusable function: seed -> one broad safe InPASS search ->
deterministic clustering -> LLM labeling.

Factored out of scripts/run_step1_bootstrap.py so the SAME code path can run
either standalone (for testing step 1 in isolation) or inline as the first
move of a single continuous chatbot session (scripts/run_examiner.py) -- a
conversation should not require you to separately run one script to harvest
and another to start talking, even though they were built and verified as
separate steps.
"""
from __future__ import annotations

import re
from pathlib import Path

from .clusters import (
    APPROACH_FACETS, _BOILERPLATE_STOPWORDS, _DOMAIN_STOPWORDS,
    expand_other_with_semantic_groups, expand_with_discovered_facets, group_documents,
    label_all_clusters,
)
from .llm import FieldSpec, LLMClient, RecordSpec, max_words
from .store import Store

# The three domain-anchor rows -- each is a mandatory AND-gate ("must
# actually be about acoustic sensing", "must actually be about drones",
# "must actually be about detection"). derive_search_rows can only ADD real
# seed vocabulary to these OR-groups, never remove or replace them.
#
# Learned live: an earlier version let the model invent whole NEW concept
# groups from the seed instead of expanding these three. It produced rows
# like "DETECTION LOW SNR NOISY CONDITIONS" / "AUDIO DATASET ML TRAINING" /
# "HARDWARE MICROPHONES SENSITIVE" -- none of which contained "DRONE" or
# "UAS" at all, even though the seed said "drone" explicitly. Since all rows
# are ANDed, the query lost its domain gate entirely and returned 870 hits
# instead of ~100, mostly unrelated. Expansion-only fixes this structurally:
# the anchor is never something the model can choose to drop.
_ANCHOR_ROWS: list[tuple[str, str, str]] = [
    ("ABS", "acoustic/audio sensing", "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE"),
    ("ABS", "drone/UAS", "DRONE DRONES UAV UAVS UAS QUADCOPTER"),
    ("ABS", "detection", "DETECTION DETECTING DETECT CLASSIFICATION"),
]

EXPAND_ROW_SYSTEM = """You read a short, already-public description of an
invention's general domain. You are given ONE search concept and its
starting list of OR'd synonym words already known to be safe and relevant.
Look at the text below and find any ADDITIONAL words that are real
synonyms, close word-forms, or directly relevant terms for THIS SAME
concept -- not a different concept, and not generic filler.

Every additional word must actually appear in the text below, or be a
common word-form of one that does (e.g. "detecting"/"detect" for
"detection"). If there is genuinely nothing more to add for this concept,
just repeat the starting words unchanged -- do not invent anything and do
not add a word just because it appears in the text if it doesn't belong to
THIS concept.

WORDS: <the starting words, plus any real additions, space separated, ALL CAPS>

Write plain text and nothing else. Write END on the last line."""


def _seed_candidate_words(seed: str) -> set[str]:
    excluded = _BOILERPLATE_STOPWORDS | _DOMAIN_STOPWORDS
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", seed.upper())
           if len(w) >= 4} - excluded


def _words_grounded_in_seed(seed_words: set[str], forbidden: set[str] | None = None):
    """EVERY WORDS entry must trace back to something actually in the seed
    text (and not be generic patent/domain boilerplate) -- exact match or a
    shared 5-letter stem, cheap enough to allow plurals/word-forms
    ("detect"/"detecting") without allowing the model to introduce a word
    that isn't really there, or that recurs across unrelated filings for
    reasons that have nothing to do with this invention (e.g. "COMPLETE
    SPECIFICATION" is standard Indian patent-filing boilerplate, not a real
    technical word -- caught live as a false grounding once, now excluded
    the same way clusters.py excludes it from facet discovery).

    REAL FAILURE, now guarded on two fronts:
    1. This used to accept the WHOLE list if just ONE word was grounded,
       letting an ungrounded word (e.g. "ML") ride along for free next to a
       genuine one. Now every word must individually ground.
    2. `forbidden` is another anchor row's OWN vocabulary (e.g. "DETECTION"
       belongs to the detection anchor). Adding it to a DIFFERENT anchor's
       row (e.g. acoustic) doesn't just broaden that row -- since rows are
       ANDed but each is satisfied by ANY one of its words, letting the same
       easy word satisfy multiple rows collapses their independence. Caught
       live: "DETECTION" ended up in all 3 anchor rows, so any document
       merely containing the word "detection" anywhere satisfied the ENTIRE
       AND structure -- 58,279 hits instead of the intended ~100.
    """
    forbidden = forbidden or set()

    def check(value: str) -> str | None:
        given = [w.strip().upper() for w in value.split() if w.strip()]
        if not given:
            return "must give at least 1 word"
        for w in given:
            if w in forbidden:
                return f"{w!r} belongs to a different anchor row, cannot be added here"
            if not any(w == sw or w[:5] == sw[:5] for sw in seed_words):
                return f"{w!r} does not trace back to a real, non-boilerplate word in the seed text"
        return None
    return check


def _expand_anchor_row(concept_label: str, starting_words: str, seed: str,
                       seed_words: set[str], other_anchor_words: set[str],
                       llm: LLMClient | None) -> str:
    """Adds real seed vocabulary to one fixed anchor row's OR-group. The
    starting words are ALWAYS kept (union, never replaced) -- this call can
    only broaden a row, never narrow or redirect it. `other_anchor_words`
    (the OTHER two anchors' own starting vocabulary) can never be added
    here, so one easy word can't end up satisfying every row at once."""
    starting_set = set(starting_words.split())
    if not llm or not llm.enabled:
        return starting_words

    user = (f"CONCEPT: {concept_label}\nSTARTING WORDS: {starting_words}\n"
           f"TEXT:\n{seed}\n\nWrite the expanded words now.")
    spec = RecordSpec(fields=[FieldSpec(
        "WORDS", validate=_words_grounded_in_seed(
            seed_words | starting_set, forbidden=other_anchor_words - starting_set))])
    result = llm.ask(EXPAND_ROW_SYSTEM, user, spec, max_tokens=100,
                     prompt_version="expand-row-v2")
    if not result.ok:
        return starting_words

    added = set(result.one["WORDS"].strip().upper().split())
    return " ".join(sorted(starting_set | added))


def derive_search_rows(seed: str, llm: LLMClient | None):
    """Builds the broad InPASS search rows by expanding the three safe
    domain-anchor rows with real vocabulary from the seed text -- never by
    inventing new concept groups from scratch, which risks losing a domain
    anchor entirely (see _ANCHOR_ROWS for what that looked like live).
    Falls back to the anchor rows completely unexpanded if the LLM is
    unavailable or the seed is too short to safely ground anything in.
    """
    from ..inpass.client import SearchRow
    return _derive_anchor_rows(seed, llm, lambda code, text: SearchRow(code, text))


def derive_search_rows_ppubs(seed: str, llm: LLMClient | None):
    """The SAME anchor-expansion logic as derive_search_rows, wrapped into
    PPUBS's own SearchRow/field-code shape (AB, not ABS) instead of
    InPASS's. The grounding discipline (expansion-only, never inventing a
    new concept, never letting one word satisfy more than one anchor) is
    identical -- only the target site's row type differs.
    """
    from ..ppubs.client import SearchRow
    return _derive_anchor_rows(seed, llm, lambda code, text: SearchRow("AB", text))


def _derive_anchor_rows(seed: str, llm: LLMClient | None, make_row):
    """Shared by derive_search_rows and derive_search_rows_ppubs: expands
    the three safe domain-anchor rows with real vocabulary from the seed
    text -- never by inventing new concept groups from scratch, which risks
    losing a domain anchor entirely (see _ANCHOR_ROWS for what that looked
    like live). Falls back to the anchor rows completely unexpanded if the
    LLM is unavailable or the seed is too short to safely ground anything
    in. `make_row(field_code, text)` builds the caller's own site-specific
    SearchRow type from the (InPASS) field code and expanded text.
    """
    fallback = [make_row(code, text) for code, _label, text in _ANCHOR_ROWS]
    seed_words = _seed_candidate_words(seed)
    if not llm or not llm.enabled or len(seed_words) < 3:
        return fallback

    rows = []
    for code, label, starting in _ANCHOR_ROWS:
        other_words = {w for _, _, other_starting in _ANCHOR_ROWS
                      if other_starting != starting
                      for w in other_starting.split()}
        expanded = _expand_anchor_row(label, starting, seed, seed_words, other_words, llm)
        rows.append(make_row(code, expanded))
    return rows


# ----------------------------------------------- experimental: fully dynamic


FULL_DERIVE_SYSTEM = """You read a short, already-public description of an
invention's general domain (never its exact proprietary technique). Identify
2 to 4 DISTINCT search concepts a patent search on this domain should cover,
each as a group of 3 to 6 synonym/related words a real search engine box
would OR together -- broad enough to catch differently-worded real filings.

Every word you write must actually appear in the text below, or be a close,
common word-form of one that does (e.g. "detecting"/"detect" for
"detection"). A word must belong to EXACTLY ONE concept -- never repeat the
same word across two different concepts, since these concepts get ANDed
together afterward, and a word shared between them would defeat that.

CONCEPT: <2 to 6 word label>
WORDS: <3 to 6 space separated ALL CAPS words>
(repeat CONCEPT/WORDS once per concept, 2 to 4 concepts total)

Write plain text lines and nothing else. Write END on the last line."""

FULL_DERIVE_SPEC = RecordSpec(
    record_key="CONCEPT", min_records=2, max_records=4,
    fields=[FieldSpec("CONCEPT", validate=max_words(6)), FieldSpec("WORDS")])


def _seed_words_including_domain(seed: str) -> set[str]:
    """The grounding pool for FULLY DYNAMIC derivation specifically --
    unlike _seed_candidate_words (used for anchor EXPANSION, where domain
    words like DRONE/AUDIO are deliberately excluded since they're already
    guaranteed present via the fixed anchor text), a from-scratch
    derivation has no such guarantee. Excluding domain words here would
    make it structurally impossible for the model to ever include them,
    reproducing the exact domain-anchor-loss failure this whole safety net
    exists to prevent -- boilerplate is still excluded, domain words are not.

    Length floor is 3, not 4 -- "UAS" (a real, meaningful domain acronym
    that appears in seed text such as "against UAS threats") was being
    silently filtered out at the 4-letter floor used elsewhere, which
    would make it structurally ungroundable here even though it is exactly
    the kind of domain anchor this function exists to allow.
    """
    return {w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", seed.upper())
           if len(w) >= 3} - _BOILERPLATE_STOPWORDS


def derive_search_rows_fully_dynamic(seed: str, llm: LLMClient | None,
                                     field_code: str = "ABS"):
    """Derives search rows entirely from the seed with no fixed anchor
    skeleton at all, unlike derive_search_rows. Used by
    prepare_search_from_disclosure, so a disclosure about something other
    than acoustic drone detection gets its own vocabulary.

    Returns (rows, reason) -- rows is None (never a guessed-at fallback
    list) if the LLM is unavailable, the response fails validation, ANY
    word is ungrounded, or the SAME word ends up satisfying more than one
    row (the exact failure that produced 58,279 hits the one time an
    ungoverned version of this was run live -- see _ANCHOR_ROWS's own
    history). `reason` says WHICH of those happened and with which
    word/concept, rather than just "rejected" -- a bare None told a person
    NOTHING about whether this is close to workable or badly wrong, which
    matters for deciding whether to keep iterating on it at all. rows is
    None means "not safe to use as-is"; the caller
    (scripts/compare_search_rows.py) shows this side by side with
    derive_search_rows so a person decides, rather than this ever running
    against the real portal on its own.
    """
    if not llm or not llm.enabled:
        return None, "LLM unavailable"
    from ..inpass.client import SearchRow

    seed_words = _seed_words_including_domain(seed)
    if len(seed_words) < 6:
        return None, "seed too short to safely ground anything in"

    user = f"TEXT:\n{seed}\n\nWrite the concepts now."
    result = llm.ask(FULL_DERIVE_SYSTEM, user, FULL_DERIVE_SPEC, max_tokens=300,
                     prompt_version="full-derive-v1")
    if not result.ok or len(result.records) < 2:
        reason = "; ".join(result.errors[:2]) or f"only {len(result.records)} concept(s) parsed"
        if "parsed 0" in reason and result.raw:
            reason += f" | raw: {result.raw[:200]!r}"
        return None, reason

    rows = []
    claimed: dict[str, str] = {}
    for rec in result.records:
        concept = rec.get("CONCEPT", "?")
        words = [w.strip().upper() for w in rec.get("WORDS", "").split() if w.strip()]
        if not words:
            return None, f"concept {concept!r} had no words at all"
        for w in words:
            if not any(w == sw or w[:5] == sw[:5] for sw in seed_words):
                return None, f"{w!r} (in concept {concept!r}) does not trace back to the seed text"
            if w in claimed and claimed[w] != concept:
                return None, f"{w!r} was claimed by both {claimed[w]!r} and {concept!r} -- cross-row overlap"
            claimed[w] = concept
        rows.append(SearchRow(field_code, " ".join(words)))
    return rows, None


_BOOTSTRAP_QID = "bootstrap"


def prepare_search(seed: str, conversation_id: str, llm: LLMClient, store: Store):
    """Derives the broad search rows from the seed and stashes them as a
    pending search -- the INTRANET half of starting a conversation (needs
    the LLM, never touches InPASS). Run this first, then run_harvest (or
    scripts/run_examiner.py, which calls it automatically) once you're back on
    internet -- the harvest step then needs no LLM call at all, so it never
    has to reach across the network split itself.

    Creates the conversation row if it doesn't already exist. Safe to call
    again for the same conversation_id before the harvest has run (e.g. to
    re-derive after editing the seed) -- overwrites the previous pending rows.
    """
    if not store.get_conversation(conversation_id):
        store.create_conversation(conversation_id, seed, site="inpass")

    rows = derive_search_rows(seed, llm)
    store.add_pending_search(conversation_id, _BOOTSTRAP_QID,
                             [(r.field_code, r.text) for r in rows], target_cluster_id=None)
    return rows


def prepare_search_ppubs(seed: str, conversation_id: str, llm: LLMClient, store: Store):
    """The PPUBS counterpart to prepare_search -- same INTRANET-only role
    (derives rows from the seed via the LLM, never touches the portal
    itself), but tags the conversation site='ppubs' so run_examiner.py and
    every later targeted search know which portal and client to keep using
    for the rest of this conversation without being told again.
    """
    if not store.get_conversation(conversation_id):
        store.create_conversation(conversation_id, seed, site="ppubs")

    rows = derive_search_rows_ppubs(seed, llm)
    store.add_pending_search(conversation_id, _BOOTSTRAP_QID,
                             [(r.field_code, r.text) for r in rows], target_cluster_id=None)
    return rows


def prepare_search_from_disclosure(domain_text: str, conversation_id: str, llm: LLMClient,
                                   store: Store, site: str = "inpass"):
    """prepare_search for an uploaded disclosure: the broad harvest's rows
    come from the disclosure's OWN domain vocabulary (title, nature,
    purpose -- never the inventive features), via the fully dynamic
    derivation, rather than the acoustic/drone anchor rows.

    Returns (rows, note). Falls back to the anchored derivation, saying why,
    if the dynamic one is rejected by its own grounding checks.
    """
    from ..inpass.client import SearchRow as InpassRow
    from ..ppubs.client import SearchRow as PpubsRow

    if not store.get_conversation(conversation_id):
        store.create_conversation(conversation_id, domain_text, site=site)

    rows, reason = derive_search_rows_fully_dynamic(
        domain_text, llm, field_code=_ANCHOR_FIELD_CODE_BY_SITE[site])
    note = None
    if rows is None:
        note = f"dynamic derivation rejected ({reason}) -- used the anchored acoustic/drone rows instead"
        rows = (derive_search_rows_ppubs if site == "ppubs" else derive_search_rows)(domain_text, llm)
    row_cls = PpubsRow if site == "ppubs" else InpassRow
    rows = [row_cls(r.field_code, r.text) for r in rows]
    store.add_pending_search(conversation_id, _BOOTSTRAP_QID,
                             [(r.field_code, r.text) for r in rows], target_cluster_id=None)
    return rows, note


def run_harvest(conversation_id: str, cfg: dict, llm: LLMClient, store: Store,
                root: Path) -> None:
    """Runs the real broad search (one captcha, typed by you), harvests and
    clusters the results, and saves everything under `conversation_id` -- the
    INTERNET half. Uses rows already prepared by prepare_search if present
    (no LLM call needed here at all); otherwise derives them inline as a
    one-shot fallback, which gracefully degrades to the generic template if
    the LLM happens to be unreachable from wherever this is actually run.
    """
    from ..inpass.captcha import ManualSolver
    from ..inpass.client import InpassClient, SearchRow

    convo = store.get_conversation(conversation_id)
    if not convo:
        raise ValueError(f"no conversation '{conversation_id}' -- call prepare_search first")

    pending = store.get_pending_searches(conversation_id, state="pending")
    prepared = next((p for p in pending if p["qid"] == _BOOTSTRAP_QID), None)
    if prepared:
        rows = [SearchRow(code, text) for code, text in prepared["rows"]]
    else:
        rows = derive_search_rows(convo["seed_text"], llm)

    print("Grounding the first guesses in real filings -- running one broad search:")
    for r in rows:
        print(f"  {r.field_code}: {r.text}")
    print()

    client = InpassClient(cfg, ManualSolver(root / "priorart/out"))
    png = client._fetch_captcha()
    cap_path = root / "priorart/out/step1_captcha.png"
    cap_path.write_bytes(png)
    print(f"Captcha saved to {cap_path} -- open it and type the 6 characters.")
    answer = input("captcha> ").strip()

    result = client.search(rows, date_from=cfg["run"]["date_from"],
                           date_to=cfg["run"]["date_to"],
                           date_field=cfg["run"]["date_field"],
                           status=cfg["run"]["status"], captcha_answer=answer)
    print(f"\nTotal hits: {result.total} across {result.total_pages} page(s)")

    hits = list(client.iter_all_hits(result))
    store.upsert_documents(hits, site="inpass")
    print(f"Harvested {len(hits)} documents (paging needed no further captcha)")

    print("Fetching abstracts (no captcha needed)...")
    suspect_count = 0
    for i, h in enumerate(hits, 1):
        try:
            detail = client.fetch_detail(h.application_number)
            if detail.get("abstract_suspect"):
                suspect_count += 1
            store.save_detail(detail)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {h.application_number}: FAILED ({exc})")
        if i % 20 == 0:
            print(f"  ...{i}/{len(hits)}")
    if suspect_count:
        print(f"  NOTE: {suspect_count} document(s) had an abstract extraction that "
             f"looked wrong (too long / unmatched page structure) -- stored as-is, "
             f"flag if clustering looks off for those documents.")

    _cluster_and_save_harvest(conversation_id, hits, llm, store, convo["seed_text"])
    if prepared:
        store.mark_search_done(conversation_id, _BOOTSTRAP_QID)


def _cluster_and_save_harvest(conversation_id: str, hits: list, llm: LLMClient,
                              store: Store, seed_text: str) -> None:
    """The site-agnostic tail shared by run_harvest and run_harvest_ppubs:
    once real documents with real abstracts are already saved to the store,
    everything from here on (clustering, facet discovery, semantic
    grouping, labeling) works the same regardless of which
    portal found them.
    """
    docs = [store.get_document(h.application_number) for h in hits]
    docs = [d for d in docs if d and d.get("abstract")]
    print(f"\n{len(docs)} documents with usable abstract text\n")

    clusters = group_documents(docs)
    docs_by_id = {d["application_number"]: d for d in docs}

    # If enough real documents didn't fit any hand-written facet, look for
    # what actually recurs among them and split off new branches for it --
    # grounded in real vocabulary from those specific leftover documents,
    # not the model inventing a category. This is what lets the branch list
    # grow over time instead of being a fixed set capped at whatever facets
    # were hand-written in advance.
    clusters, discovered = expand_with_discovered_facets(
        clusters, docs_by_id, APPROACH_FACETS, llm=llm if llm.enabled else None)
    facet_vocab = {**APPROACH_FACETS, **discovered}
    label_all_clusters(clusters, docs_by_id, llm if llm.enabled else None, facet_vocab=facet_vocab)

    if discovered:
        print("DISCOVERED NEW BRANCHES (recurring real vocabulary, not in the built-in list):")
        for key, terms in discovered.items():
            print(f"  {key}: seed term {terms}")
        print()

    # Second pass over whatever's STILL left in "other" -- reads those real
    # leftover abstracts together looking for a genuine shared technique,
    # not just a recurring word. Runs here (before the conversation even
    # starts) as well as later via dialogue.py's drill_other_bucket, so the
    # very first branch list already benefits from it.
    pre_semantic_keys = set(clusters)
    clusters = expand_other_with_semantic_groups(clusters, docs_by_id, llm if llm.enabled else None)
    semantic_keys = set(clusters) - pre_semantic_keys
    if semantic_keys:
        print("DISCOVERED NEW BRANCHES (shared technique found by reading abstracts together):")
        for key in sorted(semantic_keys):
            c = clusters[key]
            print(f"  {key:26} {len(c.doc_ids):3} documents -- {c.label}")
        print()

    print("BRANCHES FOUND")
    print("-" * 68)
    for key, c in sorted(clusters.items(), key=lambda kv: -len(kv[1].doc_ids)):
        print(f"  {key:26} {len(c.doc_ids):3} documents -- {c.label}")
        store.add_cluster(conversation_id, key, c.label, c.key_terms)
        store.link_docs_to_cluster(conversation_id, key, c.doc_ids, source="bootstrap")
    print()


def run_harvest_ppubs(conversation_id: str, cfg: dict, llm: LLMClient, store: Store) -> None:
    """The PPUBS counterpart to run_harvest -- same role (runs the real
    broad search, harvests and clusters the results), but NO captcha at
    all, so this is considerably simpler: one authenticate call, one
    search call, no image to save and type in. Everything from
    _cluster_and_save_harvest on is identical to the InPASS path.
    """
    from ..ppubs.client import PpubsClient, SearchRow

    convo = store.get_conversation(conversation_id)
    if not convo:
        raise ValueError(f"no conversation '{conversation_id}' -- call prepare_search_ppubs first")

    pending = store.get_pending_searches(conversation_id, state="pending")
    prepared = next((p for p in pending if p["qid"] == _BOOTSTRAP_QID), None)
    if prepared:
        rows = [SearchRow(code, text) for code, text in prepared["rows"]]
    else:
        rows = derive_search_rows_ppubs(convo["seed_text"], llm)

    print("Grounding the first guesses in real filings -- running one broad search (no captcha):")
    for r in rows:
        print(f"  {r.field_code}: {r.text}")
    print()

    client = PpubsClient(cfg)
    result = client.search(rows, page_size=cfg["ppubs"].get("page_size", 50))
    print(f"Total hits: {result.total}")

    hits = list(client.iter_all_hits(rows, result))
    store.upsert_documents(hits, site="ppubs")
    print(f"Harvested {len(hits)} documents")

    print("Fetching abstracts...")
    suspect_count = 0
    for i, h in enumerate(hits, 1):
        try:
            detail = client.fetch_detail(h.application_number, h.status)
            if detail.get("abstract_suspect"):
                suspect_count += 1
            store.save_detail(detail)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {h.application_number}: FAILED ({exc})")
        if i % 20 == 0:
            print(f"  ...{i}/{len(hits)}")
    if suspect_count:
        print(f"  NOTE: {suspect_count} document(s) had an abstract extraction that "
             f"looked wrong (too long / unmatched page structure) -- stored as-is, "
             f"flag if clustering looks off for those documents.")

    _cluster_and_save_harvest(conversation_id, hits, llm, store, convo["seed_text"])
    if prepared:
        store.mark_search_done(conversation_id, _BOOTSTRAP_QID)


# ------------------------------------------------------ Step 4: targeted re-search


# The InPASS anchor field code ("ABS") vs. PPUBS's own ("AB") for the same
# abstract-scoped concept -- queue_targeted_search looks up which one to use
# from the conversation's own recorded site, so callers (the examiner) never
# have to know or care which portal a conversation is using.
_ANCHOR_FIELD_CODE_BY_SITE = {"inpass": "ABS", "ppubs": "AB"}


def queue_targeted_search(store: Store, conversation_id: str, cluster: dict) -> bool:
    """Once a branch is confident enough, build a real, narrower query from
    the safe domain anchors PLUS this branch's own key terms -- terms
    already established as real and grounded (either the facet's own seed
    words, or an LLM label validated to reuse them), never invented for this
    call, so the same confidentiality discipline as the broad step-1 search
    holds here too.

    Saves it as a pending search (qid = the cluster's own id) rather than
    running it inline -- this needs internet, and the conversation that
    triggers it is running on intranet (needs the LLM), so it has to wait
    for the same network switch as everything else. Returns False if this
    cluster already has a search queued or done, so it's never queued
    twice.

    Site-aware, not a separate ppubs counterpart: the field code embedded
    in the row (InPASS's "ABS" vs PPUBS's "AB") is looked up from the
    conversation's own recorded site, so dialogue.py's call site stays
    identical regardless of which portal this conversation searches.
    """
    # Structural backstop, not just caller discipline: an "other"-style
    # leftover bucket (top-level or nested, e.g. "deep_raw_audio__other")
    # is never a real technique to search for -- caught live, one got
    # normally confirmed and queued a real search using the literal word
    # "OTHER" as search vocabulary, which the real portal repeatedly choked
    # on with 500 errors. dialogue.py's route_cluster_followup already
    # should never call this for one, but this makes it impossible even if
    # a future caller doesn't know that (same cluster_id convention as
    # dialogue.py's _is_other_bucket).
    if cluster["cluster_id"].rsplit("__", 1)[-1] == "other":
        return False

    already = (store.get_pending_searches(conversation_id, state="pending") +
              store.get_pending_searches(conversation_id, state="done"))
    if any(p["qid"] == cluster["cluster_id"] for p in already):
        return False

    convo = store.get_conversation(conversation_id)
    site = (convo or {}).get("site") or "inpass"
    field_code = _ANCHOR_FIELD_CODE_BY_SITE[site]

    anchor_rows = [(field_code, text) for _code, _label, text in _ANCHOR_ROWS]
    branch_terms = " ".join(sorted({t.upper() for t in (cluster.get("key_terms") or []) if t}))
    rows = anchor_rows + ([(field_code, branch_terms)] if branch_terms else [])

    store.add_pending_search(conversation_id, cluster["cluster_id"], rows,
                             target_cluster_id=cluster["cluster_id"])
    return True


def _search_target(store: Store, conversation_id: str, prepared: dict) -> tuple[str, str]:
    """(cluster_id to link hits to, label to print). A search queued by the
    examiner has no cluster of its own; its hits go to a `search:<qid>`
    pseudo-cluster so all_conversation_docs (which joins on clusters) sees
    them."""
    qid = prepared["qid"]
    target = prepared.get("target_cluster_id")
    cluster = store.get_cluster(conversation_id, target) if target else None
    if cluster:
        return target, cluster["label"]
    cluster_id = f"search:{qid}"
    if not store.get_cluster(conversation_id, cluster_id):
        store.add_cluster(conversation_id, cluster_id, f"follow-up search {qid}", [])
    return cluster_id, f"follow-up search {qid}"


def run_targeted_search(conversation_id: str, qid: str, cfg: dict,
                        store: Store, root: Path) -> int:
    """Runs one queued search (one captcha, typed by you) and links any real
    documents found to its cluster. Returns how many were newly linked (not
    already known to that cluster).
    """
    from ..inpass.captcha import ManualSolver
    from ..inpass.client import InpassClient, SearchRow

    pending = store.get_pending_searches(conversation_id, state="pending")
    prepared = next((p for p in pending if p["qid"] == qid), None)
    if not prepared:
        return 0
    rows = [SearchRow(code, text) for code, text in prepared["rows"]]
    cluster_id, label = _search_target(store, conversation_id, prepared)
    print(f"Targeted real search for '{label}':")
    for r in rows:
        print(f"  {r.field_code}: {r.text}")
    print()

    client = InpassClient(cfg, ManualSolver(root / "priorart/out"))
    png = client._fetch_captcha()
    cap_path = root / "priorart/out/targeted_captcha.png"
    cap_path.write_bytes(png)
    print(f"Captcha saved to {cap_path} -- open it and type the 6 characters.")
    answer = input("captcha> ").strip()

    result = client.search(rows, date_from=cfg["run"]["date_from"],
                           date_to=cfg["run"]["date_to"],
                           date_field=cfg["run"]["date_field"],
                           status=cfg["run"]["status"], captcha_answer=answer)
    print(f"\nTotal hits: {result.total} across {result.total_pages} page(s)")

    hits = list(client.iter_all_hits(result))
    existing = {d["application_number"] for d in store.docs_for_cluster(conversation_id, cluster_id)}
    store.upsert_documents(hits, site="inpass")

    print("Fetching abstracts (no captcha needed)...")
    for i, h in enumerate(hits, 1):
        try:
            store.save_detail(client.fetch_detail(h.application_number))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {h.application_number}: FAILED ({exc})")
        if i % 20 == 0:
            print(f"  ...{i}/{len(hits)}")

    store.link_docs_to_cluster(conversation_id, cluster_id,
                               [h.application_number for h in hits], source="targeted")
    store.mark_search_done(conversation_id, qid)

    new_count = sum(1 for h in hits if h.application_number not in existing)
    print(f"\n{len(hits)} documents linked to '{label}' ({new_count} newly found).")
    return new_count


def run_targeted_search_ppubs(conversation_id: str, qid: str, cfg: dict,
                              store: Store) -> int:
    """The PPUBS counterpart to run_targeted_search -- no captcha, so no
    root path is even needed (nothing to save to disk for you to read).
    Otherwise identical in shape and return value.
    """
    from ..ppubs.client import PpubsClient, SearchRow

    pending = store.get_pending_searches(conversation_id, state="pending")
    prepared = next((p for p in pending if p["qid"] == qid), None)
    if not prepared:
        return 0
    rows = [SearchRow(code, text) for code, text in prepared["rows"]]
    cluster_id, label = _search_target(store, conversation_id, prepared)
    print(f"Targeted real search for '{label}' (no captcha):")
    for r in rows:
        print(f"  {r.field_code}: {r.text}")
    print()

    client = PpubsClient(cfg)
    result = client.search(rows, page_size=cfg["ppubs"].get("page_size", 50))
    print(f"Total hits: {result.total}")

    hits = list(client.iter_all_hits(rows, result))
    existing = {d["application_number"] for d in store.docs_for_cluster(conversation_id, cluster_id)}
    store.upsert_documents(hits, site="ppubs")

    print("Fetching abstracts...")
    for i, h in enumerate(hits, 1):
        try:
            store.save_detail(client.fetch_detail(h.application_number, h.status))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {h.application_number}: FAILED ({exc})")
        if i % 20 == 0:
            print(f"  ...{i}/{len(hits)}")

    store.link_docs_to_cluster(conversation_id, cluster_id,
                               [h.application_number for h in hits], source="targeted")
    store.mark_search_done(conversation_id, qid)

    new_count = sum(1 for h in hits if h.application_number not in existing)
    print(f"\n{len(hits)} documents linked to '{label}' ({new_count} newly found).")
    return new_count


def bootstrap_conversation(seed: str, conversation_id: str, cfg: dict,
                           llm: LLMClient, store: Store, root: Path) -> None:
    """One-shot convenience: prepare_search + run_harvest back to back, in a
    single process. Only really works if the LLM and InPASS happen to both
    be reachable from wherever this runs (e.g. a dual-homed machine) -- on a
    machine with the real network split, run prepare_search on intranet and
    run_harvest (or scripts/run_examiner.py) separately on internet instead. If
    the LLM isn't reachable when this runs, prepare_search's derivation
    gracefully falls back to the generic template rather than blocking.
    """
    prepare_search(seed, conversation_id, llm, store)
    run_harvest(conversation_id, cfg, llm, store, root)
