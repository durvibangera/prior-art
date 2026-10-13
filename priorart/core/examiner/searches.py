"""Follow-up portal searches the examiner suggests, behind a safety gate.

The confidentiality rule is that the inventive COMBINATION never reaches a
public portal. Concretely: a query may use the domain vocabulary (words of
the claims' context elements, plus generic terms) and the vocabulary of at
most ONE distinguishing element -- every inventive word in the query must
come from a single element. Anything that needs two elements to explain its
words is blocked in code, whatever the model suggested. You then still see
the exact rows and approve each one.

(An earlier version only counted words UNIQUE to one element. Dependent
claims restate their parent -- "LSTM layers after the convolutional stage"
-- so "convolutional" stopped being unique and WAVELET SCATTERING +
CONVOLUTIONAL, the core combination, passed. Caught in the end-to-end run.)
"""
from __future__ import annotations

import re

from ..clusters import _BOILERPLATE_STOPWORDS

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_FIELD_CODE = {"inpass": "ABS", "ppubs": "AB"}
MAX_ROWS = 4
MAX_WORDS_PER_ROW = 12
_GENERIC = _BOILERPLATE_STOPWORDS | {"AND", "THE", "FOR", "INTO", "ONTO", "VIA", "EACH", "BY"}


def _words(text: str) -> set[str]:
    return {w.upper() for w in _WORD.findall(text or "") if len(w) >= 3} - _GENERIC


def distinctive_vocab(claims: list[dict]) -> dict[str, set[str]]:
    """element_id -> that distinguishing element's inventive words (its
    words minus the domain/context vocabulary)."""
    context_words: set[str] = set()
    dist: dict[str, set[str]] = {}
    for c in claims:
        for e in c["elements"]:
            if e["role"] == "context":
                context_words |= _words(e["text"])
            else:
                dist[e["id"]] = _words(e["text"])
    return {eid: ws - context_words for eid, ws in dist.items()}


def check_query(rows: list[list[str]], claims: list[dict]) -> str | None:
    """None if safe, else the reason it is blocked."""
    if not rows:
        return "empty query"
    if len(rows) > MAX_ROWS:
        return f"{len(rows)} rows -- at most {MAX_ROWS}"
    query_words = {w.upper() for row in rows for w in row}
    vocab = distinctive_vocab(claims)
    inventive = query_words & set().union(*vocab.values()) if vocab else set()
    if not inventive or any(inventive <= ws for ws in vocab.values()):
        return None
    sources = {eid: sorted(inventive & ws) for eid, ws in vocab.items() if inventive & ws}
    detail = "; ".join(f"{eid}: {', '.join(ws)}" for eid, ws in sorted(sources.items()))
    return (f"its inventive terms ({', '.join(sorted(inventive))}) span more than one "
            f"distinguishing element ({detail})")


def to_rows(suggestion_rows: list[list[str]], site: str) -> list[tuple[str, str]]:
    """Words only (the portal has no phrase search -- PHASE0_FINDINGS.md §6),
    upper-cased, deduplicated, one OR-box per row."""
    code = _FIELD_CODE.get(site, "ABS")
    out = []
    for row in suggestion_rows:
        words = []
        for item in row:
            for w in _WORD.findall(item or ""):
                w = w.upper()
                if w not in words and w not in _GENERIC:
                    words.append(w)
        if words:
            out.append((code, " ".join(words[:MAX_WORDS_PER_ROW])))
    return out


def search_key(suggestion: dict) -> tuple:
    """Order-insensitive identity of a suggested search, for de-duplication."""
    return tuple(sorted(tuple(sorted(w.upper() for w in row)) for row in suggestion.get("rows") or []))


def reviewed_keys(store, conversation_id: str) -> set[tuple]:
    """Searches already decided on (queued, declined or blocked) -- so the
    examiner re-suggesting one does not bring it back."""
    return {tuple(tuple(r) for r in k) for k in store.kv_get(conversation_id, "reviewed_search_keys", [])}


def review_suggestions(store, conversation_id: str, claims: list[dict], site: str) -> int:
    """Walk the pending suggestions; queue the ones you approve. Returns how
    many were queued."""
    suggestions = store.kv_get(conversation_id, "suggested_searches", [])
    if not suggestions:
        print("  No search suggestions yet.")
        return 0
    done = (store.get_pending_searches(conversation_id, "pending")
            + store.get_pending_searches(conversation_id, "done"))
    n = len([p for p in done if p["qid"].startswith("ex")])
    queued, remaining = 0, []
    reviewed = reviewed_keys(store, conversation_id)
    for s in suggestions:
        rows = to_rows(s.get("rows") or [], site)
        print(f"\n  Purpose: {s.get('purpose', '')}")
        for code, text in rows:
            print(f"    {code}: {text}")
        problem = check_query([t.split() for _, t in rows], claims)
        if problem:
            print(f"    BLOCKED: {problem}")
            reviewed.add(search_key(s))
            continue
        choice = input("    queue this search? [y]es / [n]o / [l]ater > ").strip().lower()
        if not choice.startswith("l"):
            reviewed.add(search_key(s))
        if choice.startswith("y"):
            n += 1
            store.add_pending_search(conversation_id, f"ex{n}", rows, target_cluster_id=None)
            queued += 1
        elif choice.startswith("l"):
            remaining.append(s)
    store.kv_set(conversation_id, "suggested_searches", remaining)
    store.kv_set(conversation_id, "reviewed_search_keys", [list(map(list, k)) for k in sorted(reviewed)])
    if queued:
        print(f"\n  {queued} search(es) queued. Switch to INTERNET and run the examiner "
              f"again to fetch them; the new documents are charted on the next intranet run.")
    return queued
