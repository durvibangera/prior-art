"""Step 6: the final ranked shortlist.

Builds the ~50 documents to hand to the legal team, from every document any
search in this conversation ever touched. Named directly in the approved
plan: "ranked by which branch found it and how confident that branch ended
up, with any document you directly reacted to weighted accordingly -- a
direct 'yes that's it' floats to the top; a direct 'no, not that' is dropped
outright."

A document that only ever matched a PRUNED branch, and was never touched by
anything still live or confirmed, is dropped -- that is the actual
"eliminating abstracts as we go" the conversation is supposed to deliver.
A document verdicted "rejected" (Step 4's "does this look like what you
did?") is dropped regardless of cluster score. A document verdicted
"confirmed" is pinned at the very top regardless of cluster score, since a
direct human confirmation is stronger evidence than any inferred score.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShortlistEntry:
    application_number: str
    title: str
    abstract: str
    rank_score: float
    verdict: str | None          # confirmed | partial | rejected | None
    found_by: list[str] = field(default_factory=list)


def build_shortlist(store, conversation_id: str, limit: int = 50) -> list[ShortlistEntry]:
    """Ranks and filters every document this conversation has touched.

    Exclusion: a document whose EVERY matching cluster is "pruned" (never
    touched anything still live or confirmed) is dropped, unless a direct
    verdict overrides it. A document verdicted "rejected" is dropped
    outright regardless of any cluster's score.

    Ranking: verdict "confirmed" sorts before everything else, then by the
    best score among the document's non-pruned clusters (highest first).
    """
    docs = store.documents_with_cluster_info(conversation_id)
    verdicts = store.get_verdicts(conversation_id)

    entries: list[ShortlistEntry] = []
    for d in docs:
        appno = d["application_number"]
        verdict = verdicts.get(appno)

        if verdict == "rejected":
            continue

        clusters = d["clusters"]
        live_or_confirmed = [c for c in clusters if c["status"] != "pruned"]
        if not live_or_confirmed and verdict != "confirmed":
            continue

        best_score = max((c["score"] for c in live_or_confirmed), default=0.0)
        found_by = sorted({c["cluster_id"] for c in clusters})

        entries.append(ShortlistEntry(
            application_number=appno,
            title=d.get("title") or "",
            abstract=d.get("abstract") or "",
            rank_score=best_score,
            verdict=verdict,
            found_by=found_by,
        ))

    entries.sort(key=lambda e: (0 if e.verdict == "confirmed" else 1, -e.rank_score))
    return entries[:limit]
