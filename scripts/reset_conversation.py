#!/usr/bin/env python3
"""Start a clean conversation from an existing one's real harvested documents,
without hitting InPASS (and a captcha) again.

    python3 scripts/reset_conversation.py <source_conversation_id> [new_id]

Copies the SOURCE conversation's seed and its TOP-LEVEL clusters (real
documents already harvested and labeled by step 1) into a brand-new
conversation, all scores reset to 0 and every branch live. Useful after a
conversation's scores got contaminated by mechanical/test answers rather than
genuine hints -- rather than re-running step 1's live search, this reuses the
same real document set with a clean slate to actually talk to.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <source_conversation_id> [new_id]")
    source_id = sys.argv[1]
    new_id = sys.argv[2] if len(sys.argv) > 2 else uuid.uuid4().hex[:8]

    store = Store(ROOT / "priorart/out/conversations.db")
    source = store.get_conversation(source_id)
    if not source:
        sys.exit(f"no conversation '{source_id}' found")

    if store.get_conversation(new_id):
        sys.exit(f"conversation '{new_id}' already exists -- pick a different id")

    store.create_conversation(new_id, source["seed_text"])
    top_level = store.get_clusters(source_id, parent_id=None)
    for c in top_level:
        store.add_cluster(new_id, c["cluster_id"], c["label"], c["key_terms"])
        docs = store.docs_for_cluster(source_id, c["cluster_id"])
        store.link_docs_to_cluster(new_id, c["cluster_id"],
                                   [d["application_number"] for d in docs],
                                   source="cloned")

    print(f"New conversation : {new_id}")
    print(f"Seed             : {source['seed_text']}")
    print(f"Cloned {len(top_level)} top-level branches, all reset to live/0.0, from '{source_id}'")
    print(f"\nStart it with: python3 scripts/run_step2_chat.py {new_id}")


if __name__ == "__main__":
    main()
