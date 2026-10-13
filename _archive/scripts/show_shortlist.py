#!/usr/bin/env python3
"""View the current ranked shortlist for a conversation, at any point --
does not need the conversation to have converged or stopped.

    python3 scripts/show_shortlist.py <conversation_id> [limit]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.dialogue import show_shortlist
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <conversation_id> [limit]")
    conversation_id = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    store = Store(ROOT / "priorart/out/conversations.db")
    convo = store.get_conversation(conversation_id)
    if not convo:
        sys.exit(f"no conversation '{conversation_id}' found")

    print(f"Conversation : {conversation_id}")
    print(f"Seed         : {convo['seed_text'][:80]}")
    show_shortlist(store, conversation_id, limit=limit)


if __name__ == "__main__":
    main()
