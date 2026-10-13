#!/usr/bin/env python3
"""Step 2 standalone: the question loop over branches already found by step 1.

    python3 scripts/run_step2_chat.py <conversation_id>

Kept for testing step 2 in isolation against an already-harvested
conversation. For actually talking to the chatbot from scratch, use
scripts/run_chat.py instead -- it runs step 1's harvest automatically first,
then drops straight into this same loop, so you don't run two commands.

No InPASS access needed here -- this step is pure conversation, intranet only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.dialogue import run_dialogue_loop
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <conversation_id>")
    conversation_id = sys.argv[1]

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    convo = store.get_conversation(conversation_id)
    if not convo:
        sys.exit(f"no conversation '{conversation_id}' found -- run step 1 first")

    print(f"Conversation : {conversation_id}")
    print(f"Seed         : {convo['seed_text'][:80]}")
    print(f"LLM          : {'enabled -> ' + llm.model if llm.enabled else 'disabled (template questions, neutral hint reading)'}")

    run_dialogue_loop(store, conversation_id, llm)


if __name__ == "__main__":
    main()
