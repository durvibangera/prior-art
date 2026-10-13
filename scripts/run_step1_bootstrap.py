#!/usr/bin/env python3
"""Step 1 standalone: seed -> one broad safe search -> cluster -> label.

    python3 scripts/run_step1_bootstrap.py "<seed sentence>" [conversation_id]

Kept for testing step 1 in isolation. For actually talking to the chatbot,
use scripts/run_examiner.py instead -- it runs this same step automatically as
its first move, then goes straight into the conversation, so you don't have
to run two separate commands.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.bootstrap import bootstrap_conversation
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} '<seed sentence>' [conversation_id]")
    seed = sys.argv[1]
    conversation_id = sys.argv[2] if len(sys.argv) > 2 else uuid.uuid4().hex[:8]

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    print(f"Conversation : {conversation_id}")
    print(f"Seed         : {seed}")
    print(f"LLM          : {'enabled -> ' + llm.model if llm.enabled else 'disabled (fallback labels)'}\n")

    bootstrap_conversation(seed, conversation_id, cfg, llm, store, ROOT)

    print(f"Saved to conversation '{conversation_id}' in priorart/out/conversations.db")
    print("Next: the question loop (step 2) picks two of these branches and asks about them.")


if __name__ == "__main__":
    main()
