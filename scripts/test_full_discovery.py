#!/usr/bin/env python3
"""One-off experiment, NOT part of the regular test suite: runs the
LLM-gated discovery mechanism against an ENTIRE real harvest from scratch,
with no APPROACH_FACETS at all, to see what a capable model actually finds
unassisted. Needs intranet (the real LLM endpoint).

    python3 scripts/test_full_discovery.py <conversation_id>

Prints every candidate word that cleared the frequency bar, alongside
whether the LLM gate accepted or rejected it and why -- so the decision
about whether to remove the hand-written top-level vocabulary can be made
from real evidence, not a hunch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.clusters import (
    MIN_CLUSTER_SIZE, _word_is_technical, discover_facets_from_leftovers,
)
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

    if not llm.enabled:
        sys.exit("LLM is disabled in config.yaml -- this experiment needs it reachable.")

    docs = store.all_conversation_docs(conversation_id)
    if not docs:
        sys.exit(f"no documents found for conversation '{conversation_id}'")

    print(f"Running fully-dynamic discovery over all {len(docs)} real documents "
         f"(no hand-written vocabulary at all) using {llm.model}...\n")

    # existing_vocab={} means nothing is pre-excluded as "already covered" --
    # a genuinely from-scratch pass, unlike the normal "other"-bucket use.
    discovered = discover_facets_from_leftovers(
        docs, existing_vocab={}, min_doc_support=MIN_CLUSTER_SIZE,
        max_new_facets=15, llm=llm)

    print("ACCEPTED as real technical branches:")
    print("-" * 68)
    if not discovered:
        print("  (none)")
    for key, terms in discovered.items():
        print(f"  {key:20} seed term(s): {terms}")

    print(f"\n{len(discovered)} branch(es) discovered from scratch, gate-verified.")
    print("Compare this against the hand-written APPROACH_FACETS list to judge "
         "whether full dynamic discovery is ready to replace it.")


if __name__ == "__main__":
    main()
