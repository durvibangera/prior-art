#!/usr/bin/env python3
"""EXPERIMENTAL, comparison-only: tests whether a capable model can find
genuine THEMATIC groupings among a real "other"-bucket's documents by
reading their actual abstracts together, rather than requiring a literal
recurring word (the current discover_facets_from_leftovers mechanism).
Does NOT touch the live conversation or InPASS.

    python3 scripts/test_semantic_grouping.py <conversation_id> [cluster_id]

Defaults to the top-level "other" bucket if no cluster_id is given -- pass a
nested one (e.g. "deep_raw_audio__other") to test a drilled-into bucket.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.clusters import discover_semantic_groups
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <conversation_id> [cluster_id]")
    conversation_id = sys.argv[1]
    cluster_id = sys.argv[2] if len(sys.argv) > 2 else "other"

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    if not llm.enabled:
        sys.exit("LLM is disabled in config.yaml -- this experiment needs it reachable.")

    docs = store.docs_for_cluster(conversation_id, cluster_id)
    if not docs:
        sys.exit(f"no documents found in cluster '{cluster_id}' for conversation '{conversation_id}'")

    print(f"Reading {len(docs)} real documents from '{cluster_id}' TOGETHER, looking "
         f"for genuine shared themes (not just keyword overlap), using {llm.model}...\n")

    groups = discover_semantic_groups(docs, llm)

    print("SEMANTIC GROUPS FOUND (by reading abstracts together, not word-matching):")
    print("-" * 68)
    grouped_appnos = set()
    if not groups:
        print("  (none -- either genuinely no shared theme, or the LLM call failed)")
    for label, appnos in groups.items():
        print(f"\n  {label}  ({len(appnos)} documents)")
        for appno in appnos:
            d = store.get_document(appno)
            print(f"    {appno}: {(d.get('title') or '')[:70]}")
        grouped_appnos.update(appnos)

    ungrouped = [d for d in docs if d["application_number"] not in grouped_appnos]
    print(f"\n{len(ungrouped)} document(s) left genuinely ungrouped:")
    for d in ungrouped:
        print(f"    {d['application_number']}: {(d.get('title') or '')[:70]}")

    print("\nLook at the groups above -- do they actually share a coherent, specific theme,")
    print("or is this vague/forced? This is a pure look, nothing has been saved or changed.")


if __name__ == "__main__":
    main()
