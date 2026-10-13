#!/usr/bin/env python3
"""Step 7: novelty comparison -- the final, separate step.

Compares your REAL invention (from the disclosure docx) against each
document in the final shortlist, one at a time, to help you judge how novel
your invention actually is relative to what's already been filed.

    python3 scripts/run_novelty_comparison.py <conversation_id> <disclosure_docx_path> [limit]

Runs entirely on the intranet LLM (needs it reachable) against documents
already downloaded by the conversation -- never touches InPASS or sends your
real invention details anywhere but that local model. This is deliberately a
separate step, run after the conversational narrowing has converged; it
never feeds back into or influences that narrowing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.llm import LLMClient
from priorart.core.novelty import compare_shortlist_to_invention, invention_text_from_disclosure
from priorart.core.shortlist import build_shortlist
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]

_CLOSENESS_ORDER = {"VERY_CLOSE": 0, "SOMEWHAT_CLOSE": 1, "DISTANT": 2, "UNKNOWN": 3}


def main():
    if len(sys.argv) < 3:
        sys.exit(f"usage: {sys.argv[0]} <conversation_id> <disclosure_docx_path> [limit]")
    conversation_id = sys.argv[1]
    disclosure_path = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    if not llm.enabled:
        sys.exit("endpoints.llm is disabled in config.yaml -- this step needs it reachable "
                 "(intranet).")

    convo = store.get_conversation(conversation_id)
    if not convo:
        sys.exit(f"no conversation '{conversation_id}' found")

    invention_text = invention_text_from_disclosure(disclosure_path)
    if not invention_text.strip():
        sys.exit(f"could not extract any real technical content from {disclosure_path} -- "
                 f"check the section headings still match priorart/core/ingest.py's template")

    entries = build_shortlist(store, conversation_id, limit=limit)
    if not entries:
        sys.exit("shortlist is empty -- run the conversation (scripts/run_chat.py) first")

    print(f"Comparing your real invention against {len(entries)} shortlisted document(s) "
         f"using {llm.model} (intranet only -- nothing sent to InPASS)...\n")

    documents = [{"application_number": e.application_number, "title": e.title,
                 "abstract": e.abstract} for e in entries]
    comparisons = compare_shortlist_to_invention(invention_text, documents, llm, limit=limit)
    comparisons.sort(key=lambda c: _CLOSENESS_ORDER.get(c.closeness, 3))

    for c in comparisons:
        print(f"{c.closeness:14} {c.application_number}  {c.title[:60]}")
        if c.method.startswith("llm"):
            print(f"    their technique : {c.technique}")
            print(f"    similarity      : {c.similarity}")
            print(f"    difference      : {c.difference}")
            shared = ", ".join(f.replace('_', ' ') for f in c.shared_facets) or "none"
            invention_only = ", ".join(f.replace('_', ' ') for f in c.invention_only_facets) or "none"
            print(f"    shared architecture (keyword-grounded)   : {shared}")
            print(f"    mine only, not in this doc (keyword-grounded): {invention_only}")
            if "capped" in c.method:
                print(f"    NOTE: closeness capped down -- no shared architecture keyword found, "
                     f"despite the model's own read")
        else:
            print(f"    (comparison unavailable this run -- left as UNKNOWN, not guessed)")
        print()

    very_close = [c for c in comparisons if c.closeness == "VERY_CLOSE"]
    if very_close:
        print(f"{len(very_close)} document(s) flagged VERY_CLOSE -- review these most carefully:")
        for c in very_close:
            print(f"  {c.application_number}: {c.title[:70]}")
    else:
        print("No document was flagged VERY_CLOSE.")


if __name__ == "__main__":
    main()
