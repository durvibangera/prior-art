#!/usr/bin/env python3
"""The chatbot, run across the real network split.

    python3 scripts/run_chat.py <conversation_id>

Always run with just a conversation id -- this always does whatever's next
for that conversation, and tells you which network you need if it can't
continue any further in this one call:

  1. No conversation with that id yet -> use scripts/prepare_search.py first
     (on intranet) to derive the search rows from your seed and create it.
  2. Rows prepared but not yet searched -> this call harvests InPASS with
     them (needs INTERNET: one captcha, typed by you), then stops and tells
     you to switch to intranet and run this same command again.
  3. A branch got confident enough that a real targeted search is queued
     for it -> this call runs it (needs INTERNET: another captcha), then
     stops the same way. This can happen again later in the same
     conversation as more branches confirm.
  4. Nothing queued -> goes straight into the conversation (needs INTRANET,
     for the LLM). If a branch confirms mid-conversation, a new targeted
     search gets queued for next time, but the conversation keeps going
     with whatever's already known rather than interrupting itself.

This way one command always does the right thing for wherever you are, and
you never have to remember which script does which half.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.bootstrap import (
    run_harvest, run_harvest_ppubs, run_targeted_search, run_targeted_search_ppubs,
)
from priorart.core.dialogue import run_dialogue_loop
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_QID = "bootstrap"


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <conversation_id>\n\n"
                 f"To start a brand new conversation, run scripts/prepare_search.py "
                 f"first (on intranet) -- it prints the conversation id to use here.")
    conversation_id = sys.argv[1]

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    convo = store.get_conversation(conversation_id)
    if not convo:
        sys.exit(f"no conversation '{conversation_id}' found -- run "
                 f"scripts/prepare_search.py \"<seed sentence>\" first (on intranet)")

    site = convo.get("site") or "inpass"
    print(f"Conversation : {conversation_id}")
    print(f"Site         : {site}")
    print(f"Seed         : {convo['seed_text'][:80]}")

    harvested = bool(store.get_clusters(conversation_id))
    if not harvested:
        if site == "ppubs":
            print("Harvest not run yet -- searching PPUBS now (needs INTERNET, no captcha).\n")
            run_harvest_ppubs(conversation_id, cfg, llm, store)
        else:
            print("Harvest not run yet -- searching InPASS now (needs INTERNET).\n")
            run_harvest(conversation_id, cfg, llm, store, ROOT)
        print(f"Harvest done. Switch to INTRANET, then run this again to start talking:\n"
             f"  python3 {sys.argv[0]} {conversation_id}")
        return

    pending = [p for p in store.get_pending_searches(conversation_id)
              if p["qid"] != _BOOTSTRAP_QID]
    if pending:
        print(f"{len(pending)} real targeted search(es) queued from confirmed branches -- "
             f"running now (needs INTERNET{', no captcha' if site == 'ppubs' else ''}).\n")
        failed = []
        for p in pending:
            try:
                if site == "ppubs":
                    run_targeted_search_ppubs(conversation_id, p["target_cluster_id"], cfg, store)
                else:
                    run_targeted_search(conversation_id, p["target_cluster_id"], cfg, store, ROOT)
            except Exception as exc:                                 # noqa: BLE001
                # One search failing (a transient portal error, a captcha
                # typo, etc) must not lose the captchas you already solved
                # for the OTHERS in this same batch -- left pending, so the
                # next run just retries this one instead of redoing everything.
                print(f"  '{p['target_cluster_id']}' FAILED, left queued for next time: {exc}")
                failed.append(p["target_cluster_id"])
            print()
        if failed:
            print(f"{len(failed)} search(es) still pending after a failure: {failed}")
        print(f"Done. Switch to INTRANET, then run this again to keep talking:\n"
             f"  python3 {sys.argv[0]} {conversation_id}")
        return

    print(f"LLM          : {'enabled -> ' + llm.model if llm.enabled else 'disabled (template questions, neutral hint reading)'}")
    run_dialogue_loop(store, conversation_id, llm)

    still_pending = [p for p in store.get_pending_searches(conversation_id)
                    if p["qid"] != _BOOTSTRAP_QID]
    if still_pending:
        print(f"\n{len(still_pending)} new targeted search(es) queued -- switch to "
             f"INTERNET and run this again to fetch them:\n"
             f"  python3 {sys.argv[0]} {conversation_id}")


if __name__ == "__main__":
    main()
