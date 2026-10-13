#!/usr/bin/env python3
"""Step 0, run on INTRANET: derive the broad search rows.

    python3 scripts/prepare_search.py --disclosure <form.docx> [conversation_id] [--site=inpass|ppubs]
    python3 scripts/prepare_search.py "<seed sentence>" [conversation_id] [--site=inpass|ppubs]

With --disclosure (the normal way), the search vocabulary comes from the
disclosure's own title / nature / purpose sections -- never its inventive
features -- so any invention gets searched in its own terms. The disclosure
path is remembered for the examiner.

Uses the real LLM to turn your seed into 2-3 search concepts grounded in its
own words (never the exact proprietary technique) -- this needs the intranet
LLM endpoint, not the portal itself, so it's the one piece of starting a
conversation that has to happen before you're on internet, not during.

`--site` picks which portal this conversation searches (default inpass) --
recorded on the conversation itself, so run_examiner.py never needs to be
told again. ppubs needs no captcha at all.

Saves the derived rows and prints the conversation id to use next. Once
you're on internet, run:

    python3 scripts/run_examiner.py <conversation_id>

which harvests the portal with these exact rows (no further LLM call needed
for that step), then tells you to switch back to intranet for the interview.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.bootstrap import (
    prepare_search, prepare_search_from_disclosure, prepare_search_ppubs,
)
from priorart.core.examiner.context import DOMAIN_SECTIONS, load_invention
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    site = "inpass"
    disclosure = None
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--site="):
            site = a.split("=", 1)[1]
        elif a == "--site" and i + 1 < len(args):
            site = args[i + 1]
            i += 1
        elif a.startswith("--disclosure="):
            disclosure = a.split("=", 1)[1]
        elif a == "--disclosure" and i + 1 < len(args):
            disclosure = args[i + 1]
            i += 1
        else:
            positional.append(a)
        i += 1

    if site not in ("inpass", "ppubs"):
        sys.exit(f"--site must be 'inpass' or 'ppubs', got {site!r}")
    if disclosure:
        disclosure = str(Path(disclosure).resolve())
        seed = load_invention(disclosure).text(DOMAIN_SECTIONS)
        conversation_id = positional[0] if positional else uuid.uuid4().hex[:8]
    elif positional:
        seed = positional[0]
        conversation_id = positional[1] if len(positional) > 1 else uuid.uuid4().hex[:8]
    else:
        sys.exit(f"usage: {sys.argv[0]} --disclosure <form.docx> [conversation_id] [--site=inpass|ppubs]\n"
                 f"   or: {sys.argv[0]} \"<seed sentence>\" [conversation_id] [--site=inpass|ppubs]")

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")

    if store.get_conversation(conversation_id) and store.get_clusters(conversation_id):
        sys.exit(f"conversation '{conversation_id}' already has a harvest -- "
                 f"pick a different id, or just resume it with run_examiner.py")

    print(f"Conversation : {conversation_id}")
    print(f"Site         : {site}")
    print(f"Seed         : {seed[:300]}")
    print(f"LLM          : {'enabled -> ' + llm.model if llm.enabled else 'disabled (will fall back to the generic template)'}\n")

    if disclosure:
        rows, note = prepare_search_from_disclosure(seed, conversation_id, llm, store, site=site)
        store.kv_set(conversation_id, "disclosure_path", disclosure)
        if note:
            print(f"NOTE: {note}\n")
    else:
        rows = (prepare_search_ppubs if site == "ppubs" else prepare_search)(
            seed, conversation_id, llm, store)

    print("Derived search rows (grounded in the source text's own words):")
    for r in rows:
        print(f"  {r.field_code}: {r.text}")

    print(f"\nSaved. Now switch to INTERNET and run:\n"
         f"  python3 scripts/run_examiner.py {conversation_id}")


if __name__ == "__main__":
    main()
