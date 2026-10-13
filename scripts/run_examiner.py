#!/usr/bin/env python3
"""The examiner chatbot, run across the real network split.

    python3 scripts/run_examiner.py <conversation_id> [--disclosure <form.docx>]
    python3 scripts/run_examiner.py <conversation_id> --report

Always does whatever is next for that conversation:

  1. Not harvested yet       -> harvest the portal (INTERNET), then stop.
  2. Follow-up searches queued -> run them (INTERNET), then stop.
  3. Otherwise (INTRANET, needs the LLM):
       a. draft the claims from the disclosure and let you edit / accept them
       b. read every relevant harvested reference against every claim element
          (the claim chart) -- only references not yet charted
       c. the interview: one deep question at a time, full memory, resumable
       d. /report (or --report) writes out/<conversation_id>/report.md

--disclosure is only needed the first time if prepare_search.py was not
given one; the path is remembered after that.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.bootstrap import (
    run_harvest, run_harvest_ppubs, run_targeted_search, run_targeted_search_ppubs,
)
from priorart.core.examiner.chart import build_chart
from priorart.core.examiner.claims import (
    all_elements, claims_hash, derive_claims, review_gate,
)
from priorart.core.examiner.context import load_invention, retrieve
from priorart.core.examiner.interview import Examiner
from priorart.core.examiner.report import build_report
from priorart.core.examiner.searches import review_suggestions
from priorart.core.llm import LLMClient, LLMUnavailable
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_QID = "bootstrap"


def _run_portal_steps(conversation_id, site, cfg, llm, store) -> bool:
    """Harvest / queued searches. True if something ran (and we should stop)."""
    if not store.get_clusters(conversation_id):
        if site == "ppubs":
            print("Harvest not run yet -- searching PPUBS now (needs INTERNET, no captcha).\n")
            run_harvest_ppubs(conversation_id, cfg, llm, store)
        else:
            print("Harvest not run yet -- searching InPASS now (needs INTERNET).\n")
            run_harvest(conversation_id, cfg, llm, store, ROOT)
        print(f"\nHarvest done. Switch to INTRANET, then run this again:\n"
              f"  python3 {sys.argv[0]} {conversation_id}")
        return True

    pending = [p for p in store.get_pending_searches(conversation_id) if p["qid"] != _BOOTSTRAP_QID]
    if not pending:
        return False
    print(f"{len(pending)} queued search(es) -- running now (needs INTERNET"
          f"{', no captcha' if site == 'ppubs' else ''}).\n")
    failed = []
    for p in pending:
        try:
            if site == "ppubs":
                run_targeted_search_ppubs(conversation_id, p["qid"], cfg, store)
            else:
                run_targeted_search(conversation_id, p["qid"], cfg, store, ROOT)
        except Exception as exc:                                     # noqa: BLE001
            # Keep going -- a captcha typo on one must not cost the others.
            print(f"  '{p['qid']}' FAILED, left queued for next time: {exc}")
            failed.append(p["qid"])
        print()
    if failed:
        print(f"{len(failed)} search(es) still pending: {failed}")
    print(f"Done. Switch to INTRANET, then run this again:\n  python3 {sys.argv[0]} {conversation_id}")
    return True


def _accepted_claims(store, conversation_id, llm, invention, out_dir, ex_cfg):
    claims = store.kv_get(conversation_id, "claims")
    if claims and store.kv_get(conversation_id, "claims_accepted"):
        return claims

    draft = None
    old = sorted(ROOT.glob("priorart/out/*_claims.json"))
    if old:
        import json
        try:
            draft = json.loads(old[-1].read_text())
        except (OSError, ValueError):
            draft = None
    claims_thinking = ex_cfg.get("claims_thinking", False)
    print(f"Drafting claims from the disclosure "
          f"({'thinking' if claims_thinking else 'no thinking'} -- this can take a while)...")
    claims = derive_claims(llm, invention, draft=draft, thinking=claims_thinking)
    claims = review_gate(claims, out_dir / "claims.json",
                         regenerate=lambda: derive_claims(llm, invention, draft=draft,
                                                          thinking=claims_thinking))
    if claims_hash(claims) != store.kv_get(conversation_id, "claims_hash"):
        store.clear_chart(conversation_id)
    store.kv_set(conversation_id, "claims", claims)
    store.kv_set(conversation_id, "claims_hash", claims_hash(claims))
    store.kv_set(conversation_id, "claims_accepted", True)
    return claims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("conversation_id")
    ap.add_argument("--disclosure", help="B&P disclosure .docx (remembered after first use)")
    ap.add_argument("--report", action="store_true", help="write the report and exit")
    ap.add_argument("--redo-claims", action="store_true", help="draft and review the claims again")
    args = ap.parse_args()
    cid = args.conversation_id

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    ex_cfg = cfg.get("examiner", {})
    llm = LLMClient(cfg)
    store = Store(ROOT / "priorart/out/conversations.db")
    out_dir = ROOT / "priorart/out" / cid

    convo = store.get_conversation(cid)
    if not convo:
        sys.exit(f"no conversation '{cid}' -- start one on INTRANET with:\n"
                 f"  python3 scripts/prepare_search.py --disclosure <form.docx>")
    site = convo.get("site") or "inpass"
    print(f"Conversation : {cid}   site: {site}")

    if _run_portal_steps(cid, site, cfg, llm, store):
        return

    disclosure = args.disclosure or store.kv_get(cid, "disclosure_path")
    if not disclosure:
        sys.exit("which disclosure is this conversation about? pass --disclosure <form.docx> once")
    disclosure = str(Path(disclosure).resolve())
    if not Path(disclosure).exists():
        sys.exit(f"disclosure not found: {disclosure}")
    store.kv_set(cid, "disclosure_path", disclosure)
    invention = load_invention(disclosure)
    print(f"Disclosure   : {Path(disclosure).name}")

    if not llm.ping():
        sys.exit(f"LLM endpoint {llm.base_url} is not reachable -- switch to INTRANET and run again.")
    print(f"LLM          : {llm.model} @ {llm.base_url}\n")

    if args.redo_claims:
        store.kv_set(cid, "claims_accepted", False)
    try:
        claims = _accepted_claims(store, cid, llm, invention, out_dir, ex_cfg)
    except LLMUnavailable as exc:
        sys.exit(f"could not draft claims: {exc}")

    docs = store.all_documents() if ex_cfg.get("pool_all_documents", True) else store.all_conversation_docs(cid)
    refs = retrieve(docs, [e["text"] for e in all_elements(claims)],
                    domain_query=invention.text(("title", "nature", "purpose", "description")),
                    k_per_query=ex_cfg.get("refs_per_element", 6),
                    max_docs=ex_cfg.get("max_refs", 40))
    print(f"{len(docs)} harvested document(s); {len(refs)} selected as relevant to the claim elements.")
    build_chart(llm, store, cid, claims, refs, thinking=ex_cfg.get("chart_thinking", True),
                workers=cfg.get("endpoints", {}).get("llm", {}).get("max_concurrency", 4))

    def write_report():
        try:
            path = build_report(llm, store, cid, invention, claims, out_dir)
            print(f"\nReport written: {path}\n")
        except LLMUnavailable as exc:
            print(f"  report failed: {exc}")

    if args.report:
        write_report()
        return

    examiner = Examiner(store, cid, llm, invention, claims, refs, cfg)
    examiner.run(on_search=lambda: review_suggestions(store, cid, claims, site),
                 on_report=write_report)

    pending = [p for p in store.get_pending_searches(cid) if p["qid"] != _BOOTSTRAP_QID]
    if pending:
        print(f"\n{len(pending)} follow-up search(es) queued -- switch to INTERNET and run:\n"
              f"  python3 {sys.argv[0]} {cid}")


if __name__ == "__main__":
    main()
