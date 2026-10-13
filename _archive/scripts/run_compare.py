#!/usr/bin/env python3
"""Run step (5)(6): compare harvested documents against claims -> verdict.

    python3 scripts/run_compare.py

Reads the claims saved by run_ingest.py and the documents+scores already in
the local database (from real InPASS searches). Needs the intranet LLM for
real per-claim judgement -- degrades to weak keyword-overlap rules otherwise
(rules can say "related", never "anticipates": that verdict requires actually
reading the text, which rules can't do).

No InPASS access needed here at all -- everything read is already local.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.claims import Claim
from priorart.core.compare import compare_documents, summarize_by_claim
from priorart.core.llm import LLMClient
from priorart.core.store import Store

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    profile = yaml.safe_load((ROOT / "priorart/profiles/acoustic_uav.yaml").read_text())
    llm = LLMClient(cfg)

    claims_files = sorted((ROOT / "priorart/out").glob("*_claims.json"))
    if not claims_files:
        sys.exit("No claims file found. Run scripts/run_ingest.py first.")
    raw_claims = json.loads(claims_files[-1].read_text())
    claims = [Claim(**c) for c in raw_claims]

    store = Store(ROOT / "priorart/out/priorart.db")
    run_id = store.latest_run(profile["meta"]["id"])
    if run_id is None:
        sys.exit("No search run found. Run some searches first.")

    max_tier = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = store.shortlist(run_id, max_tier=max_tier, limit=200)
    documents = [dict(r) for r in rows]

    print(f"Claims     : {len(claims)} ({sum(1 for c in claims if c.status=='claiming')} claiming)")
    print(f"Documents  : {len(documents)} (tier <= {max_tier})")
    print(f"LLM        : {'enabled -> ' + llm.model if llm.enabled else 'disabled (rules fallback)'}")
    print()

    comparisons = compare_documents(documents, claims, profile, llm=llm if llm.enabled else None)
    from collections import Counter
    method_counts = Counter(c.method for c in comparisons)
    print(f"[5/6] Compared {len(comparisons)} documents  -->  {dict(method_counts)}")
    fell_back = [c for c in comparisons if c.method == "rules"]
    if fell_back:
        print(f"      {len(fell_back)} document(s) fell back to keyword rules "
             f"(LLM call failed twice) -- these verdicts are UNVERIFIED, not a real read:")
        for c in fell_back[:5]:
            print(f"        {c.application_number}: {c.fallback_reason[:90]}")
    print()

    summary = summarize_by_claim(comparisons, claims)
    icons = {"exposed": "⚠️  EXPOSED", "watch": "👀 WATCH",
            "watch_unverified": "❓ WATCH (unverified - LLM unavailable for these)",
            "clear": "✅ CLEAR"}
    print("[6/6] VERDICT — per claim:\n")
    for cid, s in summary.items():
        print(f"  {cid} [{icons[s['status']]}]  {s['claim'][:90]}")
        for hit in s["anticipates"]:
            tag = "" if hit["verified"] else "  [UNVERIFIED - keyword rules only]"
            print(f"      ANTICIPATED BY {hit['application_number']}: {hit['title'][:65]}{tag}")
            print(f"          quote: {hit['quote'][:140]}")
        for hit in s["partial"][:3]:
            tag = "" if hit["verified"] else "  [UNVERIFIED - keyword rules only]"
            print(f"      partial overlap: {hit['application_number']}: {hit['title'][:65]}{tag}")
        print()

    out_dir = ROOT / "priorart" / "out"
    (out_dir / "comparisons.json").write_text(json.dumps(
        [{"application_number": c.application_number, "title": c.title, "method": c.method,
          "fallback_reason": c.fallback_reason,
          "verdicts": [vars(v) for v in c.verdicts]} for c in comparisons], indent=2))
    (out_dir / "verdict_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'comparisons.json'}")
    print(f"Saved: {out_dir / 'verdict_summary.json'}")


if __name__ == "__main__":
    main()
