#!/usr/bin/env python3
"""Compares the CURRENT hybrid search-row derivation (safe anchor rows,
expanded with real seed vocabulary) against a FULLY DYNAMIC derivation (no
fixed anchor skeleton at all -- the model decides everything from the seed).

Does NOT touch InPASS at all -- purely a side-by-side look, so a real
captcha and portal query are never spent on the experimental version until
someone has actually looked at what it produced and judged it trustworthy.

    python3 scripts/compare_search_rows.py "<seed sentence>"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.bootstrap import derive_search_rows, derive_search_rows_fully_dynamic
from priorart.core.llm import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} \"<seed sentence>\"")
    seed = sys.argv[1]

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    if not llm.enabled:
        sys.exit("LLM is disabled in config.yaml -- this comparison needs it reachable.")

    print(f"Seed  : {seed}")
    print(f"Model : {llm.model}\n")

    hybrid = derive_search_rows(seed, llm)
    print("CURRENT (hybrid: 3 safe anchor rows, expanded with real seed vocabulary):")
    print("-" * 68)
    for r in hybrid:
        print(f"  {r.field_code}: {r.text}")

    print()
    dynamic, reason = derive_search_rows_fully_dynamic(seed, llm)
    print("EXPERIMENTAL (fully dynamic -- no fixed anchor at all):")
    print("-" * 68)
    if dynamic is None:
        print(f"  REJECTED: {reason}")
        print("  This means it is NOT safe to use as-is; stick with the hybrid version above.")
    else:
        for r in dynamic:
            print(f"  {r.field_code}: {r.text}")
        print()
        print("Look at both. Does the dynamic version still clearly cover the SAME core")
        print("domain (e.g. drone/UAS AND detection both still represented somewhere) --")
        print("or has a whole concept quietly gone missing compared to the hybrid version?")
        print("Nothing has been run against InPASS -- this is a side-by-side look only.")


if __name__ == "__main__":
    main()
