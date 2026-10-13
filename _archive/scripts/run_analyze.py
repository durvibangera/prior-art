#!/usr/bin/env python3
"""Stage A -- read a disclosure and print the architecture it understood.

    python3 scripts/run_analyze.py [disclosure.docx]

Needs the intranet LLM. Without it, prints the deterministic pillar coverage
and says plainly that decomposition was not attempted -- it does not guess.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.ingest import parse_docx
from priorart.core.invention import build_invention_model
from priorart.core.llm import LLMClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = next(ROOT.glob("*.docx"), None)

LEVEL_MARK = {"rich": "●●●", "thin": "●○○", "empty": "○○○"}


def main():
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC
    if not doc or not doc.exists():
        sys.exit(f"usage: {sys.argv[0]} <disclosure.docx>")

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)
    print(f"Disclosure : {doc.name}")
    print(f"LLM        : {'enabled -> ' + llm.model if llm.enabled else 'disabled'}\n")

    disclosure = parse_docx(str(doc))
    model = build_invention_model(disclosure, llm=llm if llm.enabled else None)

    print(f"TITLE      : {model.title[:100]}")
    print(f"Method     : {model.method}    Stability: {model.stability}\n")

    print("PILLAR COVERAGE  (measured in code, no model involved)")
    print("-" * 68)
    for key, c in model.pillar_coverage.items():
        flag = ""
        if c["status"] == "claiming" and c["level"] == "empty":
            flag = "   <-- CLAIMING BUT EMPTY"
        print(f"  {LEVEL_MARK[c['level']]}  {key:30} {c['status']:12} "
              f"{c['hits']:3} hits{flag}")
        if c["suppressed"]:
            print(f"        false positives suppressed: {c['suppressed']}")
    print()

    if model.components:
        print("ARCHITECTURE")
        print("-" * 68)
        for c in model.components:
            marks = []
            if c.unsupported:
                marks.append("NOT IN DOCUMENT")
            elif not c.evidence_verified:
                marks.append("evidence unverified")
            if getattr(c, "misplaced", False):
                marks.append("JOB BLEED - method from a later stage")
            mark = f"   <-- {' / '.join(marks)}" if marks else ""
            print(f"\n  {c.id}  {c.name}    [{c.pillar or 'unassigned'}]{mark}")
            print(f"      {c.function}")
            if c.our_approach:
                print(f"      You use   : {c.our_approach}")
            if c.settings:
                print(f"      Settings  : {', '.join(c.settings)}")
            if c.known_alternatives:
                for a in c.known_alternatives:
                    tag = "*" if a.source == "disclosure" else " "
                    print(f"      Others use:{tag} {a.name}")
            if c.novelty_claim:
                print(f"      Newness   : {c.novelty_claim}")
            if c.alternatives_weak:
                print(f"      ! weak alternatives ({len(c.known_alternatives)}) "
                      f"- competitor search will be thin here")
        print("\n  (* = named in the disclosure's own prior-art section; "
              "unmarked = model recall, treat as a hint)")

    if model.prior_art_methods:
        print("\nCOMPETITOR METHODS named in the disclosure's own prior art")
        print("-" * 68)
        print("  " + " | ".join(model.prior_art_methods))
        print("  (these drive the 'who are my real competitors' probe)")

    if model.notes:
        print("\nNOTES")
        print("-" * 68)
        for n in model.notes:
            print(f"  - {n}")

    out = ROOT / "priorart" / "out" / f"{doc.stem[:40]}_invention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(model)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
