#!/usr/bin/env python3
"""Run steps (1)(2)(3) of the plan: disclosure -> claims -> safe search questions.

    python3 scripts/run_ingest.py [path/to/disclosure.docx] [--profile path.yaml]

Defaults to profiles/acoustic_uav.yaml. Pass --profile to use one produced by
scripts/run_derive_profile.py, for any OTHER disclosure.

Works with NO endpoint at all (falls back to keyword rules). If
endpoints.llm.enabled is true in config.yaml AND reachable, it uses the LLM
for better claim phrasing and question wording -- but the portal rows it
proposes are always built by the same deterministic, capped code either way
(see core/questions.py) so the safety guarantee doesn't depend on the LLM
behaving.

This step needs no InPASS access at all -- everything here reads the local
disclosure file and (optionally) calls the intranet LLM. Safe to run on
whichever network is up.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.claims import extract_claims
from priorart.core.ingest import parse_docx
from priorart.core.llm import LLMClient
from priorart.core.questions import generate_questions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = next(ROOT.glob("*.docx"), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", nargs="?", type=Path, default=DEFAULT_DOC)
    ap.add_argument("--profile", type=Path,
                    default=ROOT / "priorart/profiles/acoustic_uav.yaml")
    a = ap.parse_args()

    if not a.doc or not a.doc.exists():
        sys.exit(f"No disclosure document found. Pass a path: {sys.argv[0]} <file.docx> [--profile path.yaml]")
    doc_path, profile_path = a.doc, a.profile

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    profile = yaml.safe_load(profile_path.read_text())
    print(f"Profile    : {profile_path.name}")
    llm = LLMClient(cfg)

    print(f"Disclosure : {doc_path.name}")
    print(f"LLM        : {'enabled -> ' + llm.model if llm.enabled else 'disabled (rules fallback)'}")
    print()

    # ── 1. Ingest ──────────────────────────────────────────────────────
    disclosure = parse_docx(str(doc_path))
    print(f"[1/3] Ingest -- {len(disclosure.sections)} sections parsed:")
    for key, text in disclosure.sections.items():
        flag = "empty" if not text else f"{len(text)} chars"
        print(f"      {key:24} {flag}")
    print()

    # ── 2. Claims ──────────────────────────────────────────────────────
    claims, method = extract_claims(disclosure, profile, llm=llm if llm.enabled else None)
    print(f"[2/3] Claims -- {len(claims)} extracted (method: {method}):")
    for c in claims:
        flag = "" if c.status == "claiming" else f"  [{c.status.upper()} - not a claim of ours]"
        print(f"      {c.id} [{c.pillar}]{flag}")
        print(f"         {c.text[:110]}")
    print()

    # ── 3. Questions ───────────────────────────────────────────────────
    questions = generate_questions(claims, profile, llm=llm if llm.enabled else None)
    n_claiming = sum(1 for c in claims if c.status == "claiming")
    print(f"[3/3] Search questions -- {len(questions)} distinct searches "
         f"cover {n_claiming} claiming-pillar claims:")
    for q in questions:
        print(f"      covers {q.claim_ids}: {q.question}")
        for f, t in q.rows:
            print(f"         {f}: {t[:85]}")
        if q.note:
            print(f"         note: {q.note}")
    print()

    # ── save ───────────────────────────────────────────────────────────
    out_dir = ROOT / "priorart" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = doc_path.stem.replace(" ", "_")[:40]

    claims_path = out_dir / f"{stem}_claims.json"
    claims_path.write_text(json.dumps([vars(c) for c in claims], indent=2))

    questions_path = out_dir / f"{stem}_questions.json"
    questions_path.write_text(json.dumps([vars(q) for q in questions], indent=2))

    print(f"Saved: {claims_path}")
    print(f"Saved: {questions_path}")
    print()
    print("Review the claims above -- correct any before they drive a search.")
    print("Nothing above was sent anywhere; these are local files only.")


if __name__ == "__main__":
    main()
