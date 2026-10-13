#!/usr/bin/env python3
"""Derive a search profile from ANY disclosure document -- the step that
makes "upload any document" actually true (see PLAN.md 7b).

    python3 scripts/run_derive_profile.py path/to/disclosure.docx [profile_id]

Writes priorart/profiles/<profile_id>.yaml, ready to use exactly like
acoustic_uav.yaml with run_ingest.py / run_compare.py / the web app.

ALWAYS review the output before trusting it -- especially the pillar statuses
(the tool defaults every pillar to "claiming" since it cannot know your
business intent from the text alone; correct any that should be out_of_scope
or not_ours) and, if this ran on the rules fallback, every facet's word list.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.core.ingest import parse_docx
from priorart.core.llm import LLMClient
from priorart.core.profile_builder import derive_profile, to_profile_dict

ROOT = Path(__file__).resolve().parents[1]


def slugify(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "profile"


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <disclosure.docx> [profile_id]")
    doc_path = Path(sys.argv[1])
    if not doc_path.exists():
        sys.exit(f"not found: {doc_path}")

    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    llm = LLMClient(cfg)

    print(f"Disclosure : {doc_path.name}")
    print(f"LLM        : {'enabled -> ' + llm.model if llm.enabled else 'disabled (rules fallback)'}")
    print()

    disclosure = parse_docx(str(doc_path))
    proposed = derive_profile(disclosure, llm=llm if llm.enabled else None)

    profile_id = sys.argv[2] if len(sys.argv) > 2 else slugify(proposed.title)
    print(f"Method     : {proposed.method}")
    print(f"Title      : {proposed.title}")
    print(f"Profile id : {profile_id}")
    print()

    print("FACETS:")
    for key, f in proposed.facets.items():
        core_tag = "  <- CORE TECHNIQUE" if key == proposed.core_facet else ""
        req_tag = " (required)" if f.get("required") else ""
        print(f"  {key:20} w={f['weight']:<4}{req_tag}{core_tag}")
        print(f"      words:   {', '.join(f['words'])}")
        if f["phrases"]:
            print(f"      phrases: {', '.join(f['phrases'])}")
    print()

    print("PILLARS (review status: claiming / out_of_scope / not_ours):")
    for key, pl in proposed.pillars.items():
        print(f"  {key:20} [{pl.get('status')}]  {pl.get('label')}")
    print()

    print(f"FINGERPRINT: {proposed.fingerprint[:200]}...")
    print()

    if proposed.method == "rules":
        print("!! Rules fallback used (no LLM) -- this is a WEAK skeleton.")
        print("!! Review every facet's word list before running any search with it.")
        print()

    full = to_profile_dict(proposed, profile_id)
    out_path = ROOT / "priorart" / "profiles" / f"{profile_id}.yaml"
    out_path.write_text(yaml.safe_dump(full, sort_keys=False, allow_unicode=True, width=100))
    print(f"Saved: {out_path}")
    print()
    print(f"Next: python3 scripts/run_ingest.py {doc_path} --profile {out_path}")
    print("(review the YAML file above first, especially pillar statuses)")


if __name__ == "__main__":
    main()
