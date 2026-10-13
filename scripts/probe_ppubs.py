#!/usr/bin/env python3
"""Live probe for priorart/ppubs/client.py -- hits the REAL USPTO Patent
Public Search API (no captcha, no browser). Not part of the automated test
suite (scripts/test_ppubs_client.py mocks the network); this is for
re-verifying the client still works against the real site, the same way
scripts/test_semantic_grouping.py etc. are real-data experiments kept
separate from unit tests.

    python3 scripts/probe_ppubs.py

Needs INTERNET (ppubs.uspto.gov), same network as InPASS -- unlike the LLM
endpoints, this needs no intranet access at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from priorart.ppubs.client import PpubsClient, SearchRow

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "priorart/config.yaml").read_text())
    print(f"Authenticating against {cfg['ppubs']['base_url']} (no captcha)...")
    client = PpubsClient(cfg)
    print(f"Got access token: {client.token[:12]}...\n")

    rows = [SearchRow("AB", "acoustic"), SearchRow("AB", "drone"), SearchRow("AB", "detection")]
    print(f"Searching: {' '.join(f'({r.text}).{r.field_code.lower()}.' for r in rows)}")
    result = client.search(rows, page_size=5)
    print(f"Total hits: {result.total}\n")
    for h in result.hits:
        print(f"  {h.application_number}  [{h.status}]  {h.title[:70]}")

    if not result.hits:
        print("\nNo hits -- nothing further to probe.")
        return

    print(f"\nFetching all hits across pages (cursor-based pagination)...")
    all_hits = list(client.iter_all_hits(rows, result))
    print(f"  {len(all_hits)} total hit(s) collected")

    first = result.hits[0]
    print(f"\nFetching real detail/abstract for {first.application_number}...")
    detail = client.fetch_detail(first.application_number, first.status)
    print(f"  abstract_suspect: {detail['abstract_suspect']}")
    print(f"  abstract: {detail['abstract'][:400]}")


if __name__ == "__main__":
    main()
