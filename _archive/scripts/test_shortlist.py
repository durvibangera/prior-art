#!/usr/bin/env python3
"""Unit tests for priorart/core/shortlist.py -- Step 6, the final ranked
shortlist. This is what actually delivers "eliminating abstracts as we go":
a document that only ever matched a pruned branch must never appear.
Run: python3 scripts/test_shortlist.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.shortlist import build_shortlist

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class FakeStore:
    def __init__(self, docs_with_clusters, verdicts=None):
        self._docs = docs_with_clusters
        self._verdicts = verdicts or {}

    def documents_with_cluster_info(self, conversation_id):
        return self._docs

    def get_verdicts(self, conversation_id):
        return self._verdicts


def doc(appno, clusters, title="", abstract=""):
    return {"application_number": appno, "title": title or appno, "abstract": abstract,
           "clusters": clusters}


def c(cluster_id, status, score):
    return {"cluster_id": cluster_id, "status": status, "score": score}


print("\n1. a document that ONLY matched a pruned cluster is excluded")
store = FakeStore([
    doc("D1", [c("vision_based", "pruned", -2.0)]),
    doc("D2", [c("thermal_based", "confirmed", 3.0)]),
])
result = build_shortlist(store, "c1")
ids = [e.application_number for e in result]
check("D1 (pruned-only) excluded", "D1" not in ids, ids)
check("D2 (confirmed cluster) included", "D2" in ids, ids)

print("\n2. a document matching BOTH a pruned and a still-live cluster is kept")
store = FakeStore([
    doc("D3", [c("vision_based", "pruned", -2.0), c("multi_sensor_fusion", "live", 1.0)]),
])
result = build_shortlist(store, "c1")
check("D3 kept (not exclusively pruned)", len(result) == 1 and result[0].application_number == "D3", result)
check("ranked by the non-pruned cluster's score, not the pruned one",
     result[0].rank_score == 1.0, result[0].rank_score)

print("\n3. a directly REJECTED document is dropped regardless of cluster status")
store = FakeStore(
    [doc("D4", [c("thermal_based", "confirmed", 5.0)])],
    verdicts={"D4": "rejected"})
result = build_shortlist(store, "c1")
check("rejected document dropped even with a strong confirmed cluster", len(result) == 0, result)

print("\n4. a directly CONFIRMED document floats to the top regardless of score")
store = FakeStore([
    doc("D5", [c("weak_branch", "live", 0.1)]),
    doc("D6", [c("strong_branch", "confirmed", 5.0)]),
], verdicts={"D5": "confirmed"})
result = build_shortlist(store, "c1")
check("D5 (direct confirm, low score) ranks first", result[0].application_number == "D5", result)
check("D6 (high cluster score, no direct verdict) ranks second", result[1].application_number == "D6", result)

print("\n5. sorting by score among non-verdicted documents")
store = FakeStore([
    doc("LOW", [c("a", "live", 0.5)]),
    doc("HIGH", [c("b", "confirmed", 3.0)]),
    doc("MID", [c("c", "live", 1.5)]),
])
result = build_shortlist(store, "c1")
check("sorted descending by score", [e.application_number for e in result] == ["HIGH", "MID", "LOW"],
     [e.application_number for e in result])

print("\n6. limit is respected")
store = FakeStore([doc(f"D{i}", [c("x", "live", float(i))]) for i in range(10)])
result = build_shortlist(store, "c1", limit=3)
check("only 3 returned", len(result) == 3, len(result))
check("the 3 highest-scored ones", {e.application_number for e in result} == {"D9", "D8", "D7"},
     {e.application_number for e in result})

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
