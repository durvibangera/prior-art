#!/usr/bin/env python3
"""Unit tests for priorart/core/clusters.py -- deterministic document
grouping, combination sub-clustering, and dynamic facet discovery.
Run: python3 scripts/test_clusters.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.clusters import (
    APPROACH_FACETS, MIN_CLUSTER_SIZE, Cluster, _word_is_technical,
    discover_facets_from_leftovers, discover_semantic_groups, expand_other_with_semantic_groups,
    expand_with_discovered_facets, group_documents, sub_cluster_by_combination,
)
from priorart.core.llm import KVResult

PASS, FAIL = 0, 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def doc(appno, title="", abstract=""):
    return {"application_number": appno, "title": title, "abstract": abstract}


print("\n1. group_documents grounds membership in real whole-word matches")
docs = [doc(f"D{i}", abstract="uses a CNN on raw WAVEFORM audio") for i in range(6)]
clusters = group_documents(docs)
check("deep_raw_audio found", "deep_raw_audio" in clusters, clusters.keys())
check("all 6 docs in it", len(clusters["deep_raw_audio"].doc_ids) == 6)

print("\n2. below MIN_CLUSTER_SIZE folds into 'other', never silently dropped")
docs = [doc(f"D{i}", abstract="uses RADAR doppler shift") for i in range(3)]
clusters = group_documents(docs)
check("small radar bucket folded to other", "radar_based" not in clusters)
check("all 3 docs preserved in other", len(clusters.get("other", Cluster(key="other")).doc_ids) == 3)

print("\n3. sub_cluster_by_combination only produces REAL combinations, never invented ones")
docs = ([doc(f"V{i}", abstract="VISION camera") for i in range(6)] +
       [doc(f"T{i}", abstract="THERMAL infrared") for i in range(6)])
sub = sub_cluster_by_combination(docs, {"vision_based": APPROACH_FACETS["vision_based"],
                                        "thermal_based": APPROACH_FACETS["thermal_based"]})
check("no vision+thermal combo invented (none co-occur)", "thermal_based+vision_based" not in sub, sub.keys())
check("vision-only and thermal-only each present", {"vision_based", "thermal_based"} <= set(sub.keys()), sub.keys())

print("\n4. REAL FAILURE: a too-generic recurring word ('sensors') must not become a fake branch")
# Captured live: 6 genuinely unrelated real documents (drone swarm music,
# smart water bottle, honey extraction, forest management, ...) all happened
# to mention "sensors" as generic hardware boilerplate -- grouping them under
# a "sensors" branch would have looked like a real pattern but wasn't one.
docs = [
    doc("D0", title="SWARM CONTROL OF DRONES", abstract="microcontroller and sensors coordinate playback"),
    doc("D1", title="SMART WATER BOTTLE", abstract="sensors monitor water quality for health"),
    doc("D2", title="HONEY REMOVAL MODULE", abstract="sensors detect hive fullness for extraction"),
    doc("D3", title="FOREST MANAGEMENT TWIN", abstract="sensors feed a digital twin of forest resources"),
    doc("D4", title="INFRASTRUCTURE INSPECTION", abstract="sensors on a UAV payload inspect civil structures"),
    doc("D5", title="NOISE POLLUTION MONITORING", abstract="sensors log noise pollution in smart cities"),
]
# require_llm_gate=False below tests the FREQUENCY/stopword logic in
# isolation from the LLM sanity gate (tested separately further down) --
# see discover_facets_from_leftovers's own docstring for why that knob
# exists.
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           require_llm_gate=False)
check("'sensors' not discovered as a fake branch", "sensors" not in new_vocab, new_vocab)

print("\n5. a genuinely distinctive recurring term IS discovered")
docs = [doc(f"D{i}", abstract=f"a LIDAR based ranging module, variant {i}") for i in range(5)]
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           require_llm_gate=False)
check("LIDAR discovered", "lidar" in new_vocab, new_vocab)

print("\n6. discovery never rediscovers an existing facet's own vocabulary under a new name")
docs = [doc(f"D{i}", abstract="a WAVELET SCATTERING transform frontend") for i in range(5)]
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           require_llm_gate=False)
check("WAVELET already covered, not rediscovered", "wavelet" not in new_vocab, new_vocab)

print("\n7. expand_with_discovered_facets moves matching docs out of 'other', leaves the rest")
docs = ([doc(f"L{i}", abstract=f"a LIDAR ranging system {i}") for i in range(5)] +
       [doc("X0", abstract="a completely unrelated smart water bottle")])
clusters = {"other": Cluster(key="other", doc_ids=[d["application_number"] for d in docs])}
docs_by_id = {d["application_number"]: d for d in docs}
clusters, discovered = expand_with_discovered_facets(clusters, docs_by_id, APPROACH_FACETS,
                                                      require_llm_gate=False)
check("lidar branch created", "lidar" in discovered, discovered)
check("lidar branch has the 5 real docs", len(clusters["lidar"].doc_ids) == 5, clusters.get("lidar"))
check("the 1 unrelated doc stays in other", clusters["other"].doc_ids == ["X0"], clusters["other"].doc_ids)

print("\n8. below-threshold 'other' bucket is left alone, no wasted discovery pass")
docs = [doc(f"D{i}", abstract="a LIDAR system") for i in range(2)]  # below MIN_CLUSTER_SIZE
clusters = {"other": Cluster(key="other", doc_ids=[d["application_number"] for d in docs])}
docs_by_id = {d["application_number"]: d for d in docs}
clusters, discovered = expand_with_discovered_facets(clusters, docs_by_id, APPROACH_FACETS)
check("nothing discovered below the size floor", discovered == {}, discovered)

print("\n9. REAL FAILURE: without the LLM gate, generic words become fake branches by default")
# Captured live TWICE: "complete"/"specification" first, then "designed"/
# "efficient" -- words that clear the frequency bar without naming any real
# technique, presented to a person as if they were real branches to answer
# questions about. The gate defaults ON, so without an llm passed at all
# (the honest "can't verify" case) nothing gets accepted, rather than
# defaulting to trusting the raw frequency signal.
docs = [doc(f"D{i}", abstract="an efficient system designed for robust performance") for i in range(6)]
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5)
check("no llm -> gate rejects everything, no fake branches slip through", new_vocab == {}, new_vocab)


class GateLLM:
    """Says TECHNICAL for LIDAR, GENERIC for anything else -- exercises the
    real validation path via the spec's own validator, not a hardcoded
    errors=[] shortcut."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        answer = "1" if "WORD: LIDAR" in user else "2"
        validate = spec.fields[0].validate
        err = validate(answer) if validate else None
        if err:
            return KVResult(records=[], errors=[f"ANSWER {err}"])
        return KVResult(records=[{"ANSWER": answer}], errors=[])


print("\n10. the LLM gate itself: technical word accepted, generic word rejected")
check("LIDAR (technical) accepted", _word_is_technical("LIDAR", GateLLM()) is True)
check("EFFICIENT (generic) rejected", _word_is_technical("EFFICIENT", GateLLM()) is False)


class RelevanceGateLLM:
    """A real gate would say TECHNICAL for genuine sensing/detection words
    but REJECT technical words about something else entirely -- exercises
    the real validation path, not a hardcoded shortcut."""
    enabled = True
    SENSING_WORDS = ("LIDAR", "THERMAL", "RADAR")

    def ask(self, system, user, spec, **kw):
        answer = "1" if any(f"WORD: {w}" in user for w in self.SENSING_WORDS) else "2"
        validate = spec.fields[0].validate
        err = validate(answer) if validate else None
        if err:
            return KVResult(records=[], errors=[f"ANSWER {err}"])
        return KVResult(records=[{"ANSWER": answer}], errors=[])


print("\n10b. REAL FAILURE: a genuinely technical word that's irrelevant to detection")
# Captured live: a from-scratch discovery run over 110 real documents
# accepted "SOLAR" (power supply), "BLOCKCHAIN" (data security), and
# "SWARM" (multi-vehicle coordination) as real branches -- all genuinely
# specific, technical words, none of them about how the system senses or
# detects anything. Being technical is necessary but not sufficient.
for irrelevant in ("SOLAR", "BLOCKCHAIN", "SWARM"):
    check(f"{irrelevant} (real but detection-irrelevant) rejected",
         _word_is_technical(irrelevant, RelevanceGateLLM()) is False)
check("a genuine sensing word (RADAR) still accepted",
     _word_is_technical("RADAR", RelevanceGateLLM()) is True)

print("\n11. end-to-end: the gate lets a real technical word through, blocks a generic one")
docs = ([doc(f"L{i}", abstract=f"a LIDAR ranging module {i}") for i in range(5)] +
       [doc(f"E{i}", abstract=f"an efficient robust system {i}") for i in range(5)])
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           llm=GateLLM())
check("lidar accepted end-to-end", "lidar" in new_vocab, new_vocab)
check("efficient rejected end-to-end", "efficient" not in new_vocab, new_vocab)

print("\n12. REAL FAILURE: a plural word-form of an already-covered facet term slips through")
# Captured live: "CAMERAS" (plural) was discovered as a "new" sub-branch of
# an audio-only cluster even though "CAMERA" (singular) is already
# vision_based's own term -- exact-string exclusion missed the word-form
# variant, producing a nonsense "cameras" branch inside deep_raw_audio.
docs = [doc(f"D{i}", abstract="a raw WAVEFORM CNN model, verified using extra CAMERAS on site") for i in range(6)]
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           require_llm_gate=False)
check("'cameras' (plural) excluded via stem match against 'camera'", "cameras" not in new_vocab, new_vocab)

print("\n13. REAL FAILURE: a domain synonym ('aerial') and an acronym synonym ('convolutional')")
# Captured live: "AERIAL" (a synonym for drone/UAV, present in nearly every
# document by construction) and "CONVOLUTIONAL" (a spelled-out synonym for
# the already-covered "CNN") both slipped through as fake new branches.
docs = [doc(f"D{i}", abstract="an AERIAL vehicle using a CONVOLUTIONAL neural network") for i in range(6)]
new_vocab = discover_facets_from_leftovers(docs, APPROACH_FACETS, min_doc_support=5,
                                           require_llm_gate=False)
check("'aerial' excluded as a domain synonym", "aerial" not in new_vocab, new_vocab)
check("'convolutional' excluded as a CNN synonym", "convolutional" not in new_vocab, new_vocab)

print("\n14. EXPERIMENTAL: discover_semantic_groups finds a real thematic group")
semantic_docs = [
    doc("D1", title="ELEPHANT DETECTION SYSTEM", abstract="acoustic monitoring for elephant conflict"),
    doc("D2", title="BIRD MONITORING", abstract="acoustic detection of bird calls in forest"),
    doc("D3", title="SMART WATER BOTTLE", abstract="monitors hydration levels for health"),
    doc("D4", title="HONEY EXTRACTION", abstract="automated drone module for honey removal"),
]


class SemanticMockLLM:
    """Simulates a real, sensible grouping decision -- exercises the actual
    parsing/validation path, not a hand-picked shortcut."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[{"GROUP": "wildlife acoustic monitoring", "DOCS": "1 2"}], errors=[])


groups = discover_semantic_groups(semantic_docs, SemanticMockLLM())
check("real thematic group found", "wildlife acoustic monitoring" in groups, groups)
check("group contains exactly the 2 real matching documents",
     groups.get("wildlife acoustic monitoring") == ["D1", "D2"], groups)

print("\n15. REAL RISK: a hallucinated document number must never be trusted")
class HallucinatingSemanticLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[{"GROUP": "fake group", "DOCS": "1 5 99"}], errors=[])


groups2 = discover_semantic_groups(semantic_docs, HallucinatingSemanticLLM())
check("hallucinated indices (5, 99) dropped, real group falls below min size and is discarded",
     groups2 == {}, groups2)

print("\n16. no LLM available -> empty, never a forced grouping")
check("no llm returns empty dict", discover_semantic_groups(semantic_docs, None) == {})

print("\n17. expand_other_with_semantic_groups splits a real group off 'other' at bootstrap time")
bootstrap_docs_by_id = {d["application_number"]: d for d in semantic_docs}
bootstrap_clusters = {"other": Cluster(key="other", doc_ids=[d["application_number"] for d in semantic_docs])}
bootstrap_clusters = expand_other_with_semantic_groups(
    bootstrap_clusters, bootstrap_docs_by_id, SemanticMockLLM(), min_doc_support=2)
check("new top-level branch created from the semantic group",
     "wildlife_acoustic_monitoring" in bootstrap_clusters, bootstrap_clusters.keys())
new_branch = bootstrap_clusters.get("wildlife_acoustic_monitoring")
check("branch carries the LLM's own label directly (not re-labeled)",
     new_branch and new_branch.label == "wildlife acoustic monitoring", new_branch)
check("branch contains exactly the 2 real matching documents",
     new_branch and new_branch.doc_ids == ["D1", "D2"], new_branch)
check("those 2 documents removed from 'other', 2 unrelated ones remain",
     sorted(bootstrap_clusters["other"].doc_ids) == ["D3", "D4"], bootstrap_clusters["other"].doc_ids)

print("\n18. a derived key that collides with an existing branch gets disambiguated, not overwritten")
colliding_clusters = {
    "other": Cluster(key="other", doc_ids=[d["application_number"] for d in semantic_docs]),
    "wildlife_acoustic_monitoring": Cluster(key="wildlife_acoustic_monitoring", doc_ids=["PRE_EXISTING"]),
}
colliding_clusters = expand_other_with_semantic_groups(
    colliding_clusters, bootstrap_docs_by_id, SemanticMockLLM(), min_doc_support=2)
check("pre-existing branch left untouched",
     colliding_clusters["wildlife_acoustic_monitoring"].doc_ids == ["PRE_EXISTING"],
     colliding_clusters["wildlife_acoustic_monitoring"])
check("new group given a disambiguated key instead",
     "wildlife_acoustic_monitoring_2" in colliding_clusters, colliding_clusters.keys())

print("\n19. below min_doc_support -> left alone, no call even attempted")
class ExplodingLLM:
    enabled = True

    def ask(self, *a, **kw):
        raise AssertionError("should never be called below the support floor")


small_clusters = {"other": Cluster(key="other", doc_ids=["D1", "D2"])}
small_clusters = expand_other_with_semantic_groups(
    small_clusters, bootstrap_docs_by_id, ExplodingLLM(), min_doc_support=5)
check("'other' bucket unchanged", small_clusters["other"].doc_ids == ["D1", "D2"])

print("\n20. no 'other' bucket at all -> no-op")
no_other = {"vision_based": Cluster(key="vision_based", doc_ids=["D1"])}
result20 = expand_other_with_semantic_groups(no_other, bootstrap_docs_by_id, SemanticMockLLM())
check("clusters dict unchanged", result20 == no_other)

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
