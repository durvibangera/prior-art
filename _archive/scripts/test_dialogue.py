#!/usr/bin/env python3
"""Unit tests for the step-2 question loop (priorart/core/dialogue.py).

Covers the soft-scoring design properties the whole "tree" premise depends
on, plus real failures observed live against gemma-e4.
Run: python3 scripts/test_dialogue.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.dialogue import (
    CONFIRM_THRESHOLD, OPEN_QUESTION_SYSTEM_BABY, OPEN_QUESTION_SYSTEM_KID,
    OPEN_QUESTION_SYSTEM_TEENAGER, OPEN_QUESTION_SYSTEM_TODDLER,
    PRUNE_THRESHOLD, SCORE_CAP, SENTIMENT_DELTAS, HintResult,
    _is_decline, _is_other_bucket, answer_user_question, apply_seed_grounding,
    drill_into_cluster, drill_other_bucket, generate_open_question, route_cluster_followup,
    interpret_hint, pick_open_candidates, process_hint,
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


class FakeLLM:
    """Returns a fixed SENTIMENT digit regardless of input, to isolate
    interpret_hint's own logic from real model variance."""
    enabled = True

    def __init__(self, digit):
        self.digit = digit

    def ask(self, *a, **kw):
        return KVResult(records=[{"SENTIMENT": self.digit}], errors=[])


CLUSTER = {"cluster_id": "vision_based", "label": "vision based", "score": 0.0,
          "key_terms": ["CAMERA", "VIDEO", "OPTICAL"]}

print("\n1. pick_open_candidates ties broken by real document count")
live = [
    {"cluster_id": "a", "score": 0.0, "label": "a"},
    {"cluster_id": "b", "score": 0.0, "label": "b"},
    {"cluster_id": "c", "score": 0.0, "label": "c"},
    {"cluster_id": "d", "score": 0.0, "label": "d"},
]
top2 = pick_open_candidates(live, {"a": 5, "b": 90, "c": 40, "d": 1}, k=2)
check("best-evidenced two picked first",
     {c["cluster_id"] for c in top2} == {"b", "c"}, top2)

print("\n2. pick_open_candidates with no doc_counts still works (no crash)")
candidates = pick_open_candidates(live, k=3)
check("returns k candidates", len(candidates) == 3, candidates)

print("\n3. fewer than 2 live clusters signals convergence (caller checks len)")
check("returns 1 for a single cluster", len(pick_open_candidates(live[:1])) == 1)
check("returns 0 for no clusters", len(pick_open_candidates([])) == 0)

print("\n3b. REAL FAILURE: the default batch size crushed a 3rd live branch into a 2-way question")
# Captured live: 3 branches tied at score 0.0 (deep_raw_audio, handcrafted_
# features, other) all got bundled into ONE turn. The BABY-tier prompt is a
# strict two-way "closer to A or B?" contrast, so the model silently
# phrased the question around only 2 of them -- yet process_hint still
# scored ALL 3 against the single reply, and the 3rd (a 57-document "other"
# bucket) got PRUNED OUTRIGHT from a question it was never actually part
# of. Fixed by defaulting k to 2, not 4, so what gets scored always matches
# what was actually shown.
three_tied = [
    {"cluster_id": "deep_raw_audio", "score": 0.0, "label": "deep raw audio"},
    {"cluster_id": "handcrafted_features", "score": 0.0, "label": "handcrafted features"},
    {"cluster_id": "other", "score": 0.0, "label": "other"},
]
check("default batch size is 2, not 4 -- a 3rd tied branch is never silently bundled in",
     len(pick_open_candidates(three_tied)) == 2, pick_open_candidates(three_tied))

print("\n4. REAL FAILURE: reply is a bare digit, model echoes it back unchanged")
# Captured live: hint_text "1" against two different clusters both came back
# SENTIMENT "1" -- indistinguishable from parroting the input rather than
# judging it, since the model's answer digit exactly matched the human's
# own literal reply. Must NOT be trusted as a genuine confirm.
r = interpret_hint("1", CLUSTER, FakeLLM("1"))
check("echoed digit rejected, falls back neutral", r.sentiment_digit == "3" and r.delta == 0.0, r)
check("fallback method names the reason", "echo" in r.method, r.method)

print("\n5. a digit reply that does NOT match the model's answer is trusted")
# hint_text "42" can never equal a menu digit ("1".."5"), so a genuine "1"
# answer from the model must still be accepted normally -- grounded in the
# branch's own vocabulary (CAMERA), so the new ungrounded-confirm cap
# (test 9b) does not apply here.
r = interpret_hint("uses a camera feed, 42 frames a second", CLUSTER, FakeLLM("1"))
check("non-matching digit reply, grounded, still trusted at full delta",
     r.sentiment_digit == "1" and r.method == "llm" and r.delta == SENTIMENT_DELTAS["1"], r)

print("\n6. an ordinary word reply is never mistaken for the echo case")
r = interpret_hint("yeah maybe", CLUSTER, FakeLLM("2"))
check("ordinary reply trusted normally", r.sentiment_digit == "2" and r.method == "llm", r)

print("\n7. LLM disabled -> honest neutral fallback, never a fabricated read")
r = interpret_hint("thermal based", CLUSTER, None)
check("disabled llm gives neutral fallback", r.sentiment_digit == "3" and r.delta == 0.0 and r.method == "fallback", r)

print("\n8. score cap and confirm/prune thresholds are internally consistent")
check("cap is larger than confirm threshold", SCORE_CAP > CONFIRM_THRESHOLD)
check("prune threshold mirrors confirm threshold", PRUNE_THRESHOLD == -CONFIRM_THRESHOLD)

print("\n9. REAL FAILURE: a comparative reply ('second I guess') given no context")
# Captured live: asked "vision based (first) vs thermal based (second)",
# reply "second i guess" scored a weak POSITIVE for BOTH clusters instead of
# favoring thermal -- because interpret_hint judged each cluster in total
# isolation, with no idea a second candidate or an ordinal position existed.
# process_hint must now pass each cluster its own position and the other
# cluster's label so this is resolvable at all.
captured_prompts = []


class RecordingLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        captured_prompts.append(user)
        return KVResult(records=[{"SENTIMENT": "3"}], errors=[])


class FakeStore:
    def __init__(self, clusters):
        self._clusters = clusters

    def get_cluster(self, conversation_id, cluster_id):
        return self._clusters.get(cluster_id)

    def update_cluster_score(self, *a, **kw):
        return 0.0

    def set_cluster_status(self, *a, **kw):
        pass


vision = {"cluster_id": "vision_based", "label": "vision based", "score": 0.0, "status": "live"}
thermal = {"cluster_id": "thermal_based", "label": "thermal based", "score": 0.0, "status": "live"}
store = FakeStore({"vision_based": vision, "thermal_based": thermal})
process_hint(store, "c1", 1, "second i guess", ["vision_based", "thermal_based"], RecordingLLM())

vision_prompt, thermal_prompt = captured_prompts
check("vision's call knows it was named first", "(first)" in vision_prompt, vision_prompt)
check("vision's call is told the other option", "thermal based" in vision_prompt, vision_prompt)
check("thermal's call knows it was named second", "(second)" in thermal_prompt, thermal_prompt)
check("thermal's call is told the other option", "vision based" in thermal_prompt, thermal_prompt)

print("\n10. REAL FAILURE: 'neither' took 2-3 repeats to prune both branches")
# Captured live: a plain "neither" sometimes read as only weakly negative,
# needing the same question re-asked several times before both branches
# crossed the prune threshold. An exact "neither" (or close variant) is
# unambiguous and is now resolved in code, not left to the model's
# sometimes-unreliable reading of negation.
for hint in ("neither", "None", "none of these", "Not either", "NEITHER!"):
    r = interpret_hint(hint, CLUSTER, None)
    check(f"{hint!r} immediately crosses the prune threshold in one round",
         r.sentiment_digit == "5" and r.delta <= PRUNE_THRESHOLD and r.method == "rule:neither", r)

print("\n11. a HEDGED reply merely containing 'neither' still goes to the model")
r = interpret_hint("neither really, closer to the first", CLUSTER, None)
check("hedged reply does not trigger the structural rule", r.method != "rule:neither", r)

print("\n11a. REAL FAILURE: an ungrounded 'clearly confirms' single-handedly confirmed the wrong branch")
# Captured live against qwen3-max: a reply describing a plain CNN pipeline --
# conv layers, pooling, flatten, fully-connected classifier -- with NO
# mention of LSTM/RNN/GRU/sequence/temporal/attention anywhere, still scored
# a full "1" (clearly confirms) for the "sequence_model" branch, confirming
# it outright from that one reply alone (delta 2.0 == CONFIRM_THRESHOLD).
SEQUENCE_CLUSTER = {"cluster_id": "sequence_model", "label": "sequence model", "score": 0.0,
                    "key_terms": ["LSTM", "RNN", "GRU", "SEQUENCE", "TEMPORAL", "TRANSFORMER", "ATTENTION"]}
cnn_only_reply = ("processes the waveform through 1D convolutional layers with ReLU, "
                  "dropout and max pooling, then flattens into fully connected layers "
                  "for the final classification")
r = interpret_hint(cnn_only_reply, SEQUENCE_CLUSTER, FakeLLM("1"))
check("ungrounded 'clearly confirms' is capped, not trusted at full delta",
     r.sentiment_digit == "1" and r.delta == SENTIMENT_DELTAS["2"], r)
check("capped result cannot alone cross the confirm threshold",
     r.delta < CONFIRM_THRESHOLD, r)
check("method names it as an ungrounded, capped confirm", "ungrounded" in r.method and "capped" in r.method, r.method)

print("\n11b. the SAME branch, genuinely grounded, is trusted at full delta")
lstm_reply = "after the convolutional stage, an LSTM layer models the sequence over time before classification"
r = interpret_hint(lstm_reply, SEQUENCE_CLUSTER, FakeLLM("1"))
check("grounded 'clearly confirms' is trusted at full delta",
     r.sentiment_digit == "1" and r.delta == SENTIMENT_DELTAS["1"] and r.method == "llm", r)

print("\n11c. the cap only applies to '1' (confirm) -- '5' (rule out) is trusted ungrounded")
r = interpret_hint("no, that's not what I built at all", SEQUENCE_CLUSTER, FakeLLM("5"))
check("ungrounded rule-out is still trusted at full delta, not capped",
     r.sentiment_digit == "5" and r.delta == SENTIMENT_DELTAS["5"] and r.method == "llm", r)


class DocStore:
    """Serves real-ish document snippets per cluster, for testing that the
    open question is actually grounded in document text, not just labels."""
    def __init__(self, docs_by_cluster):
        self._docs = docs_by_cluster

    def docs_for_cluster(self, conversation_id, cluster_id):
        return self._docs.get(cluster_id, [])


VISION_C = {"cluster_id": "vision_based", "label": "vision based"}
THERMAL_C = {"cluster_id": "thermal_based", "label": "thermal based"}

print("\n12. generate_open_question: no LLM -> flat, CLOSED fallback naming real candidates")
# Requested directly: "ask questions like this or that... i want it to be a
# glorified version of that" -- even the offline fallback must stay closed
# ("closer to X or Y"), never an open "describe your approach" invitation.
doc_store = DocStore({})
question, method = generate_open_question(doc_store, "c1", [VISION_C, THERMAL_C], None)
check("method is template when llm unavailable", method == "template", method)
check("fallback names the real candidates", "vision based" in question and "thermal based" in question, question)
check("fallback IS a closed pick", "closer to" in question.lower(), question)

print("\n13. generate_open_question (KID/TEENAGER tiers) grounds the prompt in real document snippets")
captured = []


class RecordingQuestionLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        captured.append(user)
        return KVResult(records=[{"QUESTION": "Does yours do something like this, or is it different?"}], errors=[])


doc_store = DocStore({
    "vision_based": [{"title": "A CAMERA BASED DRONE DETECTOR", "abstract": "uses object detection on video frames"}],
    "thermal_based": [{"title": "AN INFRARED UAV SPOTTER", "abstract": "uses a thermal camera array"}],
})
# Not the first question, and both candidates are drilled children (one
# level deep) -- lands on the KID tier, which is document-grounded.
kid_candidates = [{"cluster_id": "vision_based", "label": "vision based", "parent_id": "some_parent"},
                  {"cluster_id": "thermal_based", "label": "thermal based", "parent_id": "some_parent"}]
question, method = generate_open_question(doc_store, "c1", kid_candidates, RecordingQuestionLLM())
check("llm path used", method == "llm", method)
prompt = captured[0]
check("prompt includes a real snippet from vision_based", "CAMERA BASED DRONE DETECTOR" in prompt, prompt)
check("prompt includes a real snippet from thermal_based", "INFRARED UAV SPOTTER" in prompt, prompt)

print("\n14. generate_open_question falls back honestly when no real documents exist yet")
doc_store = DocStore({})
question, method = generate_open_question(doc_store, "c1", kid_candidates, RecordingQuestionLLM())
check("no snippets -> falls back rather than asking the LLM to invent detail",
     method == "template", (question, method))

print("\n14b. REAL FAILURE: the first question jumped straight to implementation detail")
# Named directly: "way too complex... where is the foreplay why straight to
# home run", and later: "the first question the llm asked was 'give me high
# level overview of your general approach' which made me answer more than i
# had to". The VERY FIRST question of the whole conversation must now be
# BABY tier regardless of anything else -- category labels only, no
# document excerpts, no room for a free-text paragraph answer.
captured_systems = []


class SystemRecordingLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        captured_systems.append(system)
        return KVResult(records=[{"QUESTION": "Closer to A or B?"}], errors=[])


doc_store = DocStore({
    "vision_based": [{"title": "A CAMERA BASED DRONE DETECTOR", "abstract": "uses object detection"}],
    "thermal_based": [{"title": "AN INFRARED UAV SPOTTER", "abstract": "uses a thermal camera array"}],
})
top_level_candidates = [{"cluster_id": "vision_based", "label": "vision based", "parent_id": None},
                        {"cluster_id": "thermal_based", "label": "thermal based", "parent_id": None}]
generate_open_question(doc_store, "c1", top_level_candidates, SystemRecordingLLM(), is_first_question=True)
check("the very first question of the conversation gets the BABY prompt",
     captured_systems[-1] == OPEN_QUESTION_SYSTEM_BABY, captured_systems[-1][:60])
check("BABY tier never even reads document excerpts",
     "CAMERA BASED DRONE DETECTOR" not in captured_systems[-1], captured_systems[-1])

print("\n14c. after the first question, tier tracks how deep the candidates are drilled")
generate_open_question(doc_store, "c1", top_level_candidates, SystemRecordingLLM(), is_first_question=False)
check("top-level candidates (not first question) get TODDLER",
     captured_systems[-1] == OPEN_QUESTION_SYSTEM_TODDLER, captured_systems[-1][:60])

doc_store2 = DocStore({
    "deep_raw_audio__xgboost": [{"title": "A", "abstract": "uses XGBOOST on extracted features"}],
    "deep_raw_audio__mobilenet": [{"title": "B", "abstract": "uses MOBILENET for classification"}],
})
drilled_candidates = [{"cluster_id": "deep_raw_audio__xgboost", "label": "xgboost", "parent_id": "deep_raw_audio"},
                      {"cluster_id": "deep_raw_audio__mobilenet", "label": "mobilenet", "parent_id": "deep_raw_audio"}]
generate_open_question(doc_store2, "c1", drilled_candidates, SystemRecordingLLM(), is_first_question=False)
check("one level deep gets KID", captured_systems[-1] == OPEN_QUESTION_SYSTEM_KID, captured_systems[-1][:60])

doc_store3 = DocStore({
    "deep_raw_audio__other__spectral": [{"title": "C", "abstract": "uses SPECTRAL features"}],
})
deep_candidates = [{"cluster_id": "deep_raw_audio__other__spectral", "label": "spectral",
                    "parent_id": "deep_raw_audio__other"}]
generate_open_question(doc_store3, "c1", deep_candidates, SystemRecordingLLM(), is_first_question=False)
check("two levels deep gets TEENAGER",
     captured_systems[-1] == OPEN_QUESTION_SYSTEM_TEENAGER, captured_systems[-1][:60])

print("\n14d. every fallback stays a closed pick, at every tier")
question, method = generate_open_question(DocStore({}), "c1", top_level_candidates, None)
check("fallback IS a closed pick even with no LLM", "closer to" in question.lower(), question)

def doc(appno, title="", abstract=""):
    return {"application_number": appno, "title": title, "abstract": abstract}


print("\n15. REAL FAILURE: a real facet that missed the size floor by one document")
# Captured live: 4 real documents genuinely matched "handcrafted_features"
# (Mel/STFT spectrogram-style preprocessing) inside the "other" bucket, but
# MIN_CLUSTER_SIZE=5 meant it never became its own branch -- so a detailed,
# correct answer describing exactly that had nothing to be recognized
# under, and the conversation kept re-asking irrelevant candidates.


class DrillFakeStore:
    def __init__(self, docs, cluster_id="other", top_level=None):
        self._docs = docs
        self._cluster_id = cluster_id
        self._top_level = top_level or []
        self.added_clusters = {}
        self.linked = {}
        self.statuses = {}

    def docs_for_cluster(self, conversation_id, cluster_id):
        return self._docs if cluster_id == self._cluster_id else []

    def add_cluster(self, conversation_id, cluster_id, label, key_terms, parent_id=None):
        self.added_clusters[cluster_id] = {"label": label, "parent_id": parent_id}

    def link_docs_to_cluster(self, conversation_id, cluster_id, doc_ids, source=""):
        self.linked[cluster_id] = doc_ids

    def get_cluster(self, conversation_id, cluster_id):
        if cluster_id in self.added_clusters:
            info = self.added_clusters[cluster_id]
            return {"cluster_id": cluster_id, "label": info["label"],
                   "parent_id": info["parent_id"], "key_terms": []}
        return None

    def get_clusters(self, conversation_id, parent_id="__any__", status=None):
        if parent_id is None:
            return list(self._top_level)
        return list(self._top_level)

    def set_cluster_status(self, conversation_id, cluster_id, status):
        self.statuses[cluster_id] = status


docs = ([doc(f"H{i}", abstract="uses MFCC and SPECTROGRAM features for classification") for i in range(4)] +
       [doc(f"X{i}", abstract="a completely unrelated smart water bottle invention") for i in range(3)])
other_cluster = {"cluster_id": "other", "label": "other"}
fake_store = DrillFakeStore(docs)
pass1_children, pass2_children = drill_other_bucket(fake_store, "c1", other_cluster, None)
check("handcrafted_features recovered from other", any("handcrafted_features" in cid for cid in fake_store.added_clusters),
     fake_store.added_clusters)
check("recovered via pass 1 (word-frequency), not pass 2", pass1_children and not pass2_children,
     (pass1_children, pass2_children))
check("recovered as a CHILD of other (parent_id set)",
     all(info["parent_id"] == "other" for info in fake_store.added_clusters.values()), fake_store.added_clusters)

print("\n16. drill_other_bucket recovers nothing when there's genuinely no pattern")
docs = [doc(f"X{i}", abstract="a completely unrelated invention with no shared vocabulary") for i in range(6)]
fake_store2 = DrillFakeStore(docs)
children2 = drill_other_bucket(fake_store2, "c1", other_cluster, None)
check("nothing recovered, no fake branch forced", children2 == ([], []) and not fake_store2.added_clusters, children2)

print("\n16a. drill_other_bucket's SECOND pass finds a real semantic group word-frequency misses")
# Verified live: 4 real documents across completely unrelated applications
# (pest control, noise pollution, pipeline leaks, surveillance) share the
# same underlying acoustic-sensing approach but no recurring keyword --
# word-frequency discovery alone would never find this.
semantic_test_docs = [
    doc("A1", title="PEST SCOUTING", abstract="drone detects pests through acoustic signatures"),
    doc("A2", title="NOISE DETECTION", abstract="AI system detects noise pollution in cities"),
    doc("A3", title="PIPELINE LEAK", abstract="detects underground water pipeline leakage using sound"),
    doc("A4", title="PATTERN MATCH", abstract="real time pattern matching for surveillance"),
    doc("X1", title="WATER BOTTLE", abstract="a smart water bottle for hydration tracking"),
]


class TwoPassMockLLM:
    """Spec-aware: label_cluster (pass 1) and discover_semantic_groups
    (pass 2) use different response shapes -- must respond correctly to
    each, not a single hardcoded shortcut. Tests the WIRING (does
    drill_other_bucket correctly turn a returned group into a real child
    cluster, excluding the unrelated document) -- the LLM's own judgment
    quality is separately tested in test_clusters.py."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        field_names = {f.name for f in spec.fields}
        if field_names == {"GROUP", "DOCS"}:
            return KVResult(records=[{"GROUP": "Acoustic signal detection and analysis",
                                     "DOCS": "1 2 3 4"}], errors=[])
        return KVResult(records=[], errors=["no label needed for this test"])


semantic_store = DrillFakeStore(semantic_test_docs)
semantic_pass1, semantic_pass2 = drill_other_bucket(semantic_store, "c1", other_cluster, TwoPassMockLLM())
check("a real semantic group was created as a child",
     any("acoustic_signal" in c["cluster_id"] for c in semantic_pass2), semantic_pass2)
check("found via pass 2 (semantic), not pass 1", semantic_pass2 and not semantic_pass1,
     (semantic_pass1, semantic_pass2))
check("the unrelated water-bottle document was correctly left out",
     "X1" not in semantic_store.linked.get(
         next((cid for cid in semantic_store.added_clusters if "acoustic_signal" in cid), ""), []),
     semantic_store.linked)

print("\n16b. REAL FAILURE: drilling a PROCESSING-METHOD branch used to force sensor-modality vocabulary")
# Captured live: 'deep_raw_audio' (a processing-method branch) got drilled
# by a FIXED sensor-modality vocabulary into "vision based"/"thermal based+
# vision based" even though seed grounding had ALREADY pruned every
# sensor-alternative branch -- reintroducing a dimension that was already
# closed, and the resulting question dragged "multi-modal fusion" back into
# an acoustic-only conversation, making a detailed answer look ignored.
# Fixed by discovering whatever ACTUALLY recurs in the branch's own real
# documents instead of forcing a fixed lens -- verified here that camera/
# thermal words are never rediscovered (they're already part of the
# existing top-level vocabulary, so discovery correctly excludes them, with
# no special-case guard needed at all).
audio_docs = ([doc(f"V{i}", abstract="a CAMERA based drone detector using VIDEO") for i in range(6)] +
             [doc(f"T{i}", abstract="uses THERMAL infrared imaging") for i in range(6)])
audio_cluster = {"cluster_id": "deep_raw_audio", "label": "deep raw audio"}
no_llm_store = DrillFakeStore(audio_docs, cluster_id="deep_raw_audio")
children_no_llm = drill_into_cluster(no_llm_store, "c1", audio_cluster, None)
check("no LLM available -> nothing discovered, no fake children forced",
     children_no_llm == [] and not no_llm_store.added_clusters, children_no_llm)


class ArchitectureGateLLM:
    """Says TECHNICAL for the real model-architecture words the user
    actually described (XGBOOST, MOBILENET), GENERIC for everything else --
    exercises the real validation path, not a hardcoded errors=[] shortcut.
    Also has to handle label_cluster's OWN (differently-shaped) call, since
    drill_into_cluster labels each discovered child too."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        from priorart.core.llm import KVResult
        field_names = {f.name for f in spec.fields}
        if field_names == {"ANSWER"}:
            answer = "1" if ("XGBOOST" in user or "MOBILENET" in user) else "2"
            validate = spec.fields[0].validate
            err = validate(answer) if validate else None
            if err:
                return KVResult(records=[], errors=[f"ANSWER {err}"])
            return KVResult(records=[{"ANSWER": answer}], errors=[])
        # label_cluster's own call (LABEL + KEY_TERMS) -- respond honestly
        # rather than crash; a made-up label is fine here, the test only
        # cares about which documents ended up in which child cluster.
        return KVResult(records=[{"LABEL": "discovered sub-approach", "KEY_TERMS": "XGBOOST"}], errors=[])


print("\n16c. drill_into_cluster discovers REAL new sub-patterns instead of forcing sensor modality")
architecture_docs = ([doc(f"X{i}", abstract="an XGBOOST classifier on extracted audio features") for i in range(4)] +
                     [doc(f"M{i}", abstract="a MOBILENET based classifier for audio spectrograms") for i in range(4)])
arch_store = DrillFakeStore(architecture_docs, cluster_id="deep_raw_audio")
children_arch = drill_into_cluster(arch_store, "c1", audio_cluster, ArchitectureGateLLM())
check("discovers real model-architecture sub-patterns, not sensor modality",
     len(children_arch) > 0 and not any("vision" in c["cluster_id"] or "thermal" in c["cluster_id"]
                                        for c in children_arch), children_arch)

print("\n16d. REAL FAILURE: recovering a facet that already exists at the top level")
# Captured live: "other" recovered "deep_raw_audio" as a child even though a
# REAL top-level "deep_raw_audio" branch already existed -- two branches
# both displayed as "deep raw audio" with nothing to tell them apart.
mixed_docs = ([doc(f"D{i}", abstract="a CNN raw WAVEFORM deep NEURAL model") for i in range(5)] +
             [doc(f"H{i}", abstract="uses MFCC and SPECTROGRAM features") for i in range(4)])
existing_deep_audio = [{"cluster_id": "deep_raw_audio", "status": "confirmed"}]
merge_store = DrillFakeStore(mixed_docs, cluster_id="other", top_level=existing_deep_audio)
merge_pass1, merge_pass2 = drill_other_bucket(merge_store, "c1", other_cluster, None)
check("no duplicate 'other__deep_raw_audio' child created",
     not any("deep_raw_audio" in cid for cid in merge_store.added_clusters), merge_store.added_clusters)
check("recovered docs merged into the EXISTING top-level branch instead",
     "deep_raw_audio" in merge_store.linked, merge_store.linked)
check("the genuinely new facet (handcrafted_features) still gets its own child",
     any("handcrafted_features" in cid for cid in merge_store.added_clusters), merge_store.added_clusters)

print("\n17. REAL FAILURE: the seed already says 'audio... microphones' but nothing used that")
# Captured live: a seed clearly establishing acoustic sensing didn't stop
# the conversation from asking "vision or thermal" as if starting from
# zero -- the same public-safe text already used to build the real portal
# search should settle sensor-modality branches it clearly speaks to.


class SeedGroundingFakeStore:
    def __init__(self, clusters):
        self._clusters = {c["cluster_id"]: dict(c) for c in clusters}

    def get_conversation(self, conversation_id):
        return {"conversation_id": conversation_id, "site": "inpass"}

    def get_clusters(self, conversation_id, parent_id="__any__", status=None):
        return list(self._clusters.values())

    def get_cluster(self, conversation_id, cluster_id):
        return self._clusters.get(cluster_id)

    def update_cluster_score(self, conversation_id, cluster_id, delta, cap=5.0, turn_no=None):
        c = self._clusters[cluster_id]
        c["score"] = max(-cap, min(cap, c["score"] + delta))
        return c["score"]

    def set_cluster_status(self, conversation_id, cluster_id, status):
        self._clusters[cluster_id]["status"] = status

    def docs_for_cluster(self, conversation_id, cluster_id):
        return []

    def get_pending_searches(self, conversation_id, state="pending"):
        return []

    def add_pending_search(self, *a, **kw):
        pass


def branch(cid, label, score=0.0, status="live"):
    return {"cluster_id": cid, "label": label, "score": score, "status": status, "key_terms": []}


ACOUSTIC_SEED = ("Application for C-UAS. ML training on extensive audio dataset, "
                "highly sensitive microphones, robust detection at low SNRs.")

seed_store = SeedGroundingFakeStore([
    branch("vision_based", "vision based"), branch("thermal_based", "thermal based"),
    branch("radar_based", "radar based"), branch("rf_based", "rf based"),
    branch("multi_sensor_fusion", "multi sensor fusion"),
    branch("deep_raw_audio", "deep raw audio"), branch("sequence_model", "sequence model"),
])
touched = apply_seed_grounding(seed_store, "c1", ACOUSTIC_SEED, None)
check("all 5 sensor-alternative branches touched", set(touched) ==
     {"vision_based", "thermal_based", "radar_based", "rf_based", "multi_sensor_fusion"}, touched)
for cid in ("vision_based", "thermal_based", "radar_based", "rf_based", "multi_sensor_fusion"):
    check(f"{cid} pruned from acoustic-only seed", seed_store.get_cluster("c1", cid)["status"] == "pruned",
         seed_store.get_cluster("c1", cid))
check("deep_raw_audio (processing method) left untouched",
     seed_store.get_cluster("c1", "deep_raw_audio")["status"] == "live")
check("sequence_model (processing method) left untouched",
     seed_store.get_cluster("c1", "sequence_model")["status"] == "live")

print("\n18. a seed that DIRECTLY names an alternative sensor confirms that branch")
CAMERA_SEED = "A camera-based drone detection system using video object recognition."
seed_store2 = SeedGroundingFakeStore([
    branch("vision_based", "vision based"), branch("thermal_based", "thermal based"),
])
touched2 = apply_seed_grounding(seed_store2, "c1", CAMERA_SEED, None)
check("vision_based confirmed directly from the seed",
     seed_store2.get_cluster("c1", "vision_based")["status"] == "confirmed", touched2)

print("\n19. a seed silent about sensing modality touches nothing (never a guess)")
VAGUE_SEED = "A system and method for improving detection reliability in adverse conditions."
seed_store3 = SeedGroundingFakeStore([branch("vision_based", "vision based")])
touched3 = apply_seed_grounding(seed_store3, "c1", VAGUE_SEED, None)
check("nothing touched for a seed that says nothing about modality", touched3 == [], touched3)

print("\n20. a seed naming one modality alongside audio still prunes the others independently")
MIXED_SEED = "A system combining audio microphones and a camera for drone detection."
seed_store4 = SeedGroundingFakeStore([
    branch("vision_based", "vision based"), branch("thermal_based", "thermal based"),
])
touched4 = apply_seed_grounding(seed_store4, "c1", MIXED_SEED, None)
check("vision_based confirmed (directly named)",
     seed_store4.get_cluster("c1", "vision_based")["status"] == "confirmed")
check("thermal_based still pruned (acoustic established, thermal never mentioned)",
     seed_store4.get_cluster("c1", "thermal_based")["status"] == "pruned")

print("\n21. REAL FAILURE: the last remaining live branch never got asked about")
# Captured live: once everything else confirmed or was pruned, only
# 'sequence_model' remained live -- but the loop required at least 2 live
# candidates to keep going, so it declared convergence and quit without
# ever asking about it, even though it was genuinely still relevant.
last_branch = [{"cluster_id": "sequence_model", "label": "sequence model", "score": 0.0}]
solo_candidates = pick_open_candidates(last_branch, {})
check("a single remaining live branch is still returned, not treated as empty",
     len(solo_candidates) == 1, solo_candidates)

check("zero remaining live branches correctly signals convergence",
     pick_open_candidates([], {}) == [])

print("\n22. REAL FAILURE: _is_other_bucket must catch nested other buckets, not just top-level")
# Captured live: only the literal top-level "other" was special-cased
# before, so a NESTED other bucket ("deep_raw_audio__other") got normally
# confirmed and queued a real search using the literal word "OTHER" as
# search vocabulary, which the real InPASS portal repeatedly choked on.
check("top-level 'other' detected", _is_other_bucket("other") is True)
check("nested 'parent__other' detected", _is_other_bucket("deep_raw_audio__other") is True)
check("doubly-nested 'a__b__other' detected", _is_other_bucket("a__b__other") is True)
check("a real branch is NOT mistaken for an other bucket", _is_other_bucket("deep_raw_audio") is False)
check("a branch that merely CONTAINS 'other' as a substring is not mistaken either",
     _is_other_bucket("another_branch") is False)

print("\n23. route_cluster_followup sends a nested 'other' bucket to drilling, not confirming")
nested_other_docs = [doc(f"H{i}", abstract="uses MFCC and SPECTROGRAM features") for i in range(4)]
nested_other_cluster = {"cluster_id": "deep_raw_audio__other", "label": "other", "status": "confirmed", "score": 2.0}
route_store = DrillFakeStore(nested_other_docs, cluster_id="deep_raw_audio__other")
route_cluster_followup(route_store, "c1", nested_other_cluster, None)
check("no targeted-search-triggering path was taken (would need get_pending_searches, which this fake lacks)",
     True)  # reaching this line at all (no AttributeError) proves the confirm path was never entered
check("it was drilled instead -- status set to 'drilled', not left 'confirmed'",
     route_store.statuses.get("deep_raw_audio__other") == "drilled", route_store.statuses)

print("\n24. _is_decline recognizes an exact decline, never a hedge that merely mentions one")
# Requested directly: "if i wish to not answer a question i will just say i
# dont want to and the llm should understand that it cant help it and has
# to move on" -- same EXACT-match discipline as _is_unambiguous_neither.
for phrase in ("pass", "Pass", "skip that", "I don't want to answer",
              "i dont want to say that", "no comment", "I'd rather not say"):
    check(f"{phrase!r} recognized as a decline", _is_decline(phrase), phrase)
for phrase in ("I'll pass on the details but it's audio-based",
              "not sure, skip ahead a bit and I'll explain",
              "no comment on the exact number but yes it's a CNN"):
    check(f"{phrase!r} (hedged, merely CONTAINS a decline word) is NOT a decline",
         not _is_decline(phrase), phrase)

print("\n24b. REAL FAILURE: 'idk'/'not sure' are uncertain non-answers, not confidentiality refusals,")
print("     but need the SAME handling -- otherwise the one remaining branch gets re-asked forever")
# Captured live: "maybe" then "idk" to the SAME question both scored neutral
# (sentiment 3) via the LLM, which still counts as "asked and answered" --
# with nothing else live to rotate to, the identical question got asked a
# third time. Treating "idk" as a decline gives the conversation somewhere
# to actually go instead.
for phrase in ("idk", "IDK", "i dont know", "I don't know", "dont know",
              "not sure", "unsure", "no idea"):
    check(f"{phrase!r} recognized as a decline-equivalent", _is_decline(phrase), phrase)
for phrase in ("not sure, skip ahead a bit and I'll explain",
              "I don't know the exact number but it's a CNN"):
    check(f"{phrase!r} (hedged, merely CONTAINS 'not sure'/'dont know') is NOT a decline",
         not _is_decline(phrase), phrase)

print("\n25. pick_open_candidates deprioritizes recently-declined clusters, but never stalls entirely")
live4 = [
    {"cluster_id": "a", "score": 1.0, "label": "a"},
    {"cluster_id": "b", "score": 0.5, "label": "b"},
    {"cluster_id": "c", "score": 0.0, "label": "c"},
]
picked = pick_open_candidates(live4, k=2, exclude_ids={"a"})
check("declined 'a' skipped in favor of the next-best real candidates",
     {c["cluster_id"] for c in picked} == {"b", "c"}, picked)

picked_all_declined = pick_open_candidates(live4, k=2, exclude_ids={"a", "b", "c"})
check("if EVERY live candidate was recently declined, they're used anyway rather than stalling",
     len(picked_all_declined) == 2, picked_all_declined)


class DeclineFakeStore:
    def __init__(self, turns):
        self._turns = turns

    def get_turns(self, conversation_id):
        return self._turns


print("\n26. _recently_declined_cluster_ids reads the turns log, not a new schema column")
from priorart.core.dialogue import _recently_declined_cluster_ids

decline_turns = [
    {"kind": "question", "cluster_refs": ["a", "b"]},
    {"kind": "decline", "cluster_refs": ["a", "b"]},
    {"kind": "question", "cluster_refs": ["c"]},
    {"kind": "hint", "cluster_refs": ["c"]},
]
declined = _recently_declined_cluster_ids(DeclineFakeStore(decline_turns), "c1", within_last_n_turns=3)
check("declined cluster_ids recovered from the turns log", declined == {"a", "b"}, declined)
declined_far_back = _recently_declined_cluster_ids(DeclineFakeStore(decline_turns), "c1", within_last_n_turns=1)
check("outside the recency window, no longer flagged", declined_far_back == set(), declined_far_back)

print("\n26b. REAL FAILURE: a ONE-question memory oscillates between just two sets forever")
# Captured live, directly from the real stored turns log: turn 5 asked set A
# (handcrafted_features + 3 "other" children), turn 6 correctly asked a
# DIFFERENT set B -- but by turn 11, A was no longer "the last one asked" (B
# was), so A -- being the higher-scoring set -- won the pick again. Verbatim
# A -> B -> A, perceived by the person as "it keeps asking me the same
# question" even though a one-turn exclusion was technically firing each
# time. Fixed by widening the memory to the last N question turns, not just 1.
from priorart.core.dialogue import _recently_asked_cluster_ids

SET_A = {"handcrafted_features", "other__acoustic", "other__voice", "other__airborne"}
SET_B = {"other__sound_separation", "other__radar", "other__aerial_payload", "other__fusion"}

oscillating_turns = [
    {"kind": "question", "cluster_refs": list(SET_A)},   # turn 5
    {"kind": "question", "cluster_refs": list(SET_B)},   # turn 6
    {"kind": "hint", "cluster_refs": list(SET_B)},
]
asked_1window = _recently_asked_cluster_ids(DeclineFakeStore(oscillating_turns), "c1",
                                            within_last_n_questions=1)
check("a 1-question memory only remembers set B -- set A would be re-askable",
     asked_1window == SET_B, asked_1window)

asked_3window = _recently_asked_cluster_ids(DeclineFakeStore(oscillating_turns), "c1",
                                            within_last_n_questions=3)
check("a wider memory remembers BOTH recent sets -- set A stays excluded too",
     asked_3window == SET_A | SET_B, asked_3window)

no_questions_yet = _recently_asked_cluster_ids(DeclineFakeStore([]), "c1")
check("no questions asked yet -> empty, nothing to exclude", no_questions_yet == set(), no_questions_yet)

print("\n26c. end-to-end: with the wider memory, set A genuinely cannot win back-to-back-to-back")
live_many = [
    {"cluster_id": "handcrafted_features", "score": 1.4, "label": "handcrafted features"},
    {"cluster_id": "other__acoustic", "score": 0.7, "label": "a"},
    {"cluster_id": "other__voice", "score": 0.7, "label": "b"},
    {"cluster_id": "other__airborne", "score": 0.7, "label": "c"},
    {"cluster_id": "other__sound_separation", "score": 0.0, "label": "d"},
    {"cluster_id": "other__radar", "score": 0.0, "label": "e"},
    {"cluster_id": "other__aerial_payload", "score": 0.0, "label": "f"},
    {"cluster_id": "other__fusion", "score": 0.0, "label": "g"},
    {"cluster_id": "other__untouched_1", "score": 0.0, "label": "h"},
    {"cluster_id": "other__untouched_2", "score": 0.0, "label": "i"},
]
next_candidates = pick_open_candidates(live_many, exclude_ids=asked_3window)
check("set A AND set B both excluded -- genuinely untouched branches picked instead",
     {c["cluster_id"] for c in next_candidates}.isdisjoint(SET_A | SET_B), next_candidates)

print("\n26d. REAL FAILURE: exactly one live candidate left, already asked recently -- the stall signal")
# Captured live, directly from the real stored turns log: once every OTHER
# branch was pruned or already asked about in the last 3 questions,
# "other__uav_ambient_threat_detection_for_ppe" was the ONLY option left.
# pick_open_candidates's own "never return nothing" fallback re-selected it
# anyway -- this is the exact condition run_dialogue_loop now checks for
# (candidates is a non-empty SUBSET of what was just asked) and stops on,
# instead of silently repeating the identical question forever.
single_live = [{"cluster_id": "other__uav_ambient_threat_detection_for_ppe", "score": 0.0, "label": "x"}]
asked_recently_single = {"other__uav_ambient_threat_detection_for_ppe"}
stalled_candidates = pick_open_candidates(single_live, exclude_ids=asked_recently_single)
check("the fallback still returns the only candidate (never truly empty)",
     len(stalled_candidates) == 1, stalled_candidates)
check("but it IS a subset of what was just asked -- this is the stall signal",
     {c["cluster_id"] for c in stalled_candidates} <= asked_recently_single, stalled_candidates)

genuinely_new = pick_open_candidates(live_many, exclude_ids=asked_3window)
check("when something genuinely new IS available, it's NOT a subset of what was just asked",
     not ({c["cluster_id"] for c in genuinely_new} <= asked_3window), genuinely_new)

print("\n27. answer_user_question: two-way -- the person asking US back")
# Requested directly: "i should be able to ask u questions too... like
# lowkey two way communication". Deterministic, LLM-free: real word overlap
# against every document this conversation has actually downloaded.
qa_docs = [
    doc("Q1", title="MICRO-DOPPLER RADAR TARGET CLASSIFIER", abstract="classifies drones using radar doppler signatures"),
    doc("Q2", title="SMART IRRIGATION SYSTEM", abstract="waters crops based on soil moisture sensors"),
    doc("Q3", title="ACOUSTIC WAVELET DRONE DETECTOR", abstract="uses a wavelet transform on microphone audio"),
]


class QAFakeStore:
    def __init__(self, docs):
        self._docs = docs

    def all_conversation_docs(self, conversation_id):
        return self._docs


qa_store = QAFakeStore(qa_docs)
matches = answer_user_question(qa_store, "c1", "has any paper used radar or doppler for this?")
check("real overlapping document found", any(d["application_number"] == "Q1" for d in matches), matches)
check("unrelated irrigation document not matched", not any(d["application_number"] == "Q2" for d in matches), matches)

no_match = answer_user_question(qa_store, "c1", "does anything here use blockchain for supply chain tracking?")
check("no real overlap -> empty, never a guess", no_match == [], no_match)

generic_question = answer_user_question(qa_store, "c1", "what does this paper do?")
check("a question made only of generic/stopword-like terms matches nothing forced",
     generic_question == [], generic_question)

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
