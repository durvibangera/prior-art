#!/usr/bin/env python3
"""Unit tests for priorart/core/bootstrap.py's derive_search_rows -- the
broad InPASS search query must come from the seed's own text, but can only
EXPAND the three safe domain-anchor rows, never replace or drop one.
Run: python3 scripts/test_bootstrap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.bootstrap import (
    _ANCHOR_ROWS, _seed_candidate_words, _seed_words_including_domain,
    _words_grounded_in_seed, derive_search_rows, derive_search_rows_fully_dynamic,
    derive_search_rows_ppubs, prepare_search_ppubs, queue_targeted_search,
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


SEED = ("Application for C-UAS. Framework - Combination of hardware with software for "
       "effective, low-cost detection solution against UAS threats. ML training on "
       "extensive audio dataset, robust detection at low SNRs, highly sensitive "
       "microphones, detection independent of type of drone")


class FixedLLM:
    enabled = True

    def __init__(self, words_by_concept=None, errors=None):
        self._words_by_concept = words_by_concept or {}
        self._errors = errors or []

    def ask(self, system, user, spec, **kw):
        if self._errors:
            return KVResult(records=[], errors=self._errors)
        for concept, words in self._words_by_concept.items():
            if concept in user:
                return KVResult(records=[{"WORDS": words}], errors=[])
        return KVResult(records=[], errors=["no match"])


print("\n1. no LLM available -> the 3 anchor rows returned completely unexpanded")
rows = derive_search_rows(SEED, None)
check("3 anchor rows returned", len(rows) == 3, rows)
check("drone anchor present", any("DRONE" in r.text for r in rows), rows)
check("acoustic anchor present", any("ACOUSTIC" in r.text for r in rows), rows)
check("detection anchor present", any("DETECTION" in r.text for r in rows), rows)

print("\n2. REAL FAILURE: the model must never be able to drop a domain anchor")
# Captured live: an earlier design let the model invent whole new concept
# groups instead of expanding the 3 fixed ones. It produced rows with NO
# mention of "DRONE" or "UAS" at all despite the seed saying "drone"
# explicitly -- since all rows AND together, the query lost its domain gate
# and returned 870 hits (mostly unrelated) instead of ~100. Expansion-only
# design means there is no code path where an anchor word can be dropped.
llm = FixedLLM(words_by_concept={
    "acoustic/audio sensing": "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE SENSITIVE",
    "drone/UAS": "DRONE DRONES UAV UAVS UAS QUADCOPTER THREATS",
    "detection": "DETECTION DETECTING DETECT CLASSIFICATION LATENCY",
})
rows = derive_search_rows(SEED, llm)
check("3 rows still returned", len(rows) == 3, rows)
check("drone anchor cannot be dropped", any("DRONE" in r.text for r in rows), rows)
check("acoustic anchor cannot be dropped", any("ACOUSTIC" in r.text for r in rows), rows)
check("detection anchor cannot be dropped", any("DETECTION" in r.text for r in rows), rows)
check("real seed vocabulary still gets added", any("SENSITIVE" in r.text for r in rows), rows)

print("\n3. REAL FAILURE: patent-filing boilerplate ('complete specification') must not be added")
# Captured live: "COMPLETE" and "SPECIFICATION" (from the standard Indian
# "Complete Specification" filing heading) were treated as real seed
# vocabulary and became noise -- now excluded the same way clusters.py
# excludes them from facet discovery.
seed_with_boilerplate = SEED + ". This complete specification describes conditions of use."
seed_words = _seed_candidate_words(seed_with_boilerplate)
check("COMPLETE excluded from candidate words", "COMPLETE" not in seed_words, seed_words)
check("SPECIFICATION excluded from candidate words", "SPECIFICATION" not in seed_words, seed_words)
check("CONDITIONS excluded from candidate words", "CONDITIONS" not in seed_words, seed_words)

print("\n4. a genuinely ungrounded/hallucinated expansion falls back to the unexpanded anchor")
llm = FixedLLM(errors=["WORDS none of ['WAVELET', 'SCATTERING'] trace back to a real word in the seed text"])
rows = derive_search_rows(SEED, llm)
check("falls back to unexpanded anchors, not the hallucination",
     all("WAVELET" not in r.text for r in rows) and len(rows) == 3, rows)

print("\n5. too-short seed still returns the safe anchor rows, not an error")
rows = derive_search_rows("acoustic drone", FixedLLM())
check("short seed returns 3 anchor rows", len(rows) == 3, rows)

print("\n6. _words_grounded_in_seed validator directly")
# seed_words deliberately excludes domain-baseline words (DETECTION,
# MICROPHONE, DRONE, ...) since those already live in the anchor rows
# themselves -- _expand_anchor_row unions seed_words with the anchor's own
# starting words before validating, so this checks a genuinely NEW addition.
seed_words = _seed_candidate_words(SEED)
validator = _words_grounded_in_seed(seed_words)
check("real new seed vocabulary passes", validator("SENSITIVE ROBUST") is None, seed_words)
check("unrelated words rejected", validator("WAVELET SCATTERING TRANSFORM") is not None)

print("\n6b. REAL FAILURE: a valid word RIDING ALONG an ungrounded one must reject the whole entry")
# Captured live: "ML" (never a real grounded word -- too short to even be a
# candidate) rode along next to "TRAINING" (genuinely grounded) because the
# old validator only required ONE word in the list to ground. Every word
# must now ground individually.
check("ungrounded word can't ride along with a grounded one",
     validator("TRAINING ML") is not None, validator("TRAINING ML"))

print("\n6c. REAL FAILURE: a word from a DIFFERENT anchor collapses the AND structure")
# Captured live: "DETECTION" (the detection anchor's own word) got added to
# BOTH the acoustic row and the drone row during expansion. Since rows are
# ANDed but satisfied by ANY one word, a document containing only the single
# word "detection" then satisfied all 3 rows at once -- 58,279 hits instead
# of ~100. The forbidden-word check must block this.
strict_validator = _words_grounded_in_seed(seed_words, forbidden={"DETECTION", "DRONE"})
check("a real seed word that belongs to another anchor is still rejected here",
     strict_validator("DETECTION") is not None, strict_validator("DETECTION"))
check("an unrelated anchor's word mixed with a genuine local one still rejects the whole entry",
     strict_validator("SENSITIVE DETECTION") is not None, strict_validator("SENSITIVE DETECTION"))

print("\n7. queue_targeted_search: Step 4 -- a real narrower search for a confirmed branch")


class FakePendingStore:
    def __init__(self, site="inpass"):
        self.pending = []
        self._site = site

    def get_conversation(self, conversation_id):
        return {"conversation_id": conversation_id, "site": self._site}

    def get_pending_searches(self, conversation_id, state="pending"):
        return [p for p in self.pending if p["state"] == state]

    def add_pending_search(self, conversation_id, qid, rows, target_cluster_id):
        self.pending = [p for p in self.pending if p["qid"] != qid]
        self.pending.append({"qid": qid, "rows": rows, "state": "pending",
                            "target_cluster_id": target_cluster_id})


CLUSTER = {"cluster_id": "thermal_based", "label": "thermal based",
          "key_terms": ["THERMAL", "INFRARED"]}

store = FakePendingStore()
queued = queue_targeted_search(store, "c1", CLUSTER)
check("queues a new search for a fresh cluster", queued is True)
pending = store.get_pending_searches("c1")
check("exactly 1 pending search saved", len(pending) == 1, pending)
check("qid is the cluster's own id", pending[0]["qid"] == "thermal_based", pending)
check("includes the safe domain anchors", any("DRONE" in text for _, text in pending[0]["rows"]), pending[0]["rows"])
check("includes the branch's own key terms", any("THERMAL" in text for _, text in pending[0]["rows"]), pending[0]["rows"])

print("\n8. the same cluster is never queued twice")
queued_again = queue_targeted_search(store, "c1", CLUSTER)
check("second call returns False", queued_again is False)
check("still exactly 1 pending search", len(store.get_pending_searches("c1")) == 1)

print("\n8b. queue_targeted_search is site-aware: a ppubs conversation gets PPUBS's own field code")
ppubs_store = FakePendingStore(site="ppubs")
queue_targeted_search(ppubs_store, "c1", CLUSTER)
ppubs_rows = ppubs_store.get_pending_searches("c1")[0]["rows"]
check("uses PPUBS's 'AB' field code, not InPASS's 'ABS'",
     all(code == "AB" for code, _text in ppubs_rows), ppubs_rows)

inpass_store = FakePendingStore(site="inpass")
queue_targeted_search(inpass_store, "c1", CLUSTER)
inpass_rows = inpass_store.get_pending_searches("c1")[0]["rows"]
check("uses InPASS's own 'ABS' field code for an inpass conversation",
     all(code == "ABS" for code, _text in inpass_rows), inpass_rows)

print("\n9. REAL FAILURE end-to-end: model tries to add DETECTION to every row -- must be blocked")
# Reproduces the exact live failure: a model that (wrongly) wants to add the
# detection anchor's own word to the acoustic and drone rows too. Each row
# must fall back to its own unexpanded starting words rather than accept
# the cross-anchor addition, so the 3 rows stay independent AND-gates.
class LeakyLLM:
    """Actually runs the validator the caller passed in, the way the real
    llm.ask() -> parse_kv() pipeline does -- a mock that just hardcodes
    errors=[] would bypass validation entirely and prove nothing."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        if "acoustic/audio" in user:
            raw = "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE DETECTION"
        elif "drone/UAS" in user:
            raw = "DRONE DRONES UAV UAVS UAS QUADCOPTER DETECTION"
        else:
            raw = "DETECTION DETECTING DETECT CLASSIFICATION SENSITIVE"
        validate = spec.fields[0].validate
        err = validate(raw) if validate else None
        if err:
            return KVResult(records=[], errors=[f"WORDS {err}"])
        return KVResult(records=[{"WORDS": raw}], errors=[])

rows = derive_search_rows(SEED, LeakyLLM())
acoustic_row = next(r for r in rows if "ACOUSTIC" in r.text)
drone_row = next(r for r in rows if "DRONE" in r.text)
detect_row = next(r for r in rows if "DETECTION" in r.text and "ACOUSTIC" not in r.text and "DRONE" not in r.text)
check("DETECTION blocked from the acoustic row", "DETECTION" not in acoustic_row.text.split(), acoustic_row.text)
check("DETECTION blocked from the drone row", "DETECTION" not in drone_row.text.split(), drone_row.text)
check("DETECTION still present in its own row", "DETECTION" in detect_row.text.split(), detect_row.text)
check("a genuine new word in the detection row still gets through", "SENSITIVE" in detect_row.text.split(), detect_row.text)

print("\n10. experimental fully-dynamic derivation: well-formed response accepted")
FULL_SEED = ("Application for C-UAS. Framework - Combination of hardware with software for "
            "effective, low-cost detection solution against UAS threats. ML training on "
            "extensive audio dataset, robust detection at low SNRs, highly sensitive "
            "microphones, detection independent of type of drone")


class FullDeriveGoodLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[
            {"CONCEPT": "audio sensing", "WORDS": "AUDIO MICROPHONES SENSITIVE DATASET"},
            {"CONCEPT": "drone threat", "WORDS": "DRONE UAS THREATS"},
            {"CONCEPT": "detection quality", "WORDS": "DETECTION ROBUST SNRS"},
        ], errors=[])


rows, reason = derive_search_rows_fully_dynamic(FULL_SEED, FullDeriveGoodLLM())
check("well-formed response accepted, 3 rows returned",
     rows is not None and len(rows) == 3 and reason is None, (rows, reason))

print("\n11. REAL RISK: cross-row word overlap must reject the whole response")
# The exact failure class that produced 58,279 hits once before: the same
# word (here DETECTION) satisfying more than one AND-gate collapses their
# independence. With no fixed anchor skeleton to fall back on, the only
# safe response to this is to reject entirely, not guess which row it
# "really" belongs to.
class FullDeriveOverlapLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[
            {"CONCEPT": "audio sensing", "WORDS": "AUDIO MICROPHONES DETECTION"},
            {"CONCEPT": "drone threat", "WORDS": "DRONE UAS DETECTION"},
        ], errors=[])


overlap_rows, overlap_reason = derive_search_rows_fully_dynamic(FULL_SEED, FullDeriveOverlapLLM())
check("cross-row overlap rejects the whole response (None)", overlap_rows is None, overlap_rows)
check("reason names the overlapping word and both concepts",
     "DETECTION" in overlap_reason and "audio sensing" in overlap_reason, overlap_reason)

print("\n12. REAL RISK: an ungrounded (hallucinated) word must reject the whole response")
class FullDeriveHallucinatingLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[
            {"CONCEPT": "audio sensing", "WORDS": "AUDIO WAVELET SCATTERING"},
            {"CONCEPT": "drone threat", "WORDS": "DRONE UAS THREATS"},
        ], errors=[])


halluc_rows, halluc_reason = derive_search_rows_fully_dynamic(FULL_SEED, FullDeriveHallucinatingLLM())
check("ungrounded word rejects the whole response (None)", halluc_rows is None, halluc_rows)
check("reason names the specific ungrounded word", "WAVELET" in halluc_reason, halluc_reason)

print("\n13. REAL FAILURE: a short but meaningful domain acronym (UAS) must stay groundable")
# Captured live: "UAS" (3 letters) was silently filtered out by a length
# floor borrowed from a DIFFERENT function where domain words are
# deliberately excluded anyway -- here it made a genuinely present domain
# word structurally ungroundable, which is the opposite of the point.
seed_words = _seed_words_including_domain(FULL_SEED)
check("UAS is groundable despite being short", "UAS" in seed_words, seed_words)

print("\n14. no LLM available -> None, never a silent guess")
no_llm_rows, no_llm_reason = derive_search_rows_fully_dynamic(FULL_SEED, None)
check("no llm returns None", no_llm_rows is None, no_llm_rows)
check("reason says why", no_llm_reason == "LLM unavailable", no_llm_reason)

print("\n15. REAL FAILURE: a nested 'other' bucket must never be queued for a real search")
# Captured live: a NESTED "other" bucket ("deep_raw_audio__other") got
# normally confirmed like a real branch and queued a real targeted search
# using its fallback label -- literally the word "OTHER" -- as search
# vocabulary, which the real InPASS portal repeatedly choked on with 500
# errors. This is the structural backstop INSIDE queue_targeted_search
# itself, so it can never happen regardless of what calls it.
class OtherPendingStore:
    def __init__(self):
        self.pending = []

    def get_pending_searches(self, conversation_id, state="pending"):
        return [p for p in self.pending if p["state"] == state]

    def add_pending_search(self, conversation_id, qid, rows, target_cluster_id):
        self.pending.append({"qid": qid, "rows": rows, "state": "pending"})


nested_other = {"cluster_id": "deep_raw_audio__other", "label": "other", "key_terms": ["OTHER"]}
top_level_other = {"cluster_id": "other", "label": "other", "key_terms": ["OTHER"]}
store15 = OtherPendingStore()
check("nested other bucket refused, never queued",
     queue_targeted_search(store15, "c1", nested_other) is False)
check("top-level other bucket also refused",
     queue_targeted_search(store15, "c1", top_level_other) is False)
check("nothing was actually queued for either", store15.pending == [], store15.pending)

print("\n16. derive_search_rows_ppubs: no LLM -> the 3 anchor rows, PPUBS-shaped")
ppubs_rows_no_llm = derive_search_rows_ppubs(SEED, None)
check("3 rows returned", len(ppubs_rows_no_llm) == 3, ppubs_rows_no_llm)
check("uses PPUBS's own 'AB' field code, not InPASS's 'ABS'",
     all(r.field_code == "AB" for r in ppubs_rows_no_llm), ppubs_rows_no_llm)
check("same anchor vocabulary as the InPASS version",
     {r.text for r in ppubs_rows_no_llm} == {text for _c, _l, text in _ANCHOR_ROWS}, ppubs_rows_no_llm)

print("\n17. derive_search_rows_ppubs: same grounded-expansion discipline as the InPASS version")
llm = FixedLLM({"acoustic/audio sensing": "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE SENSITIVE"})
ppubs_rows = derive_search_rows_ppubs(SEED, llm)
acoustic_row = next(r for r in ppubs_rows if "MICROPHONE" in r.text)
check("expansion carried through with the PPUBS field code",
     acoustic_row.field_code == "AB" and "SENSITIVE" in acoustic_row.text, acoustic_row)

print("\n18. prepare_search_ppubs: tags the conversation site='ppubs', prepare_search leaves it 'inpass'")
import tempfile
from priorart.core.bootstrap import prepare_search
from priorart.core.store import Store

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "test.db"
    real_store = Store(db_path)

    prepare_search_ppubs(SEED, "ppubs-convo", None, real_store)
    ppubs_convo = real_store.get_conversation("ppubs-convo")
    check("ppubs conversation recorded as site='ppubs'", ppubs_convo["site"] == "ppubs", ppubs_convo)

    prepare_search(SEED, "inpass-convo", None, real_store)
    inpass_convo = real_store.get_conversation("inpass-convo")
    check("plain prepare_search still defaults to site='inpass'", inpass_convo["site"] == "inpass", inpass_convo)

    pending = real_store.get_pending_searches("ppubs-convo")
    check("pending rows use PPUBS's 'AB' field code",
         pending and all(code == "AB" for code, _text in pending[0]["rows"]), pending)

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
