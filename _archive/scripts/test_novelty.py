#!/usr/bin/env python3
"""Unit tests for priorart/core/novelty.py -- Step 7's separate, post-hoc
novelty comparison (real invention text vs. each shortlisted document's real
abstract). Run: python3 scripts/test_novelty.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.llm import KVResult, parse_kv
from priorart.core.novelty import (
    NOVELTY_COMPARE_SPEC, compare_shortlist_to_invention, compare_to_invention,
    invention_text_from_disclosure,
)

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


print("\n1. no LLM available -> honest UNKNOWN fallback, never a fabricated comparison")
d1 = doc("D1", title="Acoustic classifier", abstract="uses a CNN on raw audio")
result = compare_to_invention("my real invention text", d1, None)
check("closeness is UNKNOWN", result.closeness == "UNKNOWN", result)
check("method is fallback", result.method == "fallback")
check("no fields fabricated", result.technique == "" and result.similarity == "" and result.difference == "")


class DisabledLLM:
    enabled = False


print("\n2. LLM present but disabled -> same honest fallback")
result2 = compare_to_invention("my real invention text", d1, DisabledLLM())
check("closeness is UNKNOWN", result2.closeness == "UNKNOWN")
check("method is fallback", result2.method == "fallback")


print("\n3. a real, well-formed LLM response is parsed and the menu resolved to its label")
sent_prompts = []


class GoodLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        sent_prompts.append(user)
        raw = (
            "TECHNIQUE: Classifies drone sounds using a CNN on raw waveform audio.\n"
            "SIMILARITY: Both classify acoustic drone signatures with a neural network.\n"
            "DIFFERENCE: My invention adds a fixed wavelet scattering transform front-end and an LSTM stage.\n"
            "CLOSENESS: 2 = SOMEWHAT_CLOSE -- shares a real technical element, but meaningfully different overall\n"
            "END"
        )
        return parse_kv(raw, spec)


result3 = compare_to_invention("my real WST+CNN+LSTM invention text", d1, GoodLLM())
check("technique captured", "CNN" in result3.technique, result3.technique)
check("similarity captured", "acoustic" in result3.similarity.lower(), result3.similarity)
check("difference captured", "wavelet" in result3.difference.lower(), result3.difference)
check("closeness resolved to SOMEWHAT_CLOSE", result3.closeness == "SOMEWHAT_CLOSE", result3.closeness)
check("method is llm", result3.method == "llm")
check("real invention text was actually sent to the LLM",
     "WST+CNN+LSTM" in sent_prompts[0], sent_prompts)


print("\n4. malformed CLOSENESS (not copied from the menu) fails real validation -> fallback, not a guess")
class BadMenuLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        raw = (
            "TECHNIQUE: does something.\n"
            "SIMILARITY: none\n"
            "DIFFERENCE: everything\n"
            "CLOSENESS: pretty similar I think\n"
            "END"
        )
        return parse_kv(raw, spec)


result4 = compare_to_invention("invention text", d1, BadMenuLLM())
check("validation actually failed on the real KVResult",
     not parse_kv(
         "TECHNIQUE: x\nSIMILARITY: x\nDIFFERENCE: x\nCLOSENESS: pretty similar I think\nEND",
         NOVELTY_COMPARE_SPEC).ok)
check("falls back to UNKNOWN rather than guessing a closeness", result4.closeness == "UNKNOWN", result4.closeness)
check("method is fallback", result4.method == "fallback")


print("\n5. endpoint-down style failure (ask returns errors, never raises) -> fallback")
class DownLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        return KVResult(records=[], errors=["ENDPOINT down"])


result5 = compare_to_invention("invention text", d1, DownLLM())
check("closeness UNKNOWN", result5.closeness == "UNKNOWN")
check("method fallback", result5.method == "fallback")


print("\n6. compare_shortlist_to_invention runs one call per document and honors the limit")
calls = []


class CountingLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        calls.append(user)
        raw = "TECHNIQUE: x\nSIMILARITY: x\nDIFFERENCE: x\nCLOSENESS: 3\nEND"
        return parse_kv(raw, spec)


docs = [doc(f"D{i}") for i in range(5)]
out = compare_shortlist_to_invention("invention text", docs, CountingLLM(), limit=3)
check("exactly 3 calls made (limit honored)", len(calls) == 3, len(calls))
check("3 comparisons returned, one per document", len(out) == 3)
check("each comparison keeps its own application_number",
     [c.application_number for c in out] == ["D0", "D1", "D2"], out)


print("\n7. invention_text_from_disclosure joins only real content sections")
class FakeDisclosure:
    def joined(self, *keys):
        assert keys == ("nature", "description", "inventive_features", "technical_advancement"), keys
        return "nature text\n\ndescription text"


import priorart.core.ingest as ingest_module
_orig_parse_docx = ingest_module.parse_docx
ingest_module.parse_docx = lambda path, **kw: FakeDisclosure()
try:
    text = invention_text_from_disclosure("fake/path.docx")
finally:
    ingest_module.parse_docx = _orig_parse_docx
check("joins the real content sections, in the right order",
     text == "nature text\n\ndescription text", text)


print("\n8. architecture facets are computed by the SAME deterministic keyword match used for clustering")
wst_cnn_lstm_invention = "my system uses a WAVELET transform frontend, a CNN, and an LSTM stage"
cnn_only_doc = doc("D2", title="CNN classifier", abstract="a CNN classifies raw audio waveforms")
result8 = compare_to_invention(wst_cnn_lstm_invention, cnn_only_doc, GoodLLM())
check("shared facet found (both use CNN/deep raw audio)",
     "deep_raw_audio" in result8.shared_facets, result8.shared_facets)
check("invention-only facets found (fixed transform frontend + sequence model, absent from this doc)",
     set(result8.invention_only_facets) == {"fixed_transform_frontend", "sequence_model"},
     result8.invention_only_facets)


print("\n9. REAL RISK: a VERY_CLOSE verdict with NO shared architecture keyword gets capped down")
class OverconfidentLLM:
    """Claims VERY_CLOSE regardless of what's actually shared -- the grounded
    cap must catch this, not just trust the model's own free-text read."""
    enabled = True

    def ask(self, system, user, spec, **kw):
        raw = (
            "TECHNIQUE: does something vaguely related.\n"
            "SIMILARITY: it is about drones.\n"
            "DIFFERENCE: not much.\n"
            "CLOSENESS: 1 = VERY_CLOSE -- uses essentially the same technique for essentially the same purpose\n"
            "END"
        )
        return parse_kv(raw, spec)


unrelated_doc = doc("D3", title="Vision drone tracker", abstract="uses a CAMERA and OPTICAL flow to track a drone")
result9 = compare_to_invention(wst_cnn_lstm_invention, unrelated_doc, OverconfidentLLM())
check("no shared architecture facet in this case", result9.shared_facets == [], result9.shared_facets)
check("VERY_CLOSE capped down to SOMEWHAT_CLOSE", result9.closeness == "SOMEWHAT_CLOSE", result9.closeness)
check("method explains the cap", "capped" in result9.method, result9.method)


print("\n10. a genuinely grounded VERY_CLOSE (real shared architecture) is NOT capped")
class GenuineVeryCloseLLM:
    enabled = True

    def ask(self, system, user, spec, **kw):
        raw = (
            "TECHNIQUE: uses a wavelet transform frontend then an LSTM for classification.\n"
            "SIMILARITY: same wavelet frontend and sequence model.\n"
            "DIFFERENCE: minor.\n"
            "CLOSENESS: 1 = VERY_CLOSE -- uses essentially the same technique for essentially the same purpose\n"
            "END"
        )
        return parse_kv(raw, spec)


close_doc = doc("D4", title="Wavelet LSTM classifier",
               abstract="a WAVELET transform frontend feeds an LSTM sequence model for classification")
result10 = compare_to_invention(wst_cnn_lstm_invention, close_doc, GenuineVeryCloseLLM())
check("real shared facets found", len(result10.shared_facets) > 0, result10.shared_facets)
check("VERY_CLOSE trusted, not capped", result10.closeness == "VERY_CLOSE" and result10.method == "llm", result10)


print("\n11. compare_shortlist_to_invention computes invention facets once but keeps each doc's own facets separate")
docs_mixed = [cnn_only_doc, unrelated_doc, close_doc]
out11 = compare_shortlist_to_invention(wst_cnn_lstm_invention, docs_mixed, GoodLLM())
check("each result reflects its OWN document's facets, not a shared stale value",
     out11[0].shared_facets != out11[1].shared_facets or (out11[0].shared_facets and out11[1].shared_facets),
     [(r.application_number, r.shared_facets) for r in out11])
check("D2 (cnn only) shares deep_raw_audio", "deep_raw_audio" in out11[0].shared_facets, out11[0].shared_facets)
check("D3 (vision only) shares nothing", out11[1].shared_facets == [], out11[1].shared_facets)
check("D4 (wavelet+lstm) shares fixed_transform_frontend and sequence_model",
     set(out11[2].shared_facets) == {"fixed_transform_frontend", "sequence_model"}, out11[2].shared_facets)


print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
