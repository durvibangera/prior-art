#!/usr/bin/env python3
"""Unit tests for the KV-lines parser.

Every case below is either a real malformation observed from gemma-e4 on this
task, or a shape the parser must survive because the model produced it once.
Run: python3 scripts/test_kv_parser.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.core.llm import (
    FieldSpec, RecordSpec, parse_kv, contains_all, max_words, non_empty,
    one_of_menu,
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


# ── spec used by most tests: the A1 pipeline skeleton ──────────────────────
A1 = RecordSpec(
    record_key="COMPONENT",
    min_records=2,
    fields=[
        FieldSpec("COMPONENT"),
        FieldSpec("NAME", validate=max_words(8)),
        FieldSpec("DOES", validate=contains_all("takes", "produces")),
    ],
)

print("\n1. clean multi-record response")
r = parse_kv("""COMPONENT: C1
NAME: Acoustic feature front-end
DOES: takes raw audio and produces a time-frequency representation
COMPONENT: C2
NAME: Convolutional classifier
DOES: takes a representation and produces a class score
END""", A1)
check("parses 2 records", len(r.records) == 2, r.records)
check("no errors", r.ok, r.errors)
check("field values correct", r.records[0]["NAME"] == "Acoustic feature front-end")

print("\n2. text after END is discarded")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
DOES: takes a and produces b
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END
I hope this helps! Let me know if you need anything else.
COMPONENT: C99
NAME: should not appear""", A1)
check("stops at END", len(r.records) == 2, [x.get("COMPONENT") for x in r.records])

print("\n3. preamble chatter before the first field is discarded")
r = parse_kv("""Sure! Here are the stages of the system:

COMPONENT: C1
NAME: Stage one
DOES: takes a and produces b
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("ignores preamble", len(r.records) == 2 and r.ok, r.errors)

print("\n4. wrapped long value continues onto the next line")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
DOES: takes raw microphone audio and produces
a multi-resolution time-frequency representation
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("continuation joined", "multi-resolution" in r.records[0]["DOES"], r.records[0]["DOES"])
check("validator still passes", r.ok, r.errors)

print("\n5. THE JSON KILLER -- a value containing quotes, braces, commas")
# In JSON this exact value would need escaping and was the observed failure.
r = parse_kv("""COMPONENT: C1
NAME: Stage one
DOES: takes {"raw": audio}, commas, "quotes" and produces output
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("survives quotes/braces/commas", r.ok, r.errors)
check("value intact", '"quotes"' in r.records[0]["DOES"], r.records[0]["DOES"])

print("\n6. malformation is LOCAL, not total")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
DOES: this line forgot the template words entirely
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("bad record still parsed", len(r.records) == 2, r.records)
check("error names the field", any("DOES" in e for e in r.errors), r.errors)
check("the GOOD record survived", r.records[1]["DOES"].startswith("takes b"), r.records[1])

print("\n7. missing required field is reported, not fatal")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("missing DOES reported", any("DOES" in e and "missing" in e for e in r.errors), r.errors)
check("still returns both records", len(r.records) == 2)

print("\n8. unknown fields dropped, not fatal")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
CONFIDENCE: high
DOES: takes a and produces b
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c
END""", A1)
check("unknown recorded", "CONFIDENCE" in r.unknown_fields, r.unknown_fields)
check("not an error", r.ok, r.errors)

print("\n9. repeat modes")
MULTI = RecordSpec(fields=[FieldSpec("PRIOR", repeat="multi", required=False)])
r = parse_kv("PRIOR: MFCC\nPRIOR: mel-spectrogram\nPRIOR: LPCC\nEND", MULTI)
check("multi collects all", r.one.get("PRIOR") == ["MFCC", "mel-spectrogram", "LPCC"], r.one)

OVER = RecordSpec(fields=[FieldSpec("NAME")])
r = parse_kv("NAME: first\nNAME: second\nEND", OVER)
check("overwrite keeps last", r.one["NAME"] == "second", r.one)

APPEND = RecordSpec(fields=[FieldSpec("NOTE", repeat="append")])
r = parse_kv("NOTE: part one\nNOTE: part two\nEND", APPEND)
check("append joins", r.one["NOTE"] == "part one part two", r.one)

print("\n10. no END at all (model ran out of tokens)")
r = parse_kv("""COMPONENT: C1
NAME: Stage one
DOES: takes a and produces b
COMPONENT: C2
NAME: Stage two
DOES: takes b and produces c""", A1)
check("still parses without END", len(r.records) == 2 and r.ok, r.errors)

print("\n11. total garbage never raises")
for junk in ["", "   ", "I cannot help with that.", "```json\n{\"a\":1}\n```", "\n\n\n"]:
    try:
        r = parse_kv(junk, A1)
        check(f"survived {junk[:20]!r}", not r.ok and isinstance(r.records, list))
    except Exception as e:                                     # noqa: BLE001
        check(f"survived {junk[:20]!r}", False, f"RAISED {e}")

print("\n12. menu validator accepts digit, label, or whole line")
MENU = {"1": "Acoustic signal processing", "2": "AI and ML classification architecture",
        "3": "Counter-drone interception", "4": "Secure data handling"}
v = one_of_menu(MENU)
check("whole line ok", v("2 AI and ML classification architecture") is None, v("2 AI and ML classification architecture"))
check("digit alone ok", v("2") is None, v("2"))
check("label alone ok", v("AI and ML classification architecture") is None, v("AI and ML classification architecture"))
check("garbage rejected", v("probably the ML one") is not None)
check("empty rejected", v("") is not None)

print("\n12b. REAL FAILURE: a more capable model explains instead of copying the menu line")
# Captured live against qwen3-max: instead of copying a bare digit, it wrote
# a full explanation and never produced anything the strict check
# recognized -- the same field kept falling back to neutral every round,
# which is why the same question kept getting re-asked.
check("concluding-with-digit pattern recovered",
     v("The reply describes a system based on ML training data, so this is a 2.") is None)
check("a digit mentioned mid-sentence for an unrelated reason is NOT recovered",
     v("The 3rd point made in the reply doesn't relate to this candidate at all") is not None)
check("pure unresolved explanation still fails honestly, no guess",
     v("The reply describes a system based on training data, which suggests") is not None)

print("\n13. other validators")
check("max_words passes", max_words(5)("one two three") is None)
check("max_words fails", max_words(2)("one two three") is not None)
check("non_empty fails on blank", non_empty("   ") is not None)
check("contains_all case-insensitive", contains_all("takes")("It TAKES input") is None)

print("\n14. REAL FAILURE: model drops every colon in an otherwise correct response")
# Captured verbatim from a live gemma-e4 response that produced 0 records
# under the strict-colon parser. Content was entirely correct.
A2 = RecordSpec(fields=[
    FieldSpec("OURS"), FieldSpec("SETTING1", required=False),
    FieldSpec("SETTING2", required=False), FieldSpec("FIELD1", required=False),
    FieldSpec("FIELD2", required=False), FieldSpec("NEWNESS", required=False),
    FieldSpec("PILLAR", required=False),
])
real_failure = (
    "STAGE Acoustic signal acquisition stage\n"
    "JOB takes raw audio waveform samples and produces a time-domain signal stream\n"
    "OURS Wavelet Scattering Transform (WST) layer\n"
    "SETTING1 Q=8\nSETTING2 Morlet wavelet\n"
    "FIELD1 Wavelet Scattering Transform\nFIELD2 Short-time Fourier Transform\n"
    "NEWNESS instead of standard Fourier-based methods this system uses a non-trainable")
r = parse_kv(real_failure, A2)
check("colon-dropped OURS recovered", r.one.get("OURS") == "Wavelet Scattering Transform (WST) layer", r.one)
check("colon-dropped SETTING1 recovered", r.one.get("SETTING1") == "Q=8", r.one)
check("unknown STAGE/JOB correctly dropped, not mistaken for fields",
      "STAGE" not in r.one and "JOB" not in r.one)

print("\n15. loose match does NOT hijack a real wrapped continuation line")
# A continuation line starting with a capitalised word that is NOT a known
# field name must still be appended, not misread as a new field.
r = parse_kv("NEWNESS: instead of MFCC this system uses a Fixed multi-resolution transform\nEND",
             RecordSpec(fields=[FieldSpec("NEWNESS")]))
check("capitalised word mid-continuation is not hijacked",
      "Fixed multi-resolution" in r.one.get("NEWNESS", ""), r.one)

print("\n16. REAL FAILURE: single-field spec gets a response with no field name at all")
# Captured verbatim from live gemma-e4 in the step-2 dialogue loop. For a
# single-field spec the model sometimes skips the field name entirely --
# not just the colon (that's case 14) but the whole "NAME " wrapper -- and
# just writes the bare answer.
QUESTION_ONE = RecordSpec(fields=[FieldSpec("QUESTION", validate=max_words(45))])
r = parse_kv(
    "Was your project more like the first, vision based, or more like the "
    "second, thermal based?\nEND", QUESTION_ONE)
check("bare sentence recovered as QUESTION", r.ok and "vision based" in r.one.get("QUESTION", ""), r.one)

MENU = {"1": "clearly confirms", "2": "tentatively suggests", "3": "neutral",
        "4": "tentatively rules out", "5": "clearly rules out"}
SENTIMENT_ONE = RecordSpec(fields=[FieldSpec("SENTIMENT", validate=one_of_menu(MENU))])
r = parse_kv("5\nEND", SENTIMENT_ONE)
check("bare digit recovered as SENTIMENT", r.ok and r.one.get("SENTIMENT") == "5", r.one)

print("\n17. the single-field fallback must NOT rescue a genuine refusal")
# A refusal has no field name either, so it hits the same code path -- the
# fallback must not turn "I cannot help with that." into a fake valid answer.
# For a menu spec the validator correctly rejects it either way.
r = parse_kv("I cannot help with that.\nEND", SENTIMENT_ONE)
check("refusal still rejected by the menu validator", not r.ok, r.one)

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
