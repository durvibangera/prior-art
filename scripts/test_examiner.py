#!/usr/bin/env python3
"""Unit tests for priorart/core/examiner and LLMClient.converse.

No network: a fake LLM returns canned replies, and requests.post is stubbed
where the real client's payload/streaming handling is under test.
Run: python3 scripts/test_examiner.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import priorart.core.llm as llm_mod
from priorart.core.examiner.chart import assess, chart_reference, quote_verified
from priorart.core.examiner.claims import _number, required_elements
from priorart.core.examiner.context import Invention, Reference, reference_text, retrieve
from priorart.core.examiner.interview import Examiner
from priorart.core.examiner.searches import check_query, distinctive_vocab, to_rows
from priorart.core.llm import (
    ConfidentialityError, LLMClient, LLMReply, assert_private_endpoint, split_reasoning,
)
from priorart.core.store import Store

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
    """Returns queued replies in order; records every call."""
    enabled = True

    def __init__(self, json_replies=None, text_replies=None):
        self.json_replies = list(json_replies or [])
        self.text_replies = list(text_replies or [])
        self.calls = []

    def converse_json(self, messages, schema, **kw):
        self.calls.append(("json", messages, kw))
        return self.json_replies.pop(0)

    def converse(self, messages, **kw):
        self.calls.append(("text", messages, kw))
        text = self.text_replies.pop(0)
        if kw.get("on_token"):
            kw["on_token"](text)
        return LLMReply(text)


CLAIMS = _number([
    {"category": "method", "text": "Detecting drones from audio", "elements": [
        {"text": "capturing an audio signal with a microphone", "role": "context"},
        {"text": "wavelet scattering transform producing multi-resolution coefficients", "role": "distinguishing"},
        {"text": "one-dimensional convolutional network classifier", "role": "distinguishing"},
    ]},
    {"category": "method", "depends_on": "1", "text": "with temporal windowing", "elements": [
        {"text": "averaging predictions over three consecutive windows", "role": "distinguishing"},
    ]},
])


def cell(e, a, status):
    return {"element_id": e, "application_number": a, "status": status, "quote": "", "reasoning": ""}


print("\n1. claim numbering and dependent-claim inheritance")
check("ids assigned in code", [c["id"] for c in CLAIMS] == ["C1", "C2"], CLAIMS)
check("element ids", [e["id"] for e in CLAIMS[0]["elements"]] == ["C1.a", "C1.b", "C1.c"])
check("depends_on resolved", CLAIMS[1]["depends_on"] == "C1")
check("C2 requires C1's elements too",
      required_elements(CLAIMS, "C2") == ["C1.a", "C1.b", "C1.c", "C2.a"],
      required_elements(CLAIMS, "C2"))
bad = _number([{"category": "method", "depends_on": "5", "text": "x", "elements": [{"text": "y", "role": "context"}]}])
check("forward/self dependency dropped", bad[0]["depends_on"] is None)

print("\n2. quote verification")
src = "The system applies a Wavelet Scattering Transform (WST) to the raw audio, then a 1D-CNN."
check("exact quote verified", quote_verified("applies a Wavelet Scattering Transform (WST) to the raw audio", src))
check("case/punctuation/whitespace ignored", quote_verified("applies a wavelet  scattering transform WST", src))
check("ellipsis segments each checked", quote_verified("applies a Wavelet ... then a 1D-CNN", src))
check("fabricated quote rejected", not quote_verified("uses an LSTM over scattering coefficients", src))
check("tiny quote rejected", not quote_verified("the", src))

print("\n3. chart_reference downgrades an unverifiable quote")
ref = Reference(text=reference_text({"application_number": "IN1", "title": "Drone audio",
                                     "abstract": src, "site": "inpass"}), score=1.0)
fake = FakeLLM(json_replies=[{"summary": "s", "closeness": "very_close", "cells": [
    {"element_id": "C1.b", "status": "disclosed", "quote": "applies a Wavelet Scattering Transform (WST)", "reasoning": "r"},
    {"element_id": "C1.c", "status": "disclosed", "quote": "a two-dimensional CNN with attention", "reasoning": "r"},
    {"element_id": "C9.z", "status": "disclosed", "quote": "x", "reasoning": "unknown element"},
]}])
out = chart_reference(fake, CLAIMS, ref)
check("verified cell kept", out["cells"]["C1.b"]["status"] == "disclosed", out["cells"]["C1.b"])
check("fabricated quote -> unverified", out["cells"]["C1.c"]["status"] == "unverified", out["cells"]["C1.c"])
check("unknown element ignored", "C9.z" not in out["cells"])
check("missing elements filled as absent", out["cells"]["C2.a"]["status"] == "absent")

print("\n4. assess: the legal arithmetic is in code")
cells = [cell("C1.a", "A", "disclosed"), cell("C1.b", "A", "disclosed"), cell("C1.c", "A", "disclosed")]
v = assess(CLAIMS, cells)
check("one ref with every element -> anticipated", v["C1"]["status"] == "anticipated" and v["C1"]["anticipating"] == ["A"], v["C1"])
check("dependent claim not anticipated without its extra element", v["C2"]["status"] != "anticipated", v["C2"])
cells = [cell("C1.a", "A", "disclosed"), cell("C1.b", "A", "disclosed"), cell("C1.c", "B", "disclosed")]
v = assess(CLAIMS, cells)
check("two refs covering all -> inventive_step_risk", v["C1"]["status"] == "inventive_step_risk" and v["C1"]["pairs"] == [("A", "B")], v["C1"])
cells = [cell("C1.a", "A", "disclosed"), cell("C1.b", "B", "disclosed"), cell("C1.c", "C", "disclosed")]
check("three-ref mosaic -> all_elements_known", assess(CLAIMS, cells)["C1"]["status"] == "all_elements_known")
cells = [cell("C1.a", "A", "disclosed"), cell("C1.b", "A", "unverified"), cell("C1.c", "A", "disclosed")]
check("unverified never counts as disclosed", assess(CLAIMS, cells)["C1"]["status"] != "anticipated")
cells = [cell("C1.b", "A", "partial")]
v = assess(CLAIMS, cells)
check("partial only -> partially_covered", v["C1"]["status"] == "partially_covered", v["C1"])
check("uncovered elements listed", v["C1"]["uncovered"] == ["C1.a", "C1.c"], v["C1"]["uncovered"])
check("no cells -> clear", assess(CLAIMS, [])["C1"]["status"] == "clear")

print("\n5. search safety gate")
vocab = distinctive_vocab(CLAIMS)
check("context words are not distinctive", "MICROPHONE" not in set().union(*vocab.values()), vocab)
blocked = check_query([["WAVELET", "SCATTERING"], ["AVERAGING"]], CLAIMS)
check("REAL FAILURE: shared word across elements no longer slips through",
      check_query([["WAVELET", "SCATTERING"], ["CONVOLUTIONAL", "WINDOWS"]], CLAIMS) is not None)
check("words spanning a claim and its dependent blocked", blocked is not None, blocked)
check("single element + domain words allowed",
      check_query([["DRONE", "UAV"], ["WAVELET", "SCATTERING"]], CLAIMS) is None)
blocked = check_query([["WAVELET", "SCATTERING"], ["CONVOLUTIONAL"]], CLAIMS)
check("two inventive elements chained -> blocked", blocked is not None and "C1.b" in blocked and "C1.c" in blocked, blocked)
check("domain-only query allowed", check_query([["DRONE", "UAV"], ["AUDIO"]], CLAIMS) is None)
check("too many rows blocked", check_query([["A"]] * 5, CLAIMS) is not None)
rows = to_rows([["wavelet scattering", "Wavelets"], ["drone", "drone"]], "ppubs")
check("rows: words only, upper, deduped, site field code",
      rows == [("AB", "WAVELET SCATTERING WAVELETS"), ("AB", "DRONE")], rows)

print("\n6. confidentiality guard on the LLM endpoint")
for url in ("http://172.16.34.5:8000/v1", "http://127.0.0.1:8000/v1", "http://10.0.0.4/v1", "http://localhost:8000/v1"):
    try:
        assert_private_endpoint(url)
        check(f"accepts {url}", True)
    except ConfidentialityError as exc:
        check(f"accepts {url}", False, exc)
for url in ("https://8.8.8.8/v1", "http://1.1.1.1:8000/v1", "not a url"):
    try:
        assert_private_endpoint(url)
        check(f"refuses {url}", False)
    except ConfidentialityError:
        check(f"refuses {url}", True)
try:
    LLMClient({"endpoints": {"llm": {"enabled": True, "base_url": "https://8.8.8.8/v1"}}})
    check("LLMClient refuses a public endpoint", False)
except ConfidentialityError:
    check("LLMClient refuses a public endpoint", True)
check("disabled client is not checked",
      not LLMClient({"endpoints": {"llm": {"enabled": False, "base_url": "https://8.8.8.8"}}}).enabled)

print("\n7. reference_text on a full Indian specification")
body = "FIELD OF INVENTION the present invention relates to acoustic sensing. " * 2000
spec = (body + " WE CLAIM: 1. A method comprising receiving sound and computing wavelet coefficients. "
        + "2. The method of claim 1. " + body[:5000]
        + " ABSTRACT A drone detection system using wavelet scattering of microphone audio.")
rt = reference_text({"application_number": "IN2", "title": "T", "abstract": spec})
check("abstract taken from the trailing ABSTRACT heading", rt.abstract.startswith("A drone detection system"), rt.abstract[:60])
check("claims section found", rt.claims.startswith("WE CLAIM") and "wavelet coefficients" in rt.claims, rt.claims[:60])
check("chunked", len(rt.chunks) > 50 and all(len(c) <= 1500 for c in rt.chunks))
short = reference_text({"application_number": "IN3", "title": "T", "abstract": "Short abstract."})
check("short abstract kept whole", short.abstract == "Short abstract." and short.claims == "")

print("\n8. retrieval: per-element queries surface single-element art")
docs = [
    {"application_number": "D1", "title": "Acoustic drone detection", "abstract": "Microphone array audio drone detection using MFCC and SVM."},
    {"application_number": "D2", "title": "Bearing fault diagnosis", "abstract": "Wavelet scattering transform coefficients for vibration fault diagnosis in bearings."},
    {"application_number": "D3", "title": "Crop yield", "abstract": "Satellite imagery regression for crop yield forecasting."},
    {"application_number": "D4", "title": "Empty", "abstract": ""},
]
refs = retrieve(docs, ["wavelet scattering transform coefficients"],
                domain_query="drone audio microphone detection", k_per_query=1, max_docs=2)
got = [r.application_number for r in refs]
check("unrelated-field scattering art retrieved (technique track)", "D2" in got, got)
check("same-field art retrieved (problem track)", "D1" in got, got)
check("irrelevant doc not retrieved", "D3" not in got, got)
check("empty abstract skipped", "D4" not in got)
long_generic = {"application_number": "D5", "title": "Tax anomaly detection",
                "abstract": "Financial statements are analysed for tax anomalies. " * 60
                + "A convolutional neural network with batch normalization and LSTM layers. " * 3}
refs = retrieve(docs + [long_generic], ["convolutional network"], domain_query="drone audio microphone detection",
                k_per_query=1, max_docs=2)
check("a long generic spec cannot displace same-problem art",
      [r.application_number for r in refs][0] == "D1", [r.application_number for r in refs])

print("\n9. split_reasoning")
check("full think block stripped", split_reasoning("<think>hmm</think>\nAnswer.") == ("Answer.", "hmm"))
check("pre-opened template (lone </think>)", split_reasoning("reasoning here\n</think>\n\nAnswer.") == ("Answer.", "reasoning here"))
check("no reasoning", split_reasoning("Just text.") == ("Just text.", ""))

print("\n10. converse(): payload and streaming")
captured = {}


class _Resp:
    def __init__(self, body=None, lines=None):
        self._body, self._lines, self.ok = body, lines or [], True

    def raise_for_status(self):
        pass

    def json(self):
        return self._body

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_post(url, json=None, headers=None, timeout=None, stream=False):
    captured["payload"] = json
    if stream:
        chunks = ["Thinking about", " it...", "</think>", "Hello", " there."]
        return _Resp(lines=[f"data: {__import__('json').dumps({'choices': [{'delta': {'content': c}}]})}"
                            for c in chunks] + ["data: [DONE]"])
    return _Resp(body={"choices": [{"message": {"content": '{"a": 1}', "reasoning_content": "why"}}]})


real_post = llm_mod.requests.post
llm_mod.requests.post = fake_post
try:
    client = LLMClient({"endpoints": {"llm": {"enabled": True, "base_url": "http://127.0.0.1:8000/v1",
                                              "model": "m"}}})
    out = client.converse_json([{"role": "user", "content": "x"}], {"type": "object"})
    p = captured["payload"]
    check("guided JSON parsed", out == {"a": 1}, out)
    check("thinking off for JSON by default", p["chat_template_kwargs"] == {"enable_thinking": False}, p)
    check("response_format json_schema sent", p["response_format"]["type"] == "json_schema", p)
    seen = []
    reply = client.converse([{"role": "user", "content": "x"}], thinking=True, on_token=seen.append)
    check("stream: inline reasoning never shown", "".join(seen) == "Hello there.", seen)
    check("stream: answer and reasoning split", reply.text == "Hello there." and "Thinking" in reply.reasoning, reply)
    check("thinking on in payload", captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True})
finally:
    llm_mod.requests.post = real_post

print("\n11. examiner: messages persist only on success; case-file update applied")
tmp = Path(tempfile.mkdtemp())
store = Store(tmp / "t.db")
store.create_conversation("c1", "seed")
store.set_chart_cell("c1", "C1.b", "IN1", "partial", "q", "r")
store.set_chart_ref("c1", "IN1", "uses DWT features", "related")
inv = Invention(path="x", sections={"title": "Drone detector", "description": "WST then 1D-CNN."})
fake = FakeLLM(text_replies=["What does your WST hand to the CNN?"])
ex = Examiner(store, "c1", fake, inv, CLAIMS, [ref], {"examiner": {"history_token_budget": 10**9}})
ex._send("[kickoff]", kind="control")
msgs = store.get_messages("c1")
check("control + reply stored", [(m["role"], m["kind"]) for m in msgs] == [("user", "control"), ("assistant", "chat")], msgs)
sys_prompt = fake.calls[0][1][0]["content"]
check("system prompt carries the disclosure and chart", "WST then 1D-CNN." in sys_prompt and "C1.b vs IN1: partial" in sys_prompt)


class Failing(FakeLLM):
    def converse(self, messages, **kw):
        raise llm_mod.LLMUnavailable("down")


ex.llm = Failing()
check("failed call returns None", ex._send("my answer") is None)
check("failed call stores nothing", len(store.get_messages("c1")) == 2)
note = ex.apply_update({
    "new_facts": ["second-order scattering, J=8, Q=8", "second-order scattering, J=8, Q=8"],
    "distinguishing_features": [{"feature": "GAP over scattering paths", "claims": ["C1"], "against": ["IN1"],
                                 "strength": "moderate", "why": "w"}],
    "chart_updates": [{"element_id": "C1.b", "application_number": "IN1", "status": "absent", "reason": "DWT is not WST"},
                      {"element_id": "C7.a", "application_number": "IN1", "status": "absent", "reason": "no such cell"}],
    "open_issues": ["baseline accuracy?"],
    "suggested_searches": [{"purpose": "p", "rows": [["SCATTERING"]]}],
})
check("duplicate facts collapsed", store.get_facts("c1") == ["second-order scattering, J=8, Q=8"], store.get_facts("c1"))
cellrow = store.get_chart_cells("c1")[0]
check("justified chart cell revised, with provenance",
      cellrow["status"] == "absent" and cellrow["source"] == "interview" and "revised" in cellrow["reasoning"], cellrow)
check("update to a non-existent cell ignored", len(store.get_chart_cells("c1")) == 1)
check("issues and searches stored", store.kv_get("c1", "open_issues") == ["baseline accuracy?"]
      and len(store.kv_get("c1", "suggested_searches")) == 1)
check("note summarises changes", "+1 fact" in note and "1 chart cell" in note, note)
ex.apply_update({"new_facts": [], "distinguishing_features": [], "chart_updates": [], "open_issues": None,
                 "suggested_searches": [{"purpose": "again", "rows": [["scattering"]]}]})
check("re-suggested search not duplicated", len(store.kv_get("c1", "suggested_searches")) == 1,
      store.kv_get("c1", "suggested_searches"))

print("\n12. examiner: history compaction")
for i in range(30):
    store.add_message("c1", "user", f"answer {i} " + "x" * 400)
    store.add_message("c1", "assistant", f"question {i} " + "y" * 400)
ex.llm = FakeLLM(text_replies=["SUMMARY OF EARLY TURNS"])
ex.history_budget = 1000
ex._maybe_compact()
kept = store.get_messages("c1")
check("summary stored", store.kv_get("c1", "history_summary") == "SUMMARY OF EARLY TURNS")
check("recent window kept, starting on a user turn", 0 < len(kept) <= 20 and kept[0]["role"] == "user", len(kept))
check("archived turns still in the transcript", len(store.get_messages("c1", include_archived=True)) == 62)
check("summary reaches the system prompt", "SUMMARY OF EARLY TURNS" in ex.system_prompt())

print(f"\n{'=' * 50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
