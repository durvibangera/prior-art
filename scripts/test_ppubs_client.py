#!/usr/bin/env python3
"""Unit tests for priorart/ppubs/client.py -- the USPTO Patent Public Search
client. Network-mocked (no live calls) except where noted -- the real,
live-confirmed protocol details (query syntax, hit counts, abstract page
structure) were verified separately against the real API before this client
was written; see the module's own docstring.
Run: python3 scripts/test_ppubs_client.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from priorart.ppubs.client import (
    DATABASES, FIELDS, MAX_ROWS, Hit, PortalError, PpubsClient, SearchResult, SearchRow,
    build_query,
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


CFG = {"ppubs": {"base_url": "https://ppubs.uspto.gov", "delay_between_requests_s": 0,
                 "request_timeout_s": 5, "max_retries": 2, "page_size": 10}}


print("\n1. SearchRow validates field_code and logic")
SearchRow("AB", "acoustic")
try:
    SearchRow("NOPE", "x")
    check("unknown field_code raises", False)
except ValueError:
    check("unknown field_code raises", True)
try:
    SearchRow("AB", "x", logic="MAYBE")
    check("unknown logic raises", False)
except ValueError:
    check("unknown logic raises", True)


print("\n2. build_query: single row, no leading operator")
q = build_query([SearchRow("AB", "acoustic")])
check("single row query", q == "(acoustic).ab.", q)

print("\n3. build_query: multiple rows joined by AND, matching the real query syntax")
# Verified live: this exact multi-clause form returned the SAME hit count as
# the single-phrase form against the real API (8 hits either way).
q = build_query([SearchRow("AB", "acoustic"), SearchRow("AB", "drone"), SearchRow("AB", "detection")])
check("multi-row AND query", q == "(acoustic).ab. AND (drone).ab. AND (detection).ab.", q)

print("\n4. build_query: OR/NOT logic carried through per row")
q = build_query([SearchRow("TI", "drone"), SearchRow("TI", "uav", logic="OR"),
                 SearchRow("AB", "camera", logic="NOT")])
check("OR/NOT logic preserved", q == "(drone).ti. OR (uav).ti. NOT (camera).ab.", q)

print("\n5. build_query: empty rows rejected, never a blank query sent")
try:
    build_query([])
    check("empty rows raises", False)
except ValueError:
    check("empty rows raises", True)

print("\n5b. REAL FAILURE: a multi-word synonym row must use explicit OR, not bare juxtaposition")
# Captured live running the real first PPUBS harvest: derive_search_rows_ppubs
# builds rows like "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE" meant as an
# OR'd synonym group (exactly like InPASS's own row design). Sent bare
# ("(ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE).ab." with no "OR" between
# the words), the real API returned only 5 hits for that clause alone --
# NOT the loose "any one of these words" match InPASS's own row text gets --
# and the full 3-row AND of three such bare clauses returned 0 hits total,
# instead of the ~100 a real broad harvest should find. Confirmed live:
# joining with explicit " OR " for that SAME clause returned 82,567+ hits
# alone, and the full 3-row AND-of-explicit-ORs returned a sane 107.
multi_word_query = build_query([
    SearchRow("AB", "ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE"),
    SearchRow("AB", "DRONE DRONES UAV UAVS UAS QUADCOPTER"),
    SearchRow("AB", "DETECTION DETECTING DETECT CLASSIFICATION"),
])
check("each row's synonyms are OR'd together explicitly",
     multi_word_query == ("(ACOUSTIC OR ACOUSTICS OR AUDIO OR SOUND OR MICROPHONE).ab. AND "
                          "(DRONE OR DRONES OR UAV OR UAVS OR UAS OR QUADCOPTER).ab. AND "
                          "(DETECTION OR DETECTING OR DETECT OR CLASSIFICATION).ab."),
     multi_word_query)

print("\n5c. a genuinely single-word row is completely unaffected by the OR fix")
single_word_query = build_query([SearchRow("AB", "acoustic"), SearchRow("AB", "drone")])
check("single-word rows still join with no OR needed",
     single_word_query == "(acoustic).ab. AND (drone).ab.", single_word_query)


class FakeResponse:
    def __init__(self, json_data, status_code=200, headers=None):
        self._json = json_data
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    @property
    def text(self):
        return ""


class FakeSession:
    """Records every request it's given and returns pre-programmed
    responses in order -- exercises PpubsClient's real request-building
    code (routed through _request, shared by search and fetch_detail), not
    a shortcut that skips it."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []
        self.headers = {}

    def get(self, *a, **kw):
        return FakeResponse({})

    def request(self, method, url, params=None, data=None, headers=None, timeout=None):
        self.posts.append((url, data, headers))
        return self.responses.pop(0)


print("\n6. PpubsClient.search parses real-shaped hits and preserves the cursor")
client = PpubsClient(CFG, connect=False)
client.token = "tok-123"
client.session = FakeSession([FakeResponse({
    "cursorMarker": "AoJwwK+1abc",
    "numFound": 2,
    "docs": [
        {"documentId": "US-20230366697-A1", "title": "SOUND SOURCE MODELING OF DRONES",
         "datePublished": "2023-11-16", "type": "US-PGPUB"},
        {"documentId": "US-10358196-B2", "title": "UNMANNED AERIAL VEHICLE SYSTEM",
         "datePublished": "2019-07-23", "type": "USPAT"},
    ],
})])
result = client.search([SearchRow("AB", "acoustic"), SearchRow("AB", "drone")])
check("total parsed", result.total == 2, result)
check("cursor_marker carried through", result.cursor_marker == "AoJwwK+1abc", result)
check("2 hits parsed", len(result.hits) == 2, result.hits)
check("Hit.status carries the source needed for fetch_detail later",
     result.hits[0].status == "US-PGPUB", result.hits[0])
check("access token sent on the request", client.session.posts[0][2]["x-access-token"] == "tok-123",
     client.session.posts[0])

print("\n7. PpubsClient.search rejects more than MAX_ROWS")
client2 = PpubsClient(CFG, connect=False)
client2.token = "tok"
client2.session = FakeSession([])
too_many = [SearchRow("AB", f"w{i}") for i in range(MAX_ROWS + 1)]
try:
    client2.search(too_many)
    check("too many rows raises", False)
except ValueError:
    check("too many rows raises", True)

print("\n8. iter_all_hits follows the cursor across pages, stops when the cursor runs dry")
client3 = PpubsClient(CFG, connect=False)
client3.token = "tok"
page1 = SearchResult(total=4, cursor_marker="cursor-2", hits=[
    Hit("US-1", "A", "2020", "USPAT"), Hit("US-2", "B", "2020", "USPAT")])
client3.session = FakeSession([
    FakeResponse({"cursorMarker": "cursor-3", "numFound": 4,
                 "docs": [{"documentId": "US-3", "title": "C", "datePublished": "2020", "type": "USPAT"},
                         {"documentId": "US-4", "title": "D", "datePublished": "2020", "type": "USPAT"}]}),
    FakeResponse({"cursorMarker": "", "numFound": 4, "docs": []}),
])
all_hits = list(client3.iter_all_hits([SearchRow("AB", "x")], page1))
check("all 4 hits collected across pages", [h.application_number for h in all_hits] == ["US-1", "US-2", "US-3", "US-4"],
     all_hits)

print("\n9. iter_all_hits stops immediately if the first page already has everything")
client4 = PpubsClient(CFG, connect=False)
client4.token = "tok"
client4.session = FakeSession([])   # would raise IndexError if a page fetch were wrongly attempted
full_page = SearchResult(total=2, cursor_marker="unused", hits=[Hit("US-1", "A", "2020", "USPAT"),
                                                                Hit("US-2", "B", "2020", "USPAT")])
all_hits2 = list(client4.iter_all_hits([SearchRow("AB", "x")], full_page))
check("no extra page fetched once total is already reached", len(all_hits2) == 2, all_hits2)

print("\n10. _extract_abstract on a REAL captured page structure (US-20260202241-A1)")
# Captured live from GET /api/patents/html/US-20260202241-A1 -- trimmed to
# the relevant section, structure otherwise verbatim.
REAL_DETAIL_HTML = """
<html><body>
<section>
<h2 class="bottom-border padding">UNMANNED AERIAL VEHICLE BASED ABNORMAL NOISE DETECTION DEVICE</h2>
<article class="bottom-border padding">
    <h3>Abstract</h3>
    <p>A method for operating an unmanned aerial vehicle based abnormal noise detection device includes
    capturing, using a camera attached to a drone to generate image data, an image of a measurement target;
    collecting, using an acoustic sensor attached to the drone, acoustic data corresponding to each of the cells.</p>
</article>
</section>
</body></html>
"""
from bs4 import BeautifulSoup
soup = BeautifulSoup(REAL_DETAIL_HTML, "lxml")
text, suspect = PpubsClient._extract_abstract(soup)
check("real abstract text extracted", text.startswith("A method for operating"), text[:60])
check("not flagged suspect for a normal-length real abstract", suspect is False, suspect)

print("\n11. _extract_abstract: no Abstract heading found -> honest empty, flagged suspect")
soup_bad = BeautifulSoup("<html><body><p>nothing relevant here</p></body></html>", "lxml")
text_bad, suspect_bad = PpubsClient._extract_abstract(soup_bad)
check("empty text, never a guess", text_bad == "", text_bad)
check("flagged suspect", suspect_bad is True, suspect_bad)

print("\n12. _extract_abstract: heading present but no following <p> -> honest empty, flagged suspect")
soup_noP = BeautifulSoup("<html><body><h3>Abstract</h3><div>not a p tag</div></body></html>", "lxml")
text_noP, suspect_noP = PpubsClient._extract_abstract(soup_noP)
check("empty text when structure doesn't match", text_noP == "", text_noP)
check("flagged suspect", suspect_noP is True, suspect_noP)

print("\n13. FIELDS / DATABASES are the confirmed-live set, not silently expanded")
check("AB and TI present (both tested live)", {"AB", "TI"} <= set(FIELDS))
check("PD deliberately excluded (range syntax never confirmed live)", "PD" not in FIELDS)
check("all 3 real PPUBS collections included", set(DATABASES) == {"USPAT", "US-PGPUB", "USOCR"}, DATABASES)

print("\n14. REAL FAILURE: fetch_detail used to have NO retry at all, unlike search()")
# Captured live: partway through a real 107-document detail-fetch loop, the
# real site's rate limiter tripped (HTTP 429) -- fetch_detail did a single
# bare GET with no retry, so every document failed outright from that point
# on. Fixed by routing fetch_detail through the SAME _request retry helper
# search() already used. time.sleep is stubbed out here so this test
# doesn't actually wait through the real backoff delays.
import priorart.ppubs.client as ppubs_client_module
_real_sleep = ppubs_client_module.time.sleep
ppubs_client_module.time.sleep = lambda *_a, **_kw: None


class FlakyResponse(FakeResponse):
    def __init__(self, status_code, text=""):
        super().__init__({}, status_code=status_code)
        self._text = text

    @property
    def text(self):
        return self._text


try:
    client5 = PpubsClient(CFG, connect=False)
    client5.token = "tok"
    real_detail_html = (
        "<html><body><section><h2>title</h2><article><h3>Abstract</h3>"
        "<p>A method for operating an unmanned aerial vehicle.</p></article></section></body></html>"
    )
    # 429s on the first attempt, succeeds on the second -- exercises the
    # real retry path, not a mock that only ever returns success.
    client5.session = FakeSession([
        FlakyResponse(429),
        FlakyResponse(200, text=real_detail_html),
    ])
    detail = client5.fetch_detail("US-20260202241-A1", "US-PGPUB")
    check("fetch_detail retries past a 429 instead of failing outright",
         detail["abstract"].startswith("A method for operating"), detail)
    check("2 attempts actually made (1 failure + 1 success)", len(client5.session.posts) == 2,
         client5.session.posts)
finally:
    ppubs_client_module.time.sleep = _real_sleep

print("\n15. REAL FAILURE: fetch_detail's GET must NOT send Accept: application/json")
# Captured live: the real detail endpoint returns HTML, and responds 406 Not
# Acceptable, on EVERY attempt (not a transient failure -- every retry hit
# the same wall) when Accept: application/json is sent. Confirmed live with
# a direct side-by-side request: identical call with vs without that header
# -> 406 vs 200. search()'s POST body genuinely is JSON and still needs it.
client6 = PpubsClient(CFG, connect=False)
client6.token = "tok"
client6.session = FakeSession([FlakyResponse(200, text="<html><body></body></html>")])
client6.fetch_detail("US-1", "USPAT")
check("fetch_detail sends no Accept/Content-Type override",
     "Accept" not in client6.session.posts[0][2] and "Content-Type" not in client6.session.posts[0][2],
     client6.session.posts[0][2])

client7 = PpubsClient(CFG, connect=False)
client7.token = "tok"
client7.session = FakeSession([FakeResponse({"cursorMarker": "", "numFound": 0, "docs": []})])
client7.search([SearchRow("AB", "acoustic")])
check("search's POST still sends Accept: application/json",
     client7.session.posts[0][2].get("Accept") == "application/json", client7.session.posts[0][2])

print(f"\n{'='*50}\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)
