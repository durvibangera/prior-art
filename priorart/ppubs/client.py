"""Client for the USPTO Patent Public Search "basic search" API
(ppubs.uspto.gov/basic).

Protocol, established by reading the site's own JS source directly
(js/94649dd7.custom-scripts.js, js/models/Search.js -- unminified ES
modules, not reverse-engineered from obfuscated code) and confirmed against
the REAL live API:

  GET  /basic/                            landing page, sets session cookies
  POST /api/users/me/session              body "-1" -> X-Access-Token response header
  POST /api/searches/generic              the search itself  -- NO captcha
  GET  /api/patents/html/{documentId}     full text (title/abstract/claims/etc) -- NO captcha

Unlike InPASS, there is NO captcha anywhere in this flow -- the whole thing
is plain requests, no browser, no OCR. (Espacenet was tried first and
rejected: worldwide.espacenet.com returns an instant 403 from a WAF, with no
page served at all -- not worth pursuing without a real browser.)

The query language is `(term).fieldcode.` chunks joined by AND/OR/NOT,
directly analogous to InPASS's per-row AND design: one term per
field-scoped clause, ANDed together, rather than one loose multi-word
phrase -- which matters for the same reason noted in inpass/client.py:
InPASS's own docstring warns that a single loose multi-word row can return
hundreds of thousands of hits. Confirmed live here too: both
"(acoustic drone detection).ab." and "(acoustic).ab. AND (drone).ab. AND
(detection).ab." returned the SAME 8 hits for a real test query -- a sane,
single-digit count, not an explosion -- so the per-row-AND design carries
over safely.

Confirmed field codes (tested live): AB (abstract), TI (title). The basic
UI's own field dropdown also exposes AANM/AS/ATT/ATTY/PN/IN (applicant
name/assignee name/attorney firm/attorney name/patent number/inventor
name) via the exact same generic `(value).code.` construction in the site's
own JS, so they're included here on that basis, but were NOT individually
tested live the way AB/TI were.

NOT supported yet, on purpose rather than silently: date-range filtering.
The basic UI treats "Publication date" as a special `@pd="value"` query
fragment instead of the generic `(value).code.` one, and its real range
syntax (e.g. an actual FROM/TO range, not a single exact date) was never
confirmed live -- so PD is deliberately left out of FIELDS rather than
guessed at and silently doing the wrong thing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# code -> human label, same convention as inpass.client.FIELDS. The code
# itself (lowercased) is what actually appears in the query string, e.g.
# "(acoustic).ab.".
FIELDS = {
    "AB": "Abstract", "TI": "Title",
    "PN": "Patent/Application publication number", "IN": "Inventor name",
    "AANM": "Applicant name", "AS": "Assignee name",
    "ATT": "Attorney agent/firm", "ATTY": "Attorney name",
}
MAX_ROWS = 16   # no confirmed real limit -- kept at parity with InPASS's own
# cap as a sane default until/unless a real limit is observed.

# The three PPUBS collections the basic search UI queries by default: issued
# US patents, published US applications, and older OCR'd patents.
DATABASES = ("USPAT", "US-PGPUB", "USOCR")

_RESULT_FIELDS = ["documentId", "patentNumber", "title", "datePublished",
                  "inventors", "pageCount", "type"]


class PortalError(Exception):
    """The portal returned something we could not parse, or authentication
    failed outright."""


@dataclass
class SearchRow:
    field_code: str          # one of FIELDS
    text: str
    logic: str = "AND"       # AND | OR | NOT -- joins this row to the PREVIOUS one

    def __post_init__(self):
        if self.field_code not in FIELDS:
            raise ValueError(f"unknown field code {self.field_code!r}")
        if self.logic not in ("AND", "OR", "NOT"):
            raise ValueError(f"unknown logic {self.logic!r}")


@dataclass
class Hit:
    application_number: str   # PPUBS's own "documentId", e.g. "US-20230366697-A1"
    title: str
    application_date: str     # PPUBS's "datePublished"
    status: str                # PPUBS's "type": USPAT | US-PGPUB | USOCR --
    # also the exact `source` value fetch_detail needs for THIS SAME
    # document, so a caller never has to track it separately.


@dataclass
class SearchResult:
    total: int
    cursor_marker: str          # opaque pagination token -- PPUBS paginates
    # by cursor, not page number, unlike InPASS's total_pages/QueryString.
    hits: list[Hit] = field(default_factory=list)


def _clause(row: SearchRow) -> str:
    """One row's own words joined with an EXPLICIT "OR" -- confirmed live
    that bare space-separated words in one clause do NOT behave as an OR of
    synonyms the way a single loosely-worded InPASS row does. A real test
    against the live API: "(ACOUSTIC ACOUSTICS AUDIO SOUND MICROPHONE).ab."
    (bare) returned only 5 hits (read as something closer to "all of these
    words present", not "any one of these"), while
    "(ACOUSTIC OR ACOUSTICS OR AUDIO OR SOUND OR MICROPHONE).ab." (explicit
    OR) returned 82,567+ for that clause alone and, ANDed with two more such
    OR-groups (drone-synonyms, detection-synonyms), a sane combined 107 hits
    -- so a row meant as an OR'd synonym group (the same shape
    derive_search_rows_ppubs builds) MUST use explicit "OR", not bare
    juxtaposition, or a real harvest silently returns 0 hits instead of
    ~100. A genuinely single-word row is unaffected either way.
    """
    words = row.text.split()
    inner = " OR ".join(words) if len(words) > 1 else row.text
    return f"({inner}).{row.field_code.lower()}."


def build_query(rows: list[SearchRow]) -> str:
    """"(term).code." chunks joined by AND/OR/NOT -- the exact syntax the
    real site's own JS builds (convertValueToQuery / the box1-operator-box2
    join), generalized here to any number of rows the way InPASS's
    per-row-AND design already works. Each row's OWN words are OR'd
    together first (see _clause) -- rows are what's ANDed against each
    other, exactly mirroring InPASS's "one row per concept, each concept an
    OR-group of synonyms, rows ANDed together" design.
    """
    if not rows:
        raise ValueError("at least one search row is required")
    parts = [_clause(rows[0])]
    for row in rows[1:]:
        parts.append(f"{row.logic} {_clause(row)}")
    return " ".join(parts)


class PpubsClient:
    def __init__(self, cfg: dict, *, connect: bool = True):
        ppubs_cfg = cfg["ppubs"]
        self.base = ppubs_cfg["base_url"].rstrip("/")
        self.delay = ppubs_cfg.get("delay_between_requests_s", 1.5)
        self.timeout = ppubs_cfg.get("request_timeout_s", 30)
        self.max_retries = ppubs_cfg.get("max_retries", 3)
        self.page_size = ppubs_cfg.get("page_size", 50)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self.token: str | None = None
        if connect:
            self._authenticate()

    # ---------------------------------------------------------------- session

    def _authenticate(self) -> None:
        """Landing-page GET (cookies) then the session POST that hands back
        an X-Access-Token -- required on every request after this."""
        self.session.get(f"{self.base}/basic/", timeout=self.timeout)
        r = self.session.post(
            f"{self.base}/api/users/me/session", data="-1",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=self.timeout)
        r.raise_for_status()
        token = r.headers.get("X-Access-Token") or r.headers.get("x-access-token")
        if not token:
            raise PortalError("no access token in session response")
        self.token = token

    def _pause(self, seconds: float | None = None) -> None:
        time.sleep(self.delay if seconds is None else seconds)

    def _request(self, method: str, url: str, *, params: dict | None = None,
                data: str | None = None, extra_headers: dict | None = None
                ) -> requests.Response:
        """Shared retry-with-backoff, used by BOTH _post_json (search) and
        fetch_detail. fetch_detail previously did a single bare GET with NO
        retry at all -- caught live: partway through a real 107-document
        detail-fetch loop, the real site's rate limiter tripped (HTTP 429,
        "There are too many requests in your user session"), and EVERY
        following document failed outright with no second attempt, since
        search()'s own retry logic never covered this code path. Retries on
        ANY failure (network error, 429, 5xx) with exponential backoff, and
        a single re-auth-and-retry on 401 -- same broad "retry on anything,
        this portal is known to be flaky" precedent InPASS's own client
        already uses.

        `extra_headers` lets a caller ADD to (never silently replace) the
        base `x-access-token` header -- caught live, in an earlier version
        of this method, ALWAYS forcing `Accept: application/json` broke
        fetch_detail outright: that endpoint returns HTML, and the real
        server responds 406 Not Acceptable to that header on every single
        attempt (not transient -- every retry failed identically, unlike a
        real rate-limit or network blip). _post_json still needs and passes
        it explicitly; fetch_detail passes nothing extra.
        """
        last: Exception | None = None
        reauthed = False
        for attempt in range(self.max_retries):
            try:
                headers = {"x-access-token": self.token, **(extra_headers or {})}
                # A 401 mid-loop re-authenticates and updates self.token, but
                # fetch_detail's own `params["requestToken"]` was built once
                # by the caller before this loop started -- refresh it here
                # too so a retry after re-auth doesn't resend the now-stale
                # token as a query param while the header already has the
                # fresh one.
                if params and "requestToken" in params:
                    params = {**params, "requestToken": self.token}
                r = self.session.request(method, url, params=params, data=data,
                                         headers=headers, timeout=self.timeout)
                if r.status_code == 401 and not reauthed:
                    self._authenticate()
                    reauthed = True
                    continue
                r.raise_for_status()
                return r
            except Exception as exc:                      # noqa: BLE001
                last = exc
                wait = 2 ** attempt * 5
                log.warning("%s %s failed (%s), retry in %ss", method, url, exc, wait)
                time.sleep(wait)
        raise PortalError(f"{method} {url} failed after {self.max_retries}: {last}")

    def _post_json(self, path: str, payload: dict) -> requests.Response:
        return self._request("POST", f"{self.base}{path}", data=json.dumps(payload),
                             extra_headers={"Content-Type": "application/json",
                                           "Accept": "application/json"})

    # ----------------------------------------------------------------- search

    def search(self, rows: list[SearchRow], *, page_size: int | None = None,
              cursor_marker: str = "*") -> SearchResult:
        if len(rows) > MAX_ROWS:
            raise ValueError(f"at most {MAX_ROWS} rows supported, got {len(rows)}")
        payload = {
            "cursorMarker": cursor_marker,
            "databaseFilters": [{"databaseName": d} for d in DATABASES],
            "fields": _RESULT_FIELDS,
            "op": "AND",
            "pageSize": page_size or self.page_size,
            "q": build_query(rows),
            "searchType": 0,
            "sort": "date_publ desc",
        }
        resp = self._post_json("/api/searches/generic", payload)
        data = resp.json()
        hits = [Hit(application_number=d["documentId"], title=d.get("title") or "",
                    application_date=d.get("datePublished") or "", status=d.get("type") or "")
               for d in data.get("docs", [])]
        return SearchResult(total=data.get("numFound", 0),
                            cursor_marker=data.get("cursorMarker", ""), hits=hits)

    # -------------------------------------------------------------- paginate

    def iter_all_hits(self, rows: list[SearchRow], result: SearchResult,
                      page_size: int | None = None) -> Iterator[Hit]:
        """Yield every hit across all pages, following the cursor forward.
        No further authentication needed beyond the token already held."""
        yield from result.hits
        seen = len(result.hits)
        cursor = result.cursor_marker
        while seen < result.total and cursor:
            self._pause()
            nxt = self.search(rows, page_size=page_size, cursor_marker=cursor)
            if not nxt.hits:
                break
            yield from nxt.hits
            seen += len(nxt.hits)
            cursor = nxt.cursor_marker

    # ----------------------------------------------------------------- detail

    _ABSTRACT_LEN_SANITY_CAP = 3000  # same convention as InPASS's client:
    # a real abstract is a paragraph or two; anything past this means the
    # structural extraction grabbed more than the abstract, and the caller
    # should be told rather than silently trusting it.

    def fetch_detail(self, application_number: str, source: str) -> dict:
        """Fetch the real text-view page and pull out the abstract. No
        captcha. `source` is the document's OWN `Hit.status` value
        (USPAT/US-PGPUB/USOCR) -- required by the real endpoint, unlike
        InPASS which needs only the application number.
        """
        self._pause()
        r = self._request(
            "GET", f"{self.base}/api/patents/html/{application_number}",
            params={"source": source, "requestToken": self.token})
        soup = BeautifulSoup(r.text, "lxml")
        abstract, abstract_suspect = self._extract_abstract(soup)
        return {
            "application_number": application_number,
            "abstract": abstract,
            "abstract_suspect": abstract_suspect,
        }

    @classmethod
    def _extract_abstract(cls, soup: BeautifulSoup) -> tuple[str, bool]:
        """The real page structure (confirmed live): an <h3>Abstract</h3>
        heading immediately followed by a <p> containing just the abstract
        text -- a cleaner, more explicit boundary than InPASS's page needed
        (see inpass/client.py's own _extract_abstract for why that one has
        to hunt for a specific <td>). Returns (text, suspect) -- suspect is
        True if the expected structure wasn't found (caller gets "" rather
        than a silent guess) or the text is implausibly long for a real
        abstract.
        """
        heading = soup.find(lambda tag: tag.name in ("h2", "h3")
                            and tag.get_text(strip=True).lower() == "abstract")
        if heading is None:
            return "", True
        p = heading.find_next_sibling("p")
        if p is None:
            return "", True
        text = p.get_text(" ", strip=True)
        suspect = len(text) > cls._ABSTRACT_LEN_SANITY_CAP
        return text, suspect
