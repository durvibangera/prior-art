"""Client for the InPASS public patent search (iprsearch.ipindia.gov.in).

Protocol, established by inspecting the live site:

  GET  /                                   landing page, sets ASP.NET session
  GET  /Captcha/GetCaptchaImage            200x40 PNG, bound to that session
  POST /PublicationSearch/Search           the search itself  -- needs captcha
  POST /PublicationSearch/PatentSearchResult   paging  -- NO captcha
  POST /PublicationSearch/PatentDetails    abstract -- NO captcha

The server caches each executed query under a GUID returned in the results
page as `QueryString`, which is why paging and detail fetches need no further
captcha.  The practical consequence is that a captcha is spent per *query*,
not per *result*, so a broad multi-query sweep is cheap.

The search form is ASP.NET MVC, not WebForms: there is no __VIEWSTATE and no
__RequestVerificationToken, so a plain form POST works and no browser is
required.

Up to 16 search rows are supported, each a (field, text, logic) triple.  This
matters enormously: putting "acoustic drone detection" into ONE row matches
loosely and returns hundreds of thousands of hits, whereas one term per row
ANDed together returns tens.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests
from bs4 import BeautifulSoup

from .captcha import CaptchaSolver

log = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# Field codes accepted by ItemFieldN.
FIELDS = {
    "TI": "Title", "ABS": "Abstract", "CSP": "Complete Specification",
    "AP": "Application Number", "PN": "Publication Number",
    "PA": "Applicant Name", "IN": "Inventor Name",
    "IC": "International Patent Classification", "FO": "Filing office",
}
MAX_ROWS = 16


class CaptchaRejected(Exception):
    """The portal did not accept the captcha answer."""


class PortalError(Exception):
    """The portal returned something we could not parse."""


@dataclass
class SearchRow:
    field_code: str          # one of FIELDS
    text: str
    logic: str = "AND"       # AND | OR | NOT

    def __post_init__(self):
        if self.field_code not in FIELDS:
            raise ValueError(f"unknown field code {self.field_code!r}")
        if self.logic not in ("AND", "OR", "NOT"):
            raise ValueError(f"unknown logic {self.logic!r}")


@dataclass
class Hit:
    application_number: str
    title: str
    application_date: str
    status: str


@dataclass
class SearchResult:
    total: int
    total_pages: int
    query_guid: str
    hits: list[Hit] = field(default_factory=list)


class InpassClient:
    def __init__(self, cfg: dict, solver: CaptchaSolver, *, connect: bool = True):
        portal = cfg["portal"]
        self.base = portal["base_url"].rstrip("/")
        self.delay = portal.get("delay_between_requests_s", 1.5)
        self.query_delay = portal.get("delay_between_queries_s", 4.0)
        self.timeout = portal.get("request_timeout_s", 90)
        self.max_retries = portal.get("max_retries", 3)
        self.max_attempts = cfg.get("captcha", {}).get("max_attempts", 5)
        self.solver = solver
        self.captcha_stats = {"attempts": 0, "solved": 0, "rejected": 0}
        # `connect=False` defers the landing-page GET for callers that are
        # about to restore a captured session anyway.
        self.session = self._new_session() if connect else requests.Session()
        if not connect:
            self.session.headers.update(self._default_headers())

    # ---------------------------------------------------------------- session

    def _default_headers(self) -> dict:
        return {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": self.base.rsplit("/PublicSearch", 1)[0],
            "Referer": self.base + "/",
        }

    def _new_session(self) -> requests.Session:
        """Open a session, retrying: the fronting F5 resets connections freely."""
        last: Exception | None = None
        for attempt in range(self.max_retries):
            s = requests.Session()
            s.headers.update(self._default_headers())
            try:
                s.get(self.base + "/", timeout=self.timeout)
                return s
            except Exception as exc:                      # noqa: BLE001
                last = exc
                time.sleep(2 ** attempt * 3)
        raise PortalError(f"could not open session: {last}")

    def restore_session(self, cookies) -> None:
        """Adopt a previously captured cookie jar.

        Deliberately skips the landing-page GET: that would start a *new*
        ASP.NET session and orphan any captcha already issued against the old
        one.  Used to resume an interrupted run, and to batch captcha solving
        separately from query submission.
        """
        s = requests.Session()
        s.headers.update(self.session.headers)
        s.cookies.update(cookies)
        self.session = s

    def _pause(self, seconds: float | None = None) -> None:
        time.sleep(self.delay if seconds is None else seconds)

    def _post(self, path: str, data) -> requests.Response:
        """POST with retry on transport errors (the portal drops connections)."""
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(self.base + path, data=data,
                                      timeout=self.timeout)
                r.raise_for_status()
                return r
            except Exception as exc:                      # noqa: BLE001
                last = exc
                wait = 2 ** attempt * 3
                log.warning("POST %s failed (%s), retry in %ss", path, exc, wait)
                time.sleep(wait)
        raise PortalError(f"POST {path} failed after {self.max_retries}: {last}")

    # ---------------------------------------------------------------- captcha

    def _fetch_captcha(self) -> bytes:
        r = self.session.get(self.base + "/Captcha/GetCaptchaImage",
                             timeout=self.timeout)
        r.raise_for_status()
        return r.content

    # ----------------------------------------------------------------- search

    def search(self, rows: list[SearchRow], *, date_from: str, date_to: str,
               date_field: str = "PD", status: str = "published",
               captcha_answer: str | None = None) -> SearchResult:
        """Run one query.  Consumes one captcha; retries on rejection.

        `captcha_answer` supplies an already-solved captcha for the first
        attempt.  This matters because fetching a captcha *invalidates the
        previous one for this session*, so a caller that solved one out of band
        must be able to use it without triggering another fetch.  Later
        attempts fall back to the configured solver.
        """
        if not rows:
            raise ValueError("at least one search row is required")
        if len(rows) > MAX_ROWS:
            raise ValueError(f"portal accepts at most {MAX_ROWS} rows, got {len(rows)}")

        form = self._build_form(rows, date_from, date_to, date_field, status)

        for attempt in range(1, self.max_attempts + 1):
            self.captcha_stats["attempts"] += 1
            if captcha_answer is not None:
                answer, captcha_answer = captcha_answer, None
            else:
                answer = self.solver.solve(self._fetch_captcha())
            if not answer:
                log.warning("captcha unreadable (attempt %d/%d)",
                            attempt, self.max_attempts)
                self._pause()
                continue

            resp = self._post("/PublicationSearch/Search",
                              form + [("CaptchaText", answer), ("submit", "Search")])
            html = resp.text

            if self._captcha_rejected(html):
                self.captcha_stats["rejected"] += 1
                log.warning("captcha %r rejected (attempt %d/%d)",
                            answer, attempt, self.max_attempts)
                self.solver.report_failure(answer)
                self._pause()
                continue

            self.captcha_stats["solved"] += 1
            return self._parse_results(html)

        raise CaptchaRejected(f"failed to solve captcha in {self.max_attempts} attempts")

    def _build_form(self, rows, date_from, date_to, date_field, status):
        """Build the multipart-ordered form body.

        `Published`/`Granted` use the ASP.NET MVC checkbox idiom: a "true"
        value only when ticked, always followed by a hidden "false".  Emitting
        both in the right order is what makes the status filter bind.
        """
        form: list[tuple[str, str]] = []
        want_pub = status in ("published", "both")
        want_grant = status in ("granted", "both")
        if want_pub:
            form.append(("Published", "true"))
        form.append(("Published", "false"))
        if want_grant:
            form.append(("Granted", "true"))
        form.append(("Granted", "false"))

        form += [("DateField", date_field),
                 ("FromDate", date_from), ("ToDate", date_to),
                 ("LogicField", "AND")]

        for i, row in enumerate(rows, start=1):
            form += [(f"ItemField{i}", row.field_code),
                     (f"TextField{i}", row.text.upper()),
                     (f"LogicField{i}", row.logic)]
        return form

    @staticmethod
    def _captcha_rejected(html: str) -> bool:
        low = html.lower()
        return any(p in low for p in
                   ("invalid captcha", "wrong captcha", "captcha is invalid",
                    "please enter captcha", "incorrect captcha"))

    # ---------------------------------------------------------------- parsing

    _TOTAL_RE = re.compile(r"Total\s*Document\(s\)\s*:?\s*([\d,]+)", re.I)

    def _parse_results(self, html: str) -> SearchResult:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)

        m = self._TOTAL_RE.search(text)
        if not m:
            if "no record" in text.lower() or "not found" in text.lower():
                return SearchResult(0, 0, "", [])
            raise PortalError("could not find result count in response")
        total = int(m.group(1).replace(",", ""))

        guid = self._hidden(soup, "QueryString") or ""
        try:
            pages = int(self._hidden(soup, "TotalPages") or 1)
        except ValueError:
            pages = 1

        return SearchResult(total, pages, guid, self._parse_rows(soup))

    @staticmethod
    def _hidden(soup: BeautifulSoup, name: str) -> str | None:
        tag = soup.find("input", {"name": name})
        return tag.get("value") if tag else None

    @staticmethod
    def _parse_rows(soup: BeautifulSoup) -> list[Hit]:
        table = soup.find("table", id="tableData")
        if not table:
            return []
        hits: list[Hit] = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue        # header row
            # The application number lives in a submit button's value, padded
            # with trailing spaces by the portal.
            btn = tds[0].find("button")
            appno = (btn.get("value") if btn else tds[0].get_text()).strip()
            if not appno:
                continue
            hits.append(Hit(
                application_number=appno,
                title=tds[1].get_text(" ", strip=True),
                application_date=tds[2].get_text(strip=True),
                status=tds[3].get_text(strip=True),
            ))
        return hits

    # -------------------------------------------------------------- paginate

    def iter_all_hits(self, result: SearchResult) -> Iterator[Hit]:
        """Yield every hit across all pages.  No further captchas needed."""
        yield from result.hits
        if not result.query_guid:
            return
        for page in range(2, result.total_pages + 1):
            self._pause()
            resp = self._post("/PublicationSearch/PatentSearchResult", {
                "CurrentPage": str(page - 1),
                "TotalPages": str(result.total_pages),
                "ConnectionName": "PublicationConnection",
                "QueryString": result.query_guid,
                "TotalResult": str(result.total),
                "Title": "",
                "page": str(page),
            })
            yield from self._parse_rows(BeautifulSoup(resp.text, "lxml"))

    # ----------------------------------------------------------------- detail

    # The abstract's real boundary is structural, not textual: on the live
    # page (confirmed by fetching and inspecting the raw HTML directly) the
    # label "Abstract:" and its value live ENTIRELY inside one <td>, and the
    # next <tr> sibling starts a completely separate "Complete Specification
    # FORM 2..." block -- a different section with no heading of its own in
    # the text at all.
    #
    # The previous implementation flattened the whole page to one string and
    # searched for the NEXT of a fixed list of heading labels to bound the
    # abstract. On the real page none of those other labels exist as
    # "label:" text at all (verified live), so the search always failed and
    # the "abstract" silently captured the ENTIRE rest of the page --
    # applicant names, addresses (real PII), the full specification text,
    # and even the site's own footer ("Privacy Policy ... Page last updated
    # on..."). Extracting from the specific <td> instead is bounded by the
    # real page structure, not a guess about what heading might come next.
    _ABSTRACT_LEN_SANITY_CAP = 3000  # a real abstract is a paragraph or two;
    # anything past this means the structural extraction still grabbed more
    # than the abstract, and the caller should be told rather than silently
    # trusting it as clean.

    def fetch_detail(self, application_number: str) -> dict:
        """Fetch the record page and pull out the abstract.  No captcha."""
        self._pause()
        resp = self._post("/PublicationSearch/PatentDetails", {
            "ApplicationNumber": application_number,
            "ConnectionName": "PublicationConnection",
            "IP": "",
        })
        soup = BeautifulSoup(resp.text, "lxml")
        abstract, abstract_suspect = self._extract_abstract(soup)
        return {
            "application_number": application_number,
            "abstract": abstract,
            "abstract_suspect": abstract_suspect,
        }

    @classmethod
    def _extract_abstract(cls, soup: BeautifulSoup) -> tuple[str, bool]:
        """Finds the <td> whose own text starts with "Abstract:" and returns
        JUST that cell's content (never a flat-text heading-boundary guess).
        Returns (text, suspect) -- suspect is True if nothing matched the
        expected structure (caller gets "" rather than a silent guess) or if
        what came back is implausibly long for a real abstract, so a data
        quality problem is visible rather than silently trusted.
        """
        label = soup.find(string=lambda s: s and s.strip().lower() == "abstract:")
        if label is None:
            return "", True

        cell = label.parent
        while cell is not None and cell.name != "td":
            cell = cell.parent
        if cell is None:
            return "", True

        text = cell.get_text(" ", strip=True)
        # Strip the leading "Abstract:" label itself.
        text = re.sub(r"^abstract\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        suspect = len(text) > cls._ABSTRACT_LEN_SANITY_CAP
        return text, suspect
