"""Local web UI for driving InPASS prior-art searches.

Runs on the operator's own machine.  Nothing is exposed publicly and no
invention data leaves the host: the profile, the fingerprint and the results
database are all local files.

The captcha is rendered straight into the page for the operator to type.  That
sidesteps the constraint measured in Phase 0 -- a portal session goes stale and
returns HTTP 500 if there is a long gap between issuing a captcha and
submitting the search -- because a human at the keyboard answers in seconds.
An internal vision endpoint remains optional, for unattended sweeps.
"""
from __future__ import annotations

import base64
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory

from ..core.claims import Claim, extract_claims
from ..core.compare import compare_documents, summarize_by_claim
from ..core.ingest import parse_docx
from ..core.llm import LLMClient
from ..core.planner import QueryPlanner  # noqa: F401  (used by /api/plan)
from ..core.profile_builder import derive_profile, to_profile_dict
from ..core.questions import generate_questions
from ..core.scorer import Scorer
from ..core.store import Store
from ..inpass.captcha import build_solver
from ..inpass.client import FIELDS, InpassClient, SearchRow

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- state

@dataclass
class PortalSession:
    """One live portal session: cookies plus the captcha currently displayed."""
    sid: str
    client: InpassClient
    last_result: object | None = None


@dataclass
class Job:
    """A background harvest, polled by the UI for progress."""
    job_id: str
    kind: str
    total: int = 0
    done: int = 0
    state: str = "running"       # running | finished | failed
    message: str = ""
    result: dict = field(default_factory=dict)


class App:
    def __init__(self, config_path: Path, profile_path: Path):
        self.config = yaml.safe_load(config_path.read_text())
        self.profile = yaml.safe_load(profile_path.read_text())
        self.profile_path = profile_path
        db = self.config["run"]["db_path"]
        self.store = Store(db if Path(db).is_absolute() else PKG / db)
        self.scorer = Scorer(self.profile)
        self.sessions: dict[str, PortalSession] = {}
        self.jobs: dict[str, Job] = {}
        self.run_id = (self.store.latest_run(self.profile["meta"]["id"])
                       or self.store.start_run(self.profile["meta"]["id"],
                                               self.config["run"]["date_from"],
                                               self.config["run"]["date_to"],
                                               self.config["run"]["status"],
                                               "web ui"))
        self.llm = LLMClient(self.config)

        # Document pipeline state (single-operator tool: one working set at a
        # time, same pattern as `self.profile` above -- not per-browser-session).
        self.uploads_dir = PKG / "out" / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.last_doc_path: Path | None = None
        self.last_disclosure = None
        self.proposed_profile: dict | None = None
        self.last_claims: list[Claim] = []
        self.last_questions: list = []

        # qid -> (client, SearchResult), decoupled from PortalSession/sid.
        # Harvest used to look up the LAST captcha session instead of the
        # search that actually produced results -- fetching the next
        # captcha (which the UI does right after a successful search)
        # silently swapped the active session out from under it, so Harvest
        # failed with "no active result set" on the very next click. Keying
        # by qid instead means it does not matter what happens to sessions
        # afterward.
        self.completed_searches: dict[str, tuple] = {}

    def new_session(self) -> PortalSession:
        solver = build_solver(self.config, PKG / "out")
        client = InpassClient(self.config, solver)
        sess = PortalSession(uuid.uuid4().hex[:12], client)
        self.sessions[sess.sid] = sess
        # Keep memory bounded during a long working session.
        if len(self.sessions) > 40:
            for old in list(self.sessions)[:-20]:
                self.sessions.pop(old, None)
        return sess


def create_app(config_path: Path, profile_path: Path) -> Flask:
    state = App(config_path, profile_path)
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

    def captcha_payload(sess: PortalSession) -> str:
        return base64.b64encode(sess.client._fetch_captcha()).decode()

    # ------------------------------------------------------------------ pages

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # --------------------------------------------------------------- metadata

    @app.get("/api/profile")
    def get_profile():
        p = state.profile
        return jsonify({
            "meta": p["meta"],
            "facets": {k: {"label": v.get("label", k), "weight": v["weight"],
                           "required": v.get("required", False),
                           "words": v.get("words", []),
                           "phrases": v.get("phrases", [])}
                       for k, v in p["facets"].items()},
            "ipc_codes": p.get("ipc_codes", []),
            "presets": p.get("presets", []),
            "fields": FIELDS,
            "defaults": state.config["run"],
            "vision_enabled": state.config["endpoints"]["vision"].get("enabled", False),
        })

    @app.post("/api/profile/facet")
    def save_facet():
        """Persist edited facet vocabulary back to the profile YAML.

        The UI is the intended way to tune search vocabulary, so edits must
        survive a restart rather than living only in the browser.
        """
        d = request.get_json(force=True)
        name, words, phrases = d["name"], d.get("words", []), d.get("phrases", [])
        raw = yaml.safe_load(state.profile_path.read_text())
        if name not in raw["facets"]:
            raw["facets"][name] = {"label": d.get("label", name),
                                   "weight": d.get("weight", 1.0),
                                   "required": False}
        raw["facets"][name]["words"] = [w.strip().upper() for w in words if w.strip()]
        raw["facets"][name]["phrases"] = [p.strip().upper() for p in phrases if p.strip()]
        state.profile_path.write_text(
            yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100))
        state.profile = raw
        state.scorer = Scorer(raw)
        return jsonify({"ok": True})

    # ---------------------------------------------------------- doc pipeline

    @app.post("/api/upload")
    def upload():
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "no file given"}), 400
        dest = state.uploads_dir / f.filename
        f.save(dest)
        state.last_doc_path = dest
        return jsonify({"path": str(dest), "filename": f.filename})

    @app.post("/api/derive-profile")
    def derive_profile_endpoint():
        """Read the uploaded disclosure and propose facets/pillars/fingerprint.

        Nothing is saved or activated yet -- the proposal is staged so the
        operator can review it (especially pillar statuses, which the tool
        cannot infer from text alone) before it drives anything.
        """
        if not state.last_doc_path or not state.last_doc_path.exists():
            return jsonify({"error": "upload a document first"}), 400
        try:
            state.last_disclosure = parse_docx(str(state.last_doc_path))
            proposed = derive_profile(state.last_disclosure,
                                      llm=state.llm if state.llm.enabled else None)
        except Exception as exc:                               # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

        import re
        default_id = re.sub(r"[^a-z0-9]+", "_", proposed.title.lower()).strip("_")[:40] or "profile"
        full = to_profile_dict(proposed, default_id)
        state.proposed_profile = full
        return jsonify({"profile": full, "method": proposed.method,
                        "suggested_id": default_id})

    @app.post("/api/save-profile")
    def save_profile_endpoint():
        """Write the (possibly operator-edited) proposed profile to disk and
        make it the active profile -- same effect as restarting with
        --profile pointed at a hand-written one."""
        d = request.get_json(force=True)
        profile_dict = d.get("profile") or state.proposed_profile
        profile_id = d.get("profile_id", "").strip()
        if not profile_dict:
            return jsonify({"error": "no proposed profile - call derive-profile first"}), 400
        if not profile_id:
            return jsonify({"error": "profile_id required"}), 400

        profile_dict.setdefault("meta", {})["id"] = profile_id
        path = PKG / "profiles" / f"{profile_id}.yaml"
        path.write_text(yaml.safe_dump(profile_dict, sort_keys=False,
                                       allow_unicode=True, width=100))

        state.profile = profile_dict
        state.profile_path = path
        state.scorer = Scorer(profile_dict)
        state.run_id = (state.store.latest_run(profile_id)
                        or state.store.start_run(profile_id,
                                                 state.config["run"]["date_from"],
                                                 state.config["run"]["date_to"],
                                                 state.config["run"]["status"],
                                                 "web ui - auto-derived profile"))
        return jsonify({"ok": True, "path": str(path), "run_id": state.run_id})

    def _claims_path() -> Path:
        return PKG / "out" / f"{state.profile['meta']['id']}_claims.json"

    def _questions_path() -> Path:
        return PKG / "out" / f"{state.profile['meta']['id']}_questions.json"

    @app.post("/api/extract-claims")
    def extract_claims_endpoint():
        """Extract claim candidates from the uploaded disclosure using the
        CURRENTLY ACTIVE profile -- call save-profile first if the profile
        was just derived from this same document.

        Written to disk immediately, keyed by profile id -- not just kept in
        server memory. This is what lets an intranet-only step (this one) and
        an internet-only step (search) happen in separate browser sessions,
        possibly after a restart or a network switch in between: the
        filesystem is the same machine either way, only the network route
        changes, so a file written here is already sitting there once you
        switch and reopen the page.
        """
        if not state.last_disclosure:
            return jsonify({"error": "upload and derive a profile first"}), 400
        try:
            claims, method = extract_claims(state.last_disclosure, state.profile,
                                            llm=state.llm if state.llm.enabled else None)
        except Exception as exc:                               # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
        state.last_claims = claims
        _claims_path().write_text(json.dumps([vars(c) for c in claims], indent=2))
        return jsonify({"claims": [vars(c) for c in claims], "method": method,
                        "saved_to": str(_claims_path())})

    @app.get("/api/claims")
    def get_claims_endpoint():
        """Reload claims saved by a PRIOR intranet session for the active
        profile -- the counterpart to extract-claims's auto-save."""
        path = _claims_path()
        if not path.exists():
            return jsonify({"claims": [], "found": False})
        claims_raw = json.loads(path.read_text())
        state.last_claims = [Claim(**c) for c in claims_raw]
        return jsonify({"claims": claims_raw, "found": True})

    @app.post("/api/generate-questions")
    def generate_questions_endpoint():
        """Build safe search questions from the (possibly operator-trimmed)
        claim set. Pass {"claim_ids": [...]} to exclude any claims the
        operator removed after reviewing extract-claims' output.

        Auto-saved to disk -- see extract-claims's docstring for why."""
        d = request.get_json(silent=True) or {}
        wanted = set(d.get("claim_ids", [])) or None
        claims = [c for c in state.last_claims if wanted is None or c.id in wanted]
        if not claims:
            return jsonify({"error": "no claims - call extract-claims first"}), 400
        questions = generate_questions(claims, state.profile,
                                       llm=state.llm if state.llm.enabled else None)
        state.last_questions = questions
        _questions_path().write_text(json.dumps([vars(q) for q in questions], indent=2))
        return jsonify({"questions": [vars(q) for q in questions],
                        "saved_to": str(_questions_path())})

    @app.get("/api/questions")
    def get_questions_endpoint():
        """Reload questions saved by a PRIOR session for the active profile --
        the counterpart to generate-questions's auto-save. This is the step
        that runs AFTER switching to the internet network: the questions were
        generated on intranet, saved to disk, and are read back here without
        needing the LLM (or intranet) at all."""
        path = _questions_path()
        if not path.exists():
            return jsonify({"questions": [], "found": False})
        return jsonify({"questions": json.loads(path.read_text()), "found": True})

    @app.post("/api/compare")
    def compare_endpoint():
        """Compare harvested+scored documents against the extracted claims --
        the per-claim verdict, not just a ranked list."""
        if not state.last_claims:
            return jsonify({"error": "no claims - call extract-claims first"}), 400
        max_tier = int((request.get_json(silent=True) or {}).get("max_tier", 2))
        rows = state.store.shortlist(state.run_id, max_tier=max_tier, limit=200)
        documents = [dict(r) for r in rows]
        if not documents:
            return jsonify({"error": "nothing harvested+scored yet"}), 400

        comparisons = compare_documents(documents, state.last_claims, state.profile,
                                        llm=state.llm if state.llm.enabled else None)
        summary = summarize_by_claim(comparisons, state.last_claims)
        return jsonify({
            "summary": summary,
            "documents_compared": len(comparisons),
            "methods": {m: sum(1 for c in comparisons if c.method == m)
                       for m in {c.method for c in comparisons}},
        })

    # ---------------------------------------------------------------- session

    @app.post("/api/session")
    def new_session():
        sess = state.new_session()
        return jsonify({"sid": sess.sid, "captcha": captcha_payload(sess)})

    @app.post("/api/session/<sid>/captcha")
    def refresh_captcha(sid):
        sess = state.sessions.get(sid)
        if not sess:
            return jsonify({"error": "session expired"}), 404
        return jsonify({"captcha": captcha_payload(sess)})

    @app.post("/api/session/<sid>/solve")
    def auto_solve(sid):
        """Solve the current captcha with the configured vision endpoint."""
        sess = state.sessions.get(sid)
        if not sess:
            return jsonify({"error": "session expired"}), 404
        if not state.config["endpoints"]["vision"].get("enabled"):
            return jsonify({"error": "vision endpoint not configured"}), 400
        png = sess.client._fetch_captcha()
        answer = sess.client.solver.solve(png)
        return jsonify({"captcha": base64.b64encode(png).decode(),
                        "answer": answer or ""})

    # ----------------------------------------------------------------- search

    @app.post("/api/search")
    def search():
        d = request.get_json(force=True)
        sess = state.sessions.get(d.get("sid", ""))
        if not sess:
            return jsonify({"error": "session expired - get a new captcha"}), 400

        rows = [SearchRow(r["field"], r["text"]) for r in d["rows"] if r["text"].strip()]
        if not rows:
            return jsonify({"error": "no search terms given"}), 400

        try:
            res = sess.client.search(
                rows,
                date_from=d.get("date_from") or state.config["run"]["date_from"],
                date_to=d.get("date_to") or state.config["run"]["date_to"],
                date_field=d.get("date_field", "PD"),
                status=d.get("status", "published"),
                captcha_answer=d["captcha"],
            )
        except Exception as exc:                               # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 502

        sess.last_result = res
        qid = d.get("qid") or f"UI-{uuid.uuid4().hex[:6]}"
        desc = " AND ".join(f"{r.field_code}:{r.text}" for r in rows)
        state.store.register_queries(state.run_id, [_Q(qid, desc)])
        state.store.mark_query(state.run_id, qid, state="done", total=res.total,
                               harvested=len(res.hits))
        state.store.upsert_hits(state.run_id, qid, res.hits)
        # Keyed by qid, not sid -- see the comment on completed_searches in
        # App.__init__ for why this matters.
        state.completed_searches[qid] = (sess.client, res)

        return jsonify({
            "qid": qid, "total": res.total, "pages": res.total_pages,
            "description": desc,
            "hits": [vars(h) for h in res.hits],
        })

    # ---------------------------------------------------------------- harvest

    @app.post("/api/harvest")
    def harvest():
        """Page through a result set and optionally fetch every abstract.

        Looked up by qid, not by the browser's current captcha session --
        paging and detail-fetch need no captcha at all, so there is no reason
        this should depend on which session is "active" when the button is
        clicked (see completed_searches in App.__init__).
        """
        d = request.get_json(force=True)
        qid = d["qid"]
        entry = state.completed_searches.get(qid)
        if not entry:
            return jsonify({"error": f"no completed search for qid={qid} - run a search first"}), 400
        client, res = entry

        job = Job(uuid.uuid4().hex[:12], "harvest", total=res.total)
        state.jobs[job.job_id] = job
        want_abstracts = d.get("abstracts", True)

        def work():
            try:
                hits = []
                for h in client.iter_all_hits(res):
                    hits.append(h)
                    job.done = len(hits)
                    job.message = f"paging: {len(hits)}/{res.total}"
                state.store.upsert_hits(state.run_id, qid, hits)
                state.store.mark_query(state.run_id, qid, state="done",
                                       total=res.total, harvested=len(hits))

                if want_abstracts:
                    pending = state.store.docs_needing_detail()
                    job.total = len(pending)
                    job.done = 0
                    for i, app_no in enumerate(pending, 1):
                        try:
                            state.store.save_detail(client.fetch_detail(app_no))
                        except Exception as exc:               # noqa: BLE001
                            job.message = f"detail {app_no} failed: {exc}"
                        job.done = i
                        job.message = f"abstracts: {i}/{len(pending)}"
                job.state = "finished"
                job.result = state.store.stats(state.run_id)
            except Exception as exc:                           # noqa: BLE001
                job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"job_id": job.job_id})

    @app.post("/api/fetch-abstracts")
    def fetch_abstracts():
        """Fetch abstracts for every document still missing one.

        Detail retrieval needs no captcha, so this runs on a fresh session and
        does not depend on a prior search in this browser session.  It exists
        because a long harvest can be interrupted -- by a restart, a dropped
        connection, or the operator closing the tab -- and redoing the search
        to recover the remainder would be wasteful.
        """
        pending = state.store.docs_needing_detail()
        if not pending:
            return jsonify({"error": "every document already has an abstract"}), 400

        job = Job(uuid.uuid4().hex[:12], "abstracts", total=len(pending))
        state.jobs[job.job_id] = job
        client = InpassClient(state.config, build_solver(state.config, PKG / "out"))

        def work():
            try:
                for i, app_no in enumerate(pending, 1):
                    try:
                        state.store.save_detail(client.fetch_detail(app_no))
                    except Exception as exc:                   # noqa: BLE001
                        job.message = f"{app_no} failed: {exc}"
                    job.done = i
                    job.message = f"abstracts: {i}/{len(pending)}"
                job.state = "finished"
                job.result = state.store.stats(state.run_id)
            except Exception as exc:                           # noqa: BLE001
                job.state, job.message = "failed", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"job_id": job.job_id, "pending": len(pending)})

    @app.get("/api/job/<job_id>")
    def job_status(job_id):
        job = state.jobs.get(job_id)
        if not job:
            return jsonify({"error": "unknown job"}), 404
        return jsonify(vars(job))

    # ------------------------------------------------------------------ score

    @app.post("/api/score")
    def score():
        rows = state.store.conn.execute(
            "SELECT application_number,title,abstract FROM documents").fetchall()
        docs = [dict(r) for r in rows]
        if not docs:
            return jsonify({"error": "nothing harvested yet"}), 400
        scored = state.scorer.score_documents(docs)
        state.store.save_scores(state.run_id, scored)
        from collections import Counter
        return jsonify({"scored": len(scored),
                        "tiers": Counter(s["tier"] for s in scored)})

    @app.get("/api/shortlist")
    def shortlist():
        max_tier = int(request.args.get("max_tier", 2))
        limit = int(request.args.get("limit", 100))
        rows = state.store.shortlist(state.run_id, max_tier, limit)
        out = []
        for r in rows:
            d = dict(r)
            d["matched_facets"] = json.loads(d.get("matched_facets") or "{}")
            out.append(d)
        return jsonify({"results": out, "stats": state.store.stats(state.run_id)})

    @app.get("/api/export/<fmt>")
    def export(fmt):
        """Write the shortlist and search log to local files.

        The search log matters as much as the shortlist: it is what makes the
        exercise reproducible, and it is what external counsel will want so
        they can start from our coverage rather than from zero.
        """
        rows = [dict(r) for r in state.store.shortlist(state.run_id, 3, 1000)]
        for r in rows:
            r["matched_facets"] = json.loads(r.get("matched_facets") or "{}")
        log = [dict(r) for r in state.store.search_log(state.run_id)]
        stats = state.store.stats(state.run_id)
        outdir = PKG / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            import csv
            path = outdir / "shortlist.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["tier", "score", "application_number", "title",
                            "application_date", "applicant", "rationale",
                            "matched_facets", "abstract"])
                for r in rows:
                    w.writerow([r.get("tier"), r.get("score"),
                                r["application_number"], r.get("title", ""),
                                r.get("application_date", ""), r.get("applicant", ""),
                                r.get("rationale", ""),
                                "; ".join(f"{k}={','.join(v)}"
                                          for k, v in r["matched_facets"].items()),
                                (r.get("abstract") or "").replace("\n", " ")])
            return jsonify({"path": str(path), "rows": len(rows)})

        if fmt == "md":
            p_ = state.profile
            L = [f"# Prior Art Shortlist — {p_['meta']['title']}", "",
                 "**Classification:** Internal. Contains pre-filing invention detail.", "",
                 f"**Database:** InPASS (ipindia.gov.in) · "
                 f"{state.config['run']['date_from']}–{state.config['run']['date_to']} · "
                 f"{state.config['run']['status']}", "",
                 f"**Coverage:** {stats['queries_done']} queries · "
                 f"{stats['documents']} unique documents · "
                 f"{stats['with_abstract']} abstracts retrieved", "",
                 "> **Jurisdiction limit.** InPASS indexes Indian filings only. The art most",
                 "> likely to affect this filing is a US/CN/EP patent or an academic paper.",
                 "> This shortlist is a cheap first pass, not a novelty opinion.", ""]
            for tier, heading in ((1, "Tier 1 — read every one"),
                                  (2, "Tier 2 — skim")):
                sel = [r for r in rows if r.get("tier") == tier]
                L += [f"## {heading} ({len(sel)})", ""]
                for r in sel:
                    ws = " **◆ WHITE SPACE**" if (r.get("rationale") or "").startswith("WHITE SPACE") else ""
                    L += [f"### {r['application_number']} — {r.get('title','')}{ws}", "",
                          f"*{r.get('application_date','')} · {r.get('applicant','') or 'applicant n/a'} "
                          f"· score {r.get('score')}*", "",
                          f"**Why:** {r.get('rationale','')}", "",
                          (r.get("abstract") or "_abstract not retrieved_").strip(), ""]
            L += ["## Search log", "",
                  "| Query | Terms | Hits | Harvested | Run at |", "|---|---|---|---|---|"]
            for q in log:
                L.append(f"| `{q['qid']}` | {q['description'][:110]} | "
                         f"{q.get('total_hits','')} | {q.get('harvested','')} | "
                         f"{q.get('executed_at','')} |")
            path = outdir / "shortlist.md"
            path.write_text("\n".join(L), encoding="utf-8")
            return jsonify({"path": str(path), "rows": len(rows)})

        return jsonify({"error": f"unknown format {fmt}"}), 400

    @app.get("/api/log")
    def log():
        return jsonify({"log": [dict(r) for r in state.store.search_log(state.run_id)],
                        "stats": state.store.stats(state.run_id)})

    return app


class _Q:
    """Minimal query record for ad-hoc searches built in the UI."""
    def __init__(self, qid, desc):
        self.qid, self.axis, self.tier, self._d = qid, "UI", 1, desc

    def describe(self):
        return self._d


def main():
    import argparse
    ap = argparse.ArgumentParser(description="InPASS prior-art search UI")
    ap.add_argument("--config", default=str(PKG / "config.yaml"))
    ap.add_argument("--profile", default=str(PKG / "profiles/acoustic_uav.yaml"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    app = create_app(Path(a.config), Path(a.profile))
    print(f"\n  Prior-art search UI -> http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
