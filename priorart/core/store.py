"""SQLite store for the conversation-driven prior-art discovery tool.

Fresh schema, built for the chatbot design -- not a continuation of the
earlier keyword-facet pipeline (that whole approach was scrapped). Kept from
the old file: the connection handling (WAL mode, a lock, thread-safe), since
the web UI and any background search work will share one connection across
threads exactly like before.

Everything here has to survive a conversation pausing mid-way while the
operator switches from the intranet (LLM) to the internet (portal) network
and back -- so conversation state lives in these tables, not in memory.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    seed_text        TEXT NOT NULL,
    status           TEXT DEFAULT 'active',   -- active | awaiting_search | done
    site             TEXT DEFAULT 'inpass',   -- inpass | ppubs -- which portal this conversation searches
    turn_count       INTEGER DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    conversation_id  TEXT NOT NULL,
    cluster_id       TEXT NOT NULL,           -- e.g. "C1", "C1.2"
    parent_id        TEXT,                    -- NULL for top-level
    label            TEXT,
    key_terms        TEXT,                    -- json list[str]
    score            REAL DEFAULT 0.0,
    status           TEXT DEFAULT 'live',      -- live | confirmed | pruned
    last_touched_turn INTEGER,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS turns (
    conversation_id  TEXT NOT NULL,
    turn_no          INTEGER NOT NULL,
    kind             TEXT NOT NULL,   -- question | hint | document_reaction | system_note
    text             TEXT NOT NULL,
    cluster_refs     TEXT,            -- json list[str]
    created_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn_no)
);

CREATE TABLE IF NOT EXISTS documents (
    application_number TEXT PRIMARY KEY,
    title              TEXT,
    application_date   TEXT,
    status             TEXT,
    site               TEXT DEFAULT 'inpass',  -- inpass | ppubs -- which portal this was found on;
                                                -- for ppubs, `status` doubles as the sub-collection
                                                -- (USPAT/US-PGPUB/USOCR) fetch_detail needs as `source`
    abstract           TEXT,
    applicant          TEXT,
    ipc                TEXT,
    detail_fetched     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cluster_docs (
    conversation_id     TEXT NOT NULL,
    cluster_id          TEXT NOT NULL,
    application_number  TEXT NOT NULL,
    source              TEXT DEFAULT 'bootstrap',  -- bootstrap | search
    PRIMARY KEY (conversation_id, cluster_id, application_number)
);

CREATE TABLE IF NOT EXISTS doc_verdicts (
    conversation_id     TEXT NOT NULL,
    application_number  TEXT NOT NULL,
    verdict             TEXT NOT NULL,   -- confirmed | partial | rejected
    created_at          TEXT NOT NULL,
    PRIMARY KEY (conversation_id, application_number)
);

CREATE TABLE IF NOT EXISTS pending_searches (
    conversation_id  TEXT NOT NULL,
    qid              TEXT NOT NULL,
    rows             TEXT NOT NULL,   -- json list[[field,text],...]
    target_cluster_id TEXT,
    state            TEXT DEFAULT 'pending',  -- pending | done
    created_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, qid)
);

CREATE INDEX IF NOT EXISTS idx_cluster_docs_app ON cluster_docs(application_number);

-- Examiner interview (core/examiner). The full message history is kept;
-- `archived` marks turns folded into the running summary by compaction, so
-- the transcript is never lost even when the model no longer sees it.
CREATE TABLE IF NOT EXISTS examiner_messages (
    conversation_id  TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    role             TEXT NOT NULL,            -- user | assistant
    kind             TEXT DEFAULT 'chat',       -- chat | control (hidden prompt to the model)
    content          TEXT NOT NULL,
    archived         INTEGER DEFAULT 0,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);

-- One cell per (claim element, reference). Written by the initial chart
-- pass, revised by the interview with a reason.
CREATE TABLE IF NOT EXISTS chart_cells (
    conversation_id     TEXT NOT NULL,
    element_id          TEXT NOT NULL,
    application_number  TEXT NOT NULL,
    status              TEXT NOT NULL,   -- disclosed | partial | absent | unverified
    quote               TEXT,
    reasoning           TEXT,
    source              TEXT DEFAULT 'initial',   -- initial | interview
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (conversation_id, element_id, application_number)
);

-- Per-reference summary from the chart pass (what it does, how close).
CREATE TABLE IF NOT EXISTS chart_refs (
    conversation_id     TEXT NOT NULL,
    application_number  TEXT NOT NULL,
    summary             TEXT,
    closeness           TEXT,            -- very_close | related | distant
    PRIMARY KEY (conversation_id, application_number)
);

CREATE TABLE IF NOT EXISTS inventor_facts (
    conversation_id  TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    fact             TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);

-- Small JSON values per conversation: claims, open issues, distinguishing
-- features, suggested searches, history summary, disclosure path.
CREATE TABLE IF NOT EXISTS examiner_kv (
    conversation_id  TEXT NOT NULL,
    key              TEXT NOT NULL,
    value            TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    PRIMARY KEY (conversation_id, key)
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Adds columns SCHEMA gained after a database already existed on
        disk -- CREATE TABLE IF NOT EXISTS only helps a brand-new file; a
        real conversations.db with real conversations in it (InPASS-only,
        from before ppubs existed) needs these added in place, defaulting
        existing rows to 'inpass' since that's factually what they are.
        """
        with self.tx() as c:
            for table, col, ddl in (
                ("conversations", "site", "TEXT DEFAULT 'inpass'"),
                ("documents", "site", "TEXT DEFAULT 'inpass'"),
            ):
                cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
                if col not in cols:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    @contextmanager
    def tx(self):
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # ------------------------------------------------------------ conversations

    def create_conversation(self, conversation_id: str, seed_text: str,
                            site: str = "inpass") -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO conversations"
                "(conversation_id, seed_text, site, created_at, updated_at)"
                " VALUES (?, ?, ?, datetime('now'), datetime('now'))",
                (conversation_id, seed_text, site))

    def get_conversation(self, conversation_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id=?",
            (conversation_id,)).fetchone()
        return dict(row) if row else None

    def set_conversation_status(self, conversation_id: str, status: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE conversations SET status=?, updated_at=datetime('now')"
                " WHERE conversation_id=?", (status, conversation_id))

    def bump_turn_count(self, conversation_id: str) -> int:
        with self.tx() as c:
            c.execute(
                "UPDATE conversations SET turn_count=turn_count+1,"
                " updated_at=datetime('now') WHERE conversation_id=?",
                (conversation_id,))
            return c.execute(
                "SELECT turn_count FROM conversations WHERE conversation_id=?",
                (conversation_id,)).fetchone()[0]

    # ------------------------------------------------------------------ clusters

    def add_cluster(self, conversation_id: str, cluster_id: str, label: str,
                    key_terms: list[str], parent_id: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO clusters"
                "(conversation_id, cluster_id, parent_id, label, key_terms,"
                " score, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, 0.0, 'live', datetime('now'))",
                (conversation_id, cluster_id, parent_id, label,
                 json.dumps(key_terms)))

    def get_clusters(self, conversation_id: str, status: str | None = None,
                     parent_id: str | None = "__any__") -> list[dict]:
        sql = "SELECT * FROM clusters WHERE conversation_id=?"
        args: list = [conversation_id]
        if status:
            sql += " AND status=?"
            args.append(status)
        if parent_id != "__any__":
            if parent_id is None:
                sql += " AND parent_id IS NULL"
            else:
                sql += " AND parent_id=?"
                args.append(parent_id)
        rows = [dict(r) for r in self.conn.execute(sql, args)]
        for r in rows:
            r["key_terms"] = json.loads(r["key_terms"] or "[]")
        return rows

    def get_cluster(self, conversation_id: str, cluster_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM clusters WHERE conversation_id=? AND cluster_id=?",
            (conversation_id, cluster_id)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["key_terms"] = json.loads(d["key_terms"] or "[]")
        return d

    def update_cluster_score(self, conversation_id: str, cluster_id: str,
                             delta: float, cap: float = 5.0, turn_no: int = None) -> float:
        with self.tx() as c:
            cur = c.execute(
                "SELECT score FROM clusters WHERE conversation_id=? AND cluster_id=?",
                (conversation_id, cluster_id)).fetchone()
            if not cur:
                return 0.0
            new_score = max(-cap, min(cap, cur["score"] + delta))
            c.execute(
                "UPDATE clusters SET score=?, last_touched_turn=?"
                " WHERE conversation_id=? AND cluster_id=?",
                (new_score, turn_no, conversation_id, cluster_id))
            return new_score

    def set_cluster_status(self, conversation_id: str, cluster_id: str, status: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE clusters SET status=? WHERE conversation_id=? AND cluster_id=?",
                (status, conversation_id, cluster_id))

    # --------------------------------------------------------------------- turns

    def add_turn(self, conversation_id: str, turn_no: int, kind: str, text: str,
                cluster_refs: list[str] | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO turns"
                "(conversation_id, turn_no, kind, text, cluster_refs, created_at)"
                " VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (conversation_id, turn_no, kind, text,
                 json.dumps(cluster_refs or [])))

    def get_turns(self, conversation_id: str) -> list[dict]:
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM turns WHERE conversation_id=? ORDER BY turn_no",
            (conversation_id,))]
        for r in rows:
            r["cluster_refs"] = json.loads(r["cluster_refs"] or "[]")
        return rows

    # ----------------------------------------------------------------- documents

    def upsert_documents(self, hits, site: str = "inpass") -> None:
        """hits: objects/dicts with application_number, title, application_date, status."""
        with self.tx() as c:
            for h in hits:
                get = (h.get if isinstance(h, dict) else
                      lambda k, hh=h: getattr(hh, k, None))
                c.execute(
                    "INSERT INTO documents(application_number, title,"
                    " application_date, status, site) VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT(application_number) DO NOTHING",
                    (get("application_number"), get("title"),
                     get("application_date"), get("status"), site))

    def save_detail(self, detail: dict) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE documents SET abstract=?, applicant=?, ipc=?,"
                " detail_fetched=1 WHERE application_number=?",
                (detail.get("abstract"), detail.get("applicant"),
                 detail.get("ipc"), detail["application_number"]))

    def get_document(self, application_number: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE application_number=?",
            (application_number,)).fetchone()
        return dict(row) if row else None

    def docs_needing_detail(self, application_numbers: list[str] = None) -> list[str]:
        if application_numbers is not None:
            placeholders = ",".join("?" * len(application_numbers))
            if not application_numbers:
                return []
            sql = (f"SELECT application_number FROM documents WHERE detail_fetched=0"
                  f" AND application_number IN ({placeholders})")
            return [r[0] for r in self.conn.execute(sql, application_numbers)]
        return [r[0] for r in self.conn.execute(
            "SELECT application_number FROM documents WHERE detail_fetched=0")]

    # -------------------------------------------------------------- cluster_docs

    def link_docs_to_cluster(self, conversation_id: str, cluster_id: str,
                             application_numbers: list[str], source: str = "bootstrap") -> None:
        with self.tx() as c:
            c.executemany(
                "INSERT OR IGNORE INTO cluster_docs"
                "(conversation_id, cluster_id, application_number, source)"
                " VALUES (?, ?, ?, ?)",
                [(conversation_id, cluster_id, appno, source)
                 for appno in application_numbers])

    def cluster_doc_counts(self, conversation_id: str) -> dict[str, int]:
        """How many real documents support each cluster -- used to break ties
        when scores are equal, which is the case for EVERY cluster at the
        start of a conversation. A better-evidenced branch is a more
        informative first question than an arbitrary tie-break order."""
        rows = self.conn.execute(
            "SELECT cluster_id, COUNT(*) as n FROM cluster_docs"
            " WHERE conversation_id=? GROUP BY cluster_id", (conversation_id,))
        return {r["cluster_id"]: r["n"] for r in rows}

    def docs_for_cluster(self, conversation_id: str, cluster_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT d.* FROM documents d JOIN cluster_docs cd"
            " ON d.application_number = cd.application_number"
            " WHERE cd.conversation_id=? AND cd.cluster_id=?",
            (conversation_id, cluster_id)).fetchall()
        return [dict(r) for r in rows]

    def all_conversation_docs(self, conversation_id: str) -> list[dict]:
        """Every document any cluster in this conversation has ever touched,
        with the max score of any live cluster that found it."""
        rows = self.conn.execute("""
            SELECT d.*, MAX(cl.score) as best_cluster_score,
                   GROUP_CONCAT(DISTINCT cd.cluster_id) as found_by
              FROM documents d
              JOIN cluster_docs cd ON d.application_number = cd.application_number
              JOIN clusters cl ON cl.conversation_id = cd.conversation_id
                               AND cl.cluster_id = cd.cluster_id
             WHERE cd.conversation_id = ?
             GROUP BY d.application_number
        """, (conversation_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_documents(self) -> list[dict]:
        """Every harvested document with text, from every conversation."""
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM documents WHERE abstract IS NOT NULL AND abstract != ''")]

    def documents_with_cluster_info(self, conversation_id: str) -> list[dict]:
        """Every document this conversation has touched, each with the full
        list of (cluster_id, status, score) for every cluster that found it
        -- not just a single aggregated score. Building the final shortlist
        needs to know whether a document EVER touched a still-live or
        confirmed cluster, or only ever matched clusters that were pruned,
        which a single MAX(score) can't distinguish (a pruned cluster's
        score sits at a negative value, but so can a live one that's merely
        leaning negative so far)."""
        rows = self.conn.execute("""
            SELECT d.*, cd.cluster_id, cl.status as cluster_status, cl.score as cluster_score
              FROM documents d
              JOIN cluster_docs cd ON d.application_number = cd.application_number
              JOIN clusters cl ON cl.conversation_id = cd.conversation_id
                               AND cl.cluster_id = cd.cluster_id
             WHERE cd.conversation_id = ?
        """, (conversation_id,)).fetchall()

        by_doc: dict[str, dict] = {}
        for r in rows:
            d = dict(r)
            appno = d["application_number"]
            if appno not in by_doc:
                doc_fields = {k: v for k, v in d.items()
                             if k not in ("cluster_id", "cluster_status", "cluster_score")}
                by_doc[appno] = {**doc_fields, "clusters": []}
            by_doc[appno]["clusters"].append({
                "cluster_id": d["cluster_id"], "status": d["cluster_status"],
                "score": d["cluster_score"]})
        return list(by_doc.values())

    # ----------------------------------------------------------------- verdicts

    def set_verdict(self, conversation_id: str, application_number: str, verdict: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO doc_verdicts"
                "(conversation_id, application_number, verdict, created_at)"
                " VALUES (?, ?, ?, datetime('now'))",
                (conversation_id, application_number, verdict))

    def get_verdicts(self, conversation_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT application_number, verdict FROM doc_verdicts WHERE conversation_id=?",
            (conversation_id,))
        return {r["application_number"]: r["verdict"] for r in rows}

    # ---------------------------------------------------------- pending searches

    def add_pending_search(self, conversation_id: str, qid: str,
                           rows: list[tuple[str, str]], target_cluster_id: str | None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO pending_searches"
                "(conversation_id, qid, rows, target_cluster_id, state, created_at)"
                " VALUES (?, ?, ?, ?, 'pending', datetime('now'))",
                (conversation_id, qid, json.dumps(rows), target_cluster_id))

    def get_pending_searches(self, conversation_id: str, state: str = "pending") -> list[dict]:
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM pending_searches WHERE conversation_id=? AND state=?",
            (conversation_id, state))]
        for r in rows:
            r["rows"] = json.loads(r["rows"])
        return rows

    def mark_search_done(self, conversation_id: str, qid: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE pending_searches SET state='done'"
                " WHERE conversation_id=? AND qid=?", (conversation_id, qid))

    # ------------------------------------------------------------- examiner kv

    def kv_get(self, conversation_id: str, key: str, default=None):
        row = self.conn.execute(
            "SELECT value FROM examiner_kv WHERE conversation_id=? AND key=?",
            (conversation_id, key)).fetchone()
        return json.loads(row["value"]) if row else default

    def kv_set(self, conversation_id: str, key: str, value) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO examiner_kv(conversation_id, key, value, updated_at)"
                " VALUES (?, ?, ?, datetime('now'))",
                (conversation_id, key, json.dumps(value)))

    # ------------------------------------------------------- examiner messages

    def add_message(self, conversation_id: str, role: str, content: str,
                    kind: str = "chat") -> int:
        with self.tx() as c:
            seq = c.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM examiner_messages"
                " WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
            c.execute(
                "INSERT INTO examiner_messages"
                "(conversation_id, seq, role, kind, content, created_at)"
                " VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (conversation_id, seq, role, kind, content))
        return seq

    def get_messages(self, conversation_id: str, include_archived: bool = False) -> list[dict]:
        sql = "SELECT * FROM examiner_messages WHERE conversation_id=?"
        if not include_archived:
            sql += " AND archived=0"
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY seq", (conversation_id,))]

    def archive_messages_through(self, conversation_id: str, seq: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE examiner_messages SET archived=1"
                      " WHERE conversation_id=? AND seq<=?", (conversation_id, seq))

    # -------------------------------------------------------------- claim chart

    def set_chart_cell(self, conversation_id: str, element_id: str, application_number: str,
                       status: str, quote: str = "", reasoning: str = "",
                       source: str = "initial") -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO chart_cells(conversation_id, element_id,"
                " application_number, status, quote, reasoning, source, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (conversation_id, element_id, application_number, status,
                 quote, reasoning, source))

    def get_chart_cells(self, conversation_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM chart_cells WHERE conversation_id=?"
            " ORDER BY application_number, element_id", (conversation_id,))]

    def set_chart_ref(self, conversation_id: str, application_number: str,
                      summary: str, closeness: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO chart_refs(conversation_id, application_number,"
                " summary, closeness) VALUES (?, ?, ?, ?)",
                (conversation_id, application_number, summary, closeness))

    def get_chart_refs(self, conversation_id: str) -> dict[str, dict]:
        return {r["application_number"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM chart_refs WHERE conversation_id=?", (conversation_id,))}

    def clear_chart(self, conversation_id: str) -> None:
        """Claims changed -- every cell was judged against the old elements."""
        with self.tx() as c:
            c.execute("DELETE FROM chart_cells WHERE conversation_id=?", (conversation_id,))
            c.execute("DELETE FROM chart_refs WHERE conversation_id=?", (conversation_id,))

    # ----------------------------------------------------------- inventor facts

    def add_fact(self, conversation_id: str, fact: str) -> None:
        with self.tx() as c:
            seq = c.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM inventor_facts"
                " WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
            c.execute(
                "INSERT INTO inventor_facts(conversation_id, seq, fact, created_at)"
                " VALUES (?, ?, ?, datetime('now'))", (conversation_id, seq, fact))

    def get_facts(self, conversation_id: str) -> list[str]:
        return [r["fact"] for r in self.conn.execute(
            "SELECT fact FROM inventor_facts WHERE conversation_id=? ORDER BY seq",
            (conversation_id,))]
