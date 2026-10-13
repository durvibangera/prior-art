"""The examiner interview: one continuous conversation with full memory.

Replaces the old branch-scoring dialogue. The model sees the whole
disclosure, the draft claims, the claim chart and the closest references in
depth, plus the entire conversation so far -- and is asked to behave like a
senior patent attorney interviewing the inventor, one deep question at a
time.

Two calls per turn:
  1. the reply the inventor sees (thinking on, streamed)
  2. a guided-JSON "case file update" (thinking off) that extracts new
     facts, distinguishing features, justified chart corrections, open
     issues and follow-up search ideas -- stored, and fed back into the
     next turn's system prompt so the examiner always works from the
     current state of the case.
"""
from __future__ import annotations

import sys
import textwrap

from ..llm import LLMClient, LLMUnavailable
from .chart import STATUSES, assess, closest_refs, format_chart
from .claims import format_claims
from .context import Invention, Reference, estimate_tokens
from .searches import reviewed_keys, search_key

EXAMINER_SYSTEM = """You are a senior patent attorney with deep hands-on expertise in
the technical field of this invention, conducting a confidential pre-filing
interview with one of its inventors. You have read the inventor's
disclosure, the draft claims, and the closest published prior art found so
far, and you have a claim chart comparing them.

YOUR JOB
1. Establish precisely what this invention does that the prior art does not.
2. Test whether each difference is real, technical and non-obvious -- the
   way an Indian Patent Office examiner, and later an opponent, would.
3. Collect what the drafting attorney will need: the technical effect, the
   alternatives considered, evidence and measurements, failure cases.

LEGAL FRAME (Patents Act 1970, India)
- Novelty, s.2(1)(j): a claim is anticipated if ONE document discloses every
  element of it.
- Inventive step, s.2(1)(ja): a claim fails if a person skilled in the art
  would arrive at it by combining known documents or common general
  knowledge; it needs a technical advance or economic significance.
- Section 3(k): mathematical methods, algorithms and computer programmes per
  se are not patentable. Under the CRI Guidelines a claim survives when it is
  a technical solution with a technical effect (e.g. reliable detection at
  low SNR, lower latency or compute on given hardware, a specific
  sensor-to-decision processing chain) -- not the algorithm on its own.
- Publishable (a new scientific result) is a lower and different bar from
  patentable.

HOW YOU INTERVIEW
- Ask ONE question per message. Make it the question an expert examiner
  would actually ask: specific, technical, about mechanism and reasons. The
  depth expected: why this technique rather than the alternative used in a
  named reference; exactly what one stage hands to the next (dimensions,
  pooling, normalisation, windowing); what changes if a stage is removed or
  swapped; how a decision is made across time; what was measured, against
  which baseline, under which conditions; where it fails.
- Ground questions in the documents. Name the reference as [REF number] and
  quote or closely paraphrase the passage that matters, then ask how the
  invention differs from it.
- Priority: first the claim elements the chart shows as disclosed or partial
  in close references (real novelty threats); then likely inventive-step
  combinations; then the 3(k) technical-effect case; then gaps in the
  disclosure an examiner would exploit.
- Read each answer carefully. Start your reply by saying, in one or two
  sentences, what the answer establishes -- plainly: a genuine distinction,
  a weak one (and why an examiner would call it obvious), or not a
  distinction. Then follow up on it or move to the next issue.
- Never re-ask anything the inventor has already answered or that the
  disclosure already states. Use what you know. If an answer is vague,
  press once, concretely, then move on.
- If the inventor asks you something -- what a term means, what a reference
  actually does, whether something is patentable -- answer it properly from
  the documents and your expertise, then continue the interview.
- If the inventor declines or does not know, accept it, note it, move on.
- Every five or six exchanges, give a two-line interim read of where each
  claim stands.
- Never invent what a document says. If the documents do not say, say so.
- This interview is internal and confidential, so the inventor can be fully
  candid. Never suggest checking details in public AI tools or posting them.
- Talk like an experienced attorney to an engineer: direct, precise,
  collegial. Plain paragraphs, no headings. Under about 200 words per
  message unless you are answering a question that needs more."""

STATE_SYSTEM = """You keep the case file for a confidential patent interview. You
are given the current case file and the latest exchange between the
examiner and the inventor. Return ONLY what this exchange changes.

- new_facts: technical facts about the invention the inventor stated in THIS
  exchange, one atomic fact each, specific (numbers, techniques, data). Do
  not repeat facts already in the case file. Empty if none.
- distinguishing_features: features newly identified in this exchange that
  separate the invention from specific references. `claims`: affected claim
  ids. `against`: application numbers it distinguishes from. `strength`:
  strong (technical, non-obvious, with an effect), moderate, or weak (a
  routine design choice). `why`: one sentence.
- chart_updates: ONLY when the inventor showed that the chart misread a
  reference -- i.e. the reference text itself does not support the current
  status. The inventor merely saying "we do it differently" is NOT a chart
  update; it is a distinguishing feature. Usually empty.
- open_issues: the complete, updated list of unresolved questions that
  matter for patentability (at most 8), most important first.
- suggested_searches: only if this exchange surfaced a technical concept the
  existing prior art may not cover. Each search is 2 or 3 rows; each row is a
  list of single generic words that are synonyms or word-forms of ONE
  concept (e.g. ["WAVELET", "WAVELETS"]). Never combine two of the
  invention's distinguishing features in one search. Usually empty."""

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "new_facts": {"type": "array", "items": {"type": "string"}},
        "distinguishing_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "claims": {"type": "array", "items": {"type": "string"}},
                    "against": {"type": "array", "items": {"type": "string"}},
                    "strength": {"type": "string", "enum": ["strong", "moderate", "weak"]},
                    "why": {"type": "string"},
                },
                "required": ["feature", "claims", "against", "strength", "why"],
            },
        },
        "chart_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string"},
                    "application_number": {"type": "string"},
                    "status": {"type": "string", "enum": list(STATUSES)},
                    "reason": {"type": "string"},
                },
                "required": ["element_id", "application_number", "status", "reason"],
            },
        },
        "open_issues": {"type": "array", "items": {"type": "string"}},
        "suggested_searches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                },
                "required": ["purpose", "rows"],
            },
        },
    },
    "required": ["new_facts", "distinguishing_features", "chart_updates",
                 "open_issues", "suggested_searches"],
}

SUMMARY_SYSTEM = """Summarise this part of a confidential patent interview for the
examiner's own notes. Keep every technical fact the inventor gave, every
distinction argued and how strong it looked, every question asked and its
answer in one line, and anything left unresolved. Dense prose, no
pleasantries. At most 600 words."""

KICKOFF = ("[Session start -- not from the inventor] Open the interview. In three or "
           "four sentences tell the inventor what you have read and the two or three "
           "biggest patentability risks you see in the chart, then ask your first question.")
RESUME = ("[Session resumed -- not from the inventor] In one or two sentences recap "
          "where the interview stands, then ask the next question.")

HELP = """commands:
  /chart     claim chart (elements x closest references)
  /claims    the draft claims
  /refs      closest references with what each does
  /facts     facts established so far
  /features  distinguishing features identified so far
  /issues    open issues
  /search    review follow-up portal searches the examiner suggested
  /report    write the patentability report now
  /quit      stop (everything is saved; run again to resume)
anything else is your answer or your question to the examiner."""

RECENT_MESSAGES_KEPT = 20
DIGEST_REFS = 15


class Examiner:
    def __init__(self, store, conversation_id: str, llm: LLMClient, invention: Invention,
                 claims: list[dict], refs: list[Reference], cfg: dict | None = None):
        self.store = store
        self.cid = conversation_id
        self.llm = llm
        self.invention = invention
        self.claims = claims
        self.refs = {r.application_number: r for r in refs}
        ex_cfg = (cfg or {}).get("examiner", {})
        self.history_budget = ex_cfg.get("history_token_budget", 150_000)

    # -------------------------------------------------------------- state

    def _cells(self) -> list[dict]:
        return self.store.get_chart_cells(self.cid)

    def _ref_meta(self) -> dict[str, dict]:
        return self.store.get_chart_refs(self.cid)

    def closest(self, n: int = DIGEST_REFS) -> list[str]:
        return [a for a in closest_refs(self._cells(), self._ref_meta(), n=n * 2)
                if a in self.refs][:n]

    def _chart_section(self) -> str:
        cells = self._cells()
        verdicts = assess(self.claims, cells)
        meta = self._ref_meta()
        lines = []
        for claim in self.claims:
            v = verdicts[claim["id"]]
            extra = ""
            if v["anticipating"]:
                extra = f" by {', '.join(v['anticipating'])}"
            elif v["pairs"]:
                extra = " by e.g. " + " + ".join(v["pairs"][0])
            lines.append(f"{claim['id']}: {v['status'].replace('_', ' ')}{extra}")
        lines.append("")
        top = set(self.closest())
        for c in cells:
            if c["status"] in ("disclosed", "partial") and c["application_number"] in top:
                q = f' "{c["quote"]}"' if c["quote"] else ""
                lines.append(f"{c['element_id']} vs {c['application_number']}: {c['status']}{q} -- "
                             f"{c['reasoning']}")
        others = [f"- {a} ({m['closeness']}): {m['summary']}"
                  for a, m in meta.items() if a not in top and m.get("closeness") != "distant"]
        if others:
            lines.append("\nOther charted references (not shown in full):")
            lines.extend(others[:25])
        return "\n".join(lines)

    def system_prompt(self) -> str:
        """Stable material first (so the server's prefix cache can reuse
        it), the case file that changes each turn last."""
        digests = "\n\n".join(self.refs[a].digest() + (
            f"\nChart summary: {self._ref_meta().get(a, {}).get('summary', '')}")
            for a in self.closest())
        facts = self.store.get_facts(self.cid)
        features = self.store.kv_get(self.cid, "distinguishing_features", [])
        issues = self.store.kv_get(self.cid, "open_issues", [])
        summary = self.store.kv_get(self.cid, "history_summary", "")
        parts = [
            EXAMINER_SYSTEM,
            f"=== INVENTOR'S DISCLOSURE ===\n{self.invention.text()}",
            f"=== DRAFT CLAIMS (* = distinguishing element) ===\n{format_claims(self.claims)}",
            f"=== CLOSEST PRIOR ART (read in full) ===\n{digests or '(none charted yet)'}",
            f"=== CLAIM CHART ===\n{self._chart_section()}",
            "=== CASE FILE ===\nFacts established by the inventor:\n"
            + ("\n".join(f"- {f}" for f in facts) or "- none yet")
            + "\n\nDistinguishing features identified:\n"
            + ("\n".join(f"- [{d['strength']}] {d['feature']} (vs {', '.join(d['against']) or 'general art'})"
                         for d in features) or "- none yet")
            + "\n\nOpen issues:\n" + ("\n".join(f"- {i}" for i in issues) or "- none recorded"),
        ]
        if summary:
            parts.append(f"=== EARLIER PART OF THIS INTERVIEW (summary) ===\n{summary}")
        return "\n\n".join(parts)

    def _history(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]}
                for m in self.store.get_messages(self.cid)]

    # -------------------------------------------------------------- turns

    def _reply(self, history: list[dict]) -> str:
        print("\nexaminer: ", end="", flush=True)

        def emit(piece: str):
            sys.stdout.write(piece)
            sys.stdout.flush()

        reply = self.llm.converse([{"role": "system", "content": self.system_prompt()}] + history,
                                  thinking=True, max_tokens=6000, on_token=emit)
        print("\n")
        return reply.text

    def _update_case_file(self, question: str, answer: str, reply: str) -> str:
        """Second call per turn. Returns a one-line note of what changed."""
        case = (f"DRAFT CLAIMS\n{format_claims(self.claims)}\n\n"
                f"KNOWN FACTS\n" + "\n".join(f"- {f}" for f in self.store.get_facts(self.cid))
                + f"\n\nOPEN ISSUES\n" + "\n".join(f"- {i}" for i in self.store.kv_get(self.cid, "open_issues", []))
                + f"\n\nCHART (relevant cells)\n{self._chart_section()}")
        exchange = (f"EXAMINER ASKED:\n{question}\n\nINVENTOR ANSWERED:\n{answer}\n\n"
                    f"EXAMINER REPLIED:\n{reply}")
        try:
            upd = self.llm.converse_json(
                [{"role": "system", "content": STATE_SYSTEM},
                 {"role": "user", "content": f"{case}\n\n=== LATEST EXCHANGE ===\n{exchange}"}],
                STATE_SCHEMA, thinking=False, max_tokens=3000)
        except LLMUnavailable as exc:
            return f"(case file not updated: {exc})"
        return self.apply_update(upd)

    def apply_update(self, upd: dict) -> str:
        notes = []
        known = set(self.store.get_facts(self.cid))
        new_facts = []
        for f in (x.strip() for x in upd.get("new_facts") or []):
            if f and f not in known:
                known.add(f)
                new_facts.append(f)
        for f in new_facts:
            self.store.add_fact(self.cid, f)
        if new_facts:
            notes.append(f"+{len(new_facts)} fact(s)")

        existing = self.store.kv_get(self.cid, "distinguishing_features", [])
        known_feats = {d["feature"].strip().lower() for d in existing}
        feats = []
        for d in upd.get("distinguishing_features") or []:
            name = (d.get("feature") or "").strip().lower()
            if name and name not in known_feats:
                known_feats.add(name)
                feats.append(d)
        if feats:
            self.store.kv_set(self.cid, "distinguishing_features", existing + feats)
            notes.append(f"+{len(feats)} distinguishing feature(s)")

        cells = {(c["element_id"], c["application_number"]): c for c in self._cells()}
        changed = 0
        for u in upd.get("chart_updates") or []:
            key = (u.get("element_id"), u.get("application_number"))
            if key in cells and u.get("status") in STATUSES and u["status"] != cells[key]["status"]:
                old = cells[key]
                self.store.set_chart_cell(self.cid, key[0], key[1], u["status"], old["quote"],
                                          f"[revised in interview from {old['status']}] {u.get('reason', '')}",
                                          source="interview")
                changed += 1
        if changed:
            notes.append(f"{changed} chart cell(s) revised")

        if upd.get("open_issues") is not None:
            self.store.kv_set(self.cid, "open_issues", [i for i in upd["open_issues"] if i.strip()][:8])

        pending = self.store.kv_get(self.cid, "suggested_searches", [])
        seen = {search_key(s) for s in pending} | reviewed_keys(self.store, self.cid)
        searches = []
        for s in upd.get("suggested_searches") or []:
            if s.get("rows") and search_key(s) not in seen:
                seen.add(search_key(s))
                searches.append(s)
        if searches:
            self.store.kv_set(self.cid, "suggested_searches", pending + searches)
            notes.append(f"{len(searches)} search idea(s) -- /search to review")
        return ", ".join(notes)

    def _maybe_compact(self) -> None:
        msgs = self.store.get_messages(self.cid)
        total = estimate_tokens(self.system_prompt()) + sum(estimate_tokens(m["content"]) for m in msgs)
        if total <= self.history_budget or len(msgs) <= RECENT_MESSAGES_KEPT:
            return
        old = msgs[:-RECENT_MESSAGES_KEPT]
        prior = self.store.kv_get(self.cid, "history_summary", "")
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in old if m["kind"] == "chat")
        try:
            reply = self.llm.converse(
                [{"role": "system", "content": SUMMARY_SYSTEM},
                 {"role": "user", "content": (f"EARLIER SUMMARY:\n{prior}\n\n" if prior else "")
                  + f"TRANSCRIPT TO SUMMARISE:\n{transcript}"}],
                thinking=False, max_tokens=1500)
        except LLMUnavailable:
            return
        self.store.kv_set(self.cid, "history_summary", reply.text)
        self.store.archive_messages_through(self.cid, old[-1]["seq"])
        # The kept window must open with a user turn for the chat template.
        kept = self.store.get_messages(self.cid)
        if kept and kept[0]["role"] == "assistant":
            self.store.archive_messages_through(self.cid, kept[0]["seq"])

    def _last_question(self) -> str:
        for m in reversed(self.store.get_messages(self.cid)):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    def _send(self, content: str, kind: str = "chat") -> str | None:
        """Send one user message; persist it and the reply only on success."""
        history = self._history() + [{"role": "user", "content": content}]
        try:
            reply = self._reply(history)
        except LLMUnavailable as exc:
            print(f"\n  LLM call failed ({exc}). Nothing was lost -- say it again, "
                  f"or /quit and resume once the intranet endpoint is reachable.")
            return None
        self.store.add_message(self.cid, "user", content, kind=kind)
        self.store.add_message(self.cid, "assistant", reply)
        return reply

    # -------------------------------------------------------------- display

    def show(self, what: str) -> None:
        cells = self._cells()
        if what == "chart":
            print(format_chart(self.claims, cells, self.closest(12)))
        elif what == "claims":
            print(format_claims(self.claims))
        elif what == "refs":
            meta = self._ref_meta()
            for i, a in enumerate(self.closest(), 1):
                m = meta.get(a, {})
                print(f"R{i:<2} {a}  [{m.get('closeness', '?')}]  {self.refs[a].text.title[:70]}")
                print(textwrap.indent(textwrap.fill(m.get("summary", ""), 96), "      "))
        elif what == "facts":
            for f in self.store.get_facts(self.cid) or ["(none yet)"]:
                print(f"  - {f}")
        elif what == "features":
            for d in self.store.kv_get(self.cid, "distinguishing_features", []) or [{"feature": "(none yet)", "strength": "-", "against": [], "why": ""}]:
                print(f"  [{d['strength']}] {d['feature']}  vs {', '.join(d['against']) or '-'}")
                if d.get("why"):
                    print(f"           {d['why']}")
        elif what == "issues":
            for i in self.store.kv_get(self.cid, "open_issues", []) or ["(none recorded)"]:
                print(f"  - {i}")

    # -------------------------------------------------------------- loop

    def run(self, on_search=None, on_report=None) -> None:
        print(HELP)
        if not self.store.get_messages(self.cid, include_archived=True):
            if self._send(KICKOFF, kind="control") is None:
                return
        else:
            self._send(RESUME, kind="control")

        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Everything is saved.")
                return
            if not text:
                continue
            if text.startswith("/"):
                cmd = text[1:].split()[0].lower()
                if cmd in ("quit", "exit", "stop"):
                    print("Stopped. Everything is saved -- run the same command to resume.")
                    return
                if cmd == "help":
                    print(HELP)
                elif cmd == "search" and on_search:
                    on_search()
                elif cmd == "report" and on_report:
                    on_report()
                elif cmd in ("chart", "claims", "refs", "facts", "features", "issues"):
                    self.show(cmd)
                else:
                    print(f"  unknown command /{cmd} -- /help lists them")
                continue

            question = self._last_question()
            reply = self._send(text)
            if reply is None:
                continue
            note = self._update_case_file(question, text, reply)
            if note:
                print(f"  [case file: {note}]")
            self._maybe_compact()
