"""The patentability report.

The per-claim novelty status at the top is computed in code from the chart
(chart.assess) and printed as-is -- the model writes the analysis around
it but cannot overrule it. Evidence (verified quotes) is appended from the
chart directly, so every claim the memo makes can be checked against a
real passage.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from ..llm import LLMClient
from .chart import assess, cells_by_ref
from .claims import format_claims
from .context import Invention

_STATUS_TEXT = {
    "anticipated": "ANTICIPATED -- one reference discloses every element",
    "inventive_step_risk": "INVENTIVE-STEP RISK -- two references together disclose every element",
    "all_elements_known": "ELEMENTS ALL KNOWN -- each element is disclosed somewhere (3+ references)",
    "partially_covered": "PARTLY COVERED -- some elements disclosed or partly disclosed",
    "clear": "NO CLOSE ART FOUND in the documents searched",
}

REPORT_SYSTEM = """You are a senior patent attorney writing an internal pre-filing
assessment for the inventors and their management. They will decide from it
whether to commission a formal prior-art search and filing from outside
counsel (cost roughly INR 30,000-50,000 per search) or to park the idea.

You are given the draft claims, a per-claim novelty status COMPUTED FROM THE
CLAIM CHART (treat it as fact -- do not contradict it; you may explain and
qualify it), the chart evidence, the facts and distinguishing features
established in the interview, and the open issues.

Write these sections, as Markdown with these exact headings:

## Bottom line
Three to five sentences: is there likely a patentable invention here, in
which claims, and should they spend on the formal search -- yes, yes with a
narrowed claim set, or park. Be direct.

## Novelty, claim by claim
For each claim: the computed status, the specific references and what they
disclose, and which element(s) carry novelty if any.

## Inventive step
Which combinations of references (or common general knowledge) an examiner
would use against each surviving claim, and how strong the distinguishing
features from the interview are against that attack.

## Section 3(k)
Whether the claims read as an algorithm per se or as a technical solution;
the technical effect to emphasise and how to frame the claims to show it.

## White space
Elements or combinations that nothing searched discloses, and whether that
white space is commercially and technically meaningful.

## Publishable vs patentable
Whether the work is publishable as research, and how a publication would
interact with filing (file first; publication becomes prior art).

## Recommended claim strategy
Concrete narrowing or amendments, using the distinguishing features found.

## Gaps the drafting attorney will ask about
What the disclosure or the interview has not yet established.

Rules: cite references as [application number]. Never claim a document says
something the evidence does not show. Plain, decisive professional prose."""


def _computed_table(claims: list[dict], verdicts: dict[str, dict]) -> str:
    lines = ["| Claim | Status | Closest art | Elements not found anywhere |",
             "|---|---|---|---|"]
    for c in claims:
        v = verdicts[c["id"]]
        if v["anticipating"]:
            art = ", ".join(v["anticipating"])
        elif v["pairs"]:
            art = "; ".join(" + ".join(p) for p in v["pairs"][:3])
        else:
            refs = sorted({a for e in v["required"] for a in v["element_refs"][e] + v["element_partial"][e]})
            art = ", ".join(refs[:4]) or "--"
        lines.append(f"| {c['id']} | {_STATUS_TEXT[v['status']]} | {art} | "
                     f"{', '.join(v['uncovered']) or '--'} |")
    return "\n".join(lines)


def _evidence(claims: list[dict], cells: list[dict], per_element: int = 3) -> str:
    by_ref = cells_by_ref(cells)
    lines = []
    for c in claims:
        for e in c["elements"]:
            rows = [(a, rc[e["id"]]) for a, rc in by_ref.items()
                    if e["id"] in rc and rc[e["id"]]["status"] in ("disclosed", "partial")]
            rows.sort(key=lambda t: t[1]["status"] != "disclosed")
            lines.append(f"\n**{e['id']}** {e['text']}")
            if not rows:
                lines.append("- nothing found")
            for a, cell in rows[:per_element]:
                q = f' -- "{cell["quote"]}"' if cell["quote"] else ""
                lines.append(f"- [{a}] {cell['status']}{q}  \n  {cell['reasoning']}")
    return "\n".join(lines)


def build_report(llm: LLMClient, store, conversation_id: str, invention: Invention,
                 claims: list[dict], out_dir: Path) -> Path:
    cells = store.get_chart_cells(conversation_id)
    verdicts = assess(claims, cells)
    meta = store.get_chart_refs(conversation_id)
    facts = store.get_facts(conversation_id)
    features = store.kv_get(conversation_id, "distinguishing_features", [])
    issues = store.kv_get(conversation_id, "open_issues", [])
    summary = store.kv_get(conversation_id, "history_summary", "")
    recent = [m for m in store.get_messages(conversation_id) if m["kind"] == "chat"][-30:]

    table = _computed_table(claims, verdicts)
    evidence = _evidence(claims, cells)
    refs_line = "\n".join(f"- [{a}] ({m['closeness']}) {m['summary']}" for a, m in meta.items()
                          if m.get("closeness") != "distant")
    user = "\n\n".join([
        f"=== DISCLOSURE ===\n{invention.text()}",
        f"=== DRAFT CLAIMS ===\n{format_claims(claims)}",
        f"=== COMPUTED NOVELTY STATUS (from the claim chart) ===\n{table}",
        f"=== CHART EVIDENCE ===\n{evidence}",
        f"=== CHARTED REFERENCES (non-distant) ===\n{refs_line or '(none)'}",
        "=== FACTS FROM THE INTERVIEW ===\n" + ("\n".join(f"- {f}" for f in facts) or "(none)"),
        "=== DISTINGUISHING FEATURES ===\n" + ("\n".join(
            f"- [{d['strength']}] {d['feature']} vs {', '.join(d['against']) or 'general art'}: {d['why']}"
            for d in features) or "(none)"),
        "=== OPEN ISSUES ===\n" + ("\n".join(f"- {i}" for i in issues) or "(none)"),
        (f"=== INTERVIEW SUMMARY ===\n{summary}" if summary else ""),
        "=== RECENT INTERVIEW ===\n" + "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent),
        "Write the assessment now.",
    ])
    print("Writing the report (this takes a minute)...")
    memo = llm.converse([{"role": "system", "content": REPORT_SYSTEM},
                         {"role": "user", "content": user}],
                        thinking=True, max_tokens=10000).text

    n_refs = len(meta)
    sites = sorted({(store.get_document(a) or {}).get("site") or "inpass" for a in meta})
    md = "\n\n".join([
        f"# Pre-filing patentability assessment -- {invention.title or conversation_id}",
        f"_Internal and confidential. Generated {date.today().isoformat()} from conversation "
        f"`{conversation_id}`; {n_refs} references charted ({', '.join(sites) or 'none'})._",
        "## Computed novelty status\n" + table,
        memo,
        "## Appendix A -- Draft claims\n```\n" + format_claims(claims) + "\n```",
        "## Appendix B -- Chart evidence (verified quotes)" + evidence,
        "## Limits of this assessment\nOnly documents harvested from InPASS (India) and USPTO "
        "were searched. Non-patent literature (IEEE, arXiv, journals) -- which the disclosure's own "
        "prior-art section shows is where this field publishes -- and CN/EP/JP patents were not. "
        "This is a screening assessment to decide whether a formal search is worth commissioning, "
        "not a novelty opinion.",
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.md"
    path.write_text(md)
    return path
