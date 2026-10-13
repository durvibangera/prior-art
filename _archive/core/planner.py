"""Turn an invention profile into a list of portal queries.

Query shape is dictated by measured portal behaviour (PHASE0_FINDINGS.md):

  * A search box ORs the words inside it.
  * Matching is whole-word, so DRONE does not find DRONES or UAV.
  * Boxes are ANDed together, up to 16 of them.

So the efficient unit is one box per *facet*, holding every synonym and
morphological variant of that facet, with boxes ANDed.  Measured effect on the
core query: 34 hits with one term per facet, 109 with full OR-boxes.  The naive
form silently dropped 69% of the relevant set.

Multi-word phrases cannot live in a box -- "UNMANNED AERIAL" would be ORed
apart into UNMANNED or AERIAL -- so each active phrase becomes its own row.

Three axes:
  A  novelty         modality x platform x task, refined by method facets
  B  whitespace      method_core against any domain, platform dropped
  C  classification  IPC subgroups, independent of our keyword vocabulary

Axis C matters because A and B can only find art that shares our vocabulary.
IPC is the cross-check on our own blind spots.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..inpass.client import MAX_ROWS, SearchRow

# A query chaining more facets than this describes the invention too closely
# to be safe to submit to an external system.
MAX_FACETS_PER_QUERY = 3


@dataclass(frozen=True)
class PlannedQuery:
    qid: str
    axis: str
    tier: int
    facets: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]      # (field_code, text)
    ipc: str | None = None
    note: str = ""

    def to_rows(self) -> list[SearchRow]:
        out = [SearchRow("IC", self.ipc)] if self.ipc else []
        out += [SearchRow(f, t) for f, t in self.rows]
        return out

    def describe(self) -> str:
        parts = [f"IC:{self.ipc}"] if self.ipc else []
        parts += [f"{f}:{t}" for f, t in self.rows]
        return " AND ".join(parts)


class QueryPlanner:
    def __init__(self, profile: dict, *, field_code: str = "ABS",
                 use_phrases: bool = False, max_queries: int = 60):
        self.profile = profile
        self.facets = profile["facets"]
        self.field = field_code
        self.use_phrases = use_phrases
        self.max_queries = max_queries

    # ------------------------------------------------------------------ rows

    def _facet_rows(self, name: str) -> list[tuple[str, str]]:
        """One OR-box of single words, plus one row per phrase if enabled."""
        facet = self.facets[name]
        rows: list[tuple[str, str]] = []
        words = facet.get("words", [])
        if words:
            rows.append((self.field, " ".join(words)))
        if self.use_phrases:
            rows += [(self.field, p) for p in facet.get("phrases", [])]
        return rows

    def _make(self, qid, axis, tier, facets, *, ipc=None, note="") -> PlannedQuery | None:
        rows: list[tuple[str, str]] = []
        for f in facets:
            rows.extend(self._facet_rows(f))
        if not rows:
            return None
        if len(rows) + (1 if ipc else 0) > MAX_ROWS:
            # Drop phrase rows rather than emit an invalid query.
            rows = [r for f in facets for r in self._facet_rows(f)[:1]]
        return PlannedQuery(qid, axis, tier, tuple(facets), tuple(rows), ipc, note)

    # ------------------------------------------------------------------ axes

    def _axis_a(self) -> list[PlannedQuery]:
        axis = self.profile["axes"]["A_novelty"]
        anchors = list(axis["anchor_facets"])
        out = [self._make("A1-001", "A_novelty", 1, anchors,
                          note="Core sweep: every acoustic x UAV x task variant")]
        for i, refine in enumerate(axis["refine_facets"], 1):
            if refine not in self.facets:
                continue
            out.append(self._make(f"A2-{i:03d}", "A_novelty", 2,
                                  anchors[:2] + [refine],
                                  note=f"Acoustic UAV art using {refine}"))
        # Tier 3: the closest possible art -- core sweep plus the core
        # technique. Reads the core facet from axes.B_whitespace rather than
        # a hardcoded name, so this works for an auto-derived profile whose
        # core facet isn't literally called "method_core" (see PLAN.md 7b).
        core = self.profile.get("axes", {}).get("B_whitespace", {}).get("anchor_facets", [None])
        core = core[0] if core else None
        if core and core in self.facets:
            out.append(self._make("A3-001", "A_novelty", 3, anchors + [core],
                                  note=f"Closest art: application + {core}"))
        return [q for q in out if q]

    def _axis_b(self) -> list[PlannedQuery]:
        axis = self.profile["axes"]["B_whitespace"]
        core = axis["anchor_facets"][0]
        out = [self._make("B1-001", "B_whitespace", 1, [core],
                          note="Is wavelet scattering claimed anywhere at all?")]
        for i, refine in enumerate(axis["refine_facets"], 1):
            if refine not in self.facets:
                continue
            out.append(self._make(f"B2-{i:03d}", "B_whitespace", 2, [core, refine],
                                  note=f"Wavelet scattering applied with {refine}"))
        return [q for q in out if q]

    def _axis_c(self) -> list[PlannedQuery]:
        """IPC subgroups crossed with our two strongest facets.

        Bare section codes return zero (measured), so the profile must supply
        full symbols including subgroup.
        """
        out = []
        for i, entry in enumerate(self.profile.get("ipc_codes", []), 1):
            for j, facet in enumerate(("platform", "modality")):
                if facet not in self.facets:
                    continue
                out.append(self._make(f"C1-{i:02d}{j}", "C_classification", 2,
                                      [facet], ipc=entry["code"],
                                      note=entry.get("note", "")))
        return [q for q in out if q]

    # ----------------------------------------------------------------- build

    def build(self, axes: tuple[str, ...] = ("A", "B", "C")) -> list[PlannedQuery]:
        gen = {"A": self._axis_a, "B": self._axis_b, "C": self._axis_c}
        queries: list[PlannedQuery] = []
        for a in axes:
            queries.extend(gen[a]())

        seen, unique = set(), []
        for q in queries:
            key = (q.ipc, tuple(sorted(q.rows)))
            if key in seen:
                continue
            seen.add(key)
            self._assert_safe(q)
            unique.append(q)

        # Tier order, so an interrupted run still covers the broad sweep first.
        unique.sort(key=lambda q: (q.tier, q.axis, q.qid))
        return unique[:self.max_queries]

    def _assert_safe(self, q: PlannedQuery) -> None:
        """Refuse any query that reconstructs the inventive combination.

        A query submitted to an external portal is a disclosure, so this is a
        structural guard rather than advice. Generalised over the profile's
        OWN core facet rather than a hardcoded name, so this still holds for
        an auto-derived profile whose technique facets are named differently
        (see PLAN.md 7b) -- the property being protected is "don't chain the
        core technique with multiple other technique facets at once", not
        "don't mention these three specific facet names".
        """
        if len(q.facets) > MAX_FACETS_PER_QUERY + 1:
            raise ValueError(f"{q.qid} combines too many facets: {q.facets}")

        anchors = set(self.profile.get("axes", {}).get("A_novelty", {}).get("anchor_facets", []))
        core_list = self.profile.get("axes", {}).get("B_whitespace", {}).get("anchor_facets", [])
        core = core_list[0] if core_list else None
        non_anchor = set(q.facets) - anchors
        if core and core in non_anchor and len(non_anchor) >= 3:
            raise ValueError(f"{q.qid} chains the core technique with multiple "
                             f"other technique facets: {non_anchor}")
