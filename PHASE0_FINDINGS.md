# Phase 0 — Measured Portal Semantics

**Date:** 21 September 2026
**Target:** `iprsearch.ipindia.gov.in/PublicSearch`
**Settings for all measurements:** Published, Publication Date, 01/01/2020–09/05/2026

Every number below is measured against the live portal, not inferred.

---

## 1. Headline

| Approach | Hits |
|---|---|
| One box: `ACOUSTIC DRONE DETECTION` | **48,979** |
| Three rows ANDed, one term each | **34** |
| Three rows ANDed, synonym OR-boxes | **109** |

The naive ANDed query returns 34 and looks authoritative. It misses **75 of 109
documents — 69% of the relevant set.** This is the failure mode the whole
design is built to avoid.

---

## 2. The single text box ORs its words

Proven by exact set arithmetic on a small, clean pair:

```
|WAVELET|                      =   482
|SCATTERING|                   =   663
|WAVELET AND SCATTERING|       =    11     (two rows)
                                 ──────
union = 482 + 663 − 11         = 1,134
one box "WAVELET SCATTERING"   = 1,134     ← exact match
```

Confirmed on the original triple: one box `ACOUSTIC DRONE DETECTION` = 48,979,
which necessarily exceeds `DETECTION` alone at 45,296, as a union must.

**This is the entire explanation for the "hundreds of thousands of hits"
problem.** It is not a database-size issue.

---

## 3. Matching is whole-word — no stemming, no substring

| Term | Hits |
|---|---|
| `DETECT` | 26,248 |
| `DETECTION` | 45,296 |
| `DETECTING` | 24,684 |
| `DRONE` | 1,941 |
| `DRONES` | 1,126 |
| `UAV` | 1,166 |

Two independent proofs:

- **Not substring or prefix.** If `DETECT` matched inside "detection" it would
  have to return at least 45,296. It returns 26,248.
- **Not stemmed.** `DRONE` and `DRONES` return different counts, as do
  `DETECTION` and `DETECTING`.

**Consequence:** every morphological variant must be searched explicitly.
`DRONE` does not find "drones". This is the single biggest recall trap in the
portal, and it is invisible to the user — the query succeeds and silently
returns a subset.

---

## 4. OR-in-box composes with AND-across-rows

This is the finding that makes the whole project cheap.

| Query | Hits |
|---|---|
| `ACOUSTIC` AND `DRONE` AND `DETECTION` | 34 |
| `ACOUSTIC` AND `DRONE DRONES UAV UAVS QUADCOPTER` AND `DETECTION` | **45** |
| All three facets as full OR-boxes | **109** |

Widening only the platform box recovered 11 documents (+32%). Widening all
three recovered 75 (+221%).

So the recommended query shape is one box per facet, every synonym and
morphological variant ORed inside it, boxes ANDed together:

```
Row 1  ABS  ACOUSTIC AUDIO SOUND MICROPHONE ACOUSTICAL          AND
Row 2  ABS  DRONE DRONES UAV UAVS QUADCOPTER MULTIROTOR UAS     AND
Row 3  ABS  DETECTION DETECTING DETECT CLASSIFICATION ...
```

The behaviour that caused the 48,979 problem is the same behaviour that
solves it — once it is applied *within* a facet instead of *across* facets.

---

## 5. IPC search needs full codes

| Query | Hits |
|---|---|
| `IC: G01S` | **0** |
| `IC: G01S13/00` | 1 |

Bare four-character section codes return nothing. The field requires a full
classification symbol including the subgroup. Axis C must therefore enumerate
specific subgroups rather than sweep sections.

---

## 6. No phrase search

Because a box ORs its words, there is no way to search the exact phrase
`"WAVELET SCATTERING"`. The closest equivalent is two ANDed rows (`WAVELET`
AND `SCATTERING` = 11), which matches documents containing both words
anywhere, not adjacently.

**Consequence for the profile:** facet terms must be split into
single words (safe inside an OR-box) and multi-word phrases (emitted as
separate ANDed rows). Putting `UNMANNED AERIAL` in an OR-box silently becomes
`UNMANNED` OR `AERIAL`, which is far broader than intended.

---

## 7. Full-text search works but is broad

`CSP: WAVELET SCATTERING` = 13,011 — consistent with OR across full
specifications. Usable for precision queries as ANDed rows; not usable as a
broad sweep.

---

## 8. Operational constraints

| Constraint | Detail |
|---|---|
| **Sessions go stale** | A long gap between captcha issue and search submit yields **HTTP 500**, not a captcha error. Batching captcha solving separately from submission does not work. Solve and submit immediately |
| Captcha is per query | Paging and abstract retrieval need none |
| A fetched captcha invalidates the previous one | For that session |
| Connections reset freely | An F5 fronts the service; retry with backoff is required |
| Broad queries are slow but do not fail | `DETECTION` at 45,296 returned fine on a fresh session |

---

## 9. What this changes in the plan

1. **Query count collapses.** The original per-synonym cross-product was 413
   queries. With OR-boxes the same coverage is roughly 15–30 queries.
2. **The profile needs restructuring** — terms split into single words
   (OR-box safe) vs phrases (must be ANDed rows).
3. **Morphological variants become mandatory**, not optional, for every facet.
4. **Axis C needs real IPC subgroups**, not section codes.
5. **Captcha solving must be inline**, which makes the internal vision
   endpoint more valuable than expected — manual solving is workable for
   15–30 queries but tedious, and the stale-session limit makes batching
   impossible.
