# Prior-Art Search Automation — Plan

**Status:** draft for review
**Date:** 21 September 2026
**Classification:** Internal — L&T PES. Contains pre-filing invention detail.
Do not paste into public AI tools or public search engines.

---

## 1. What we are trying to do

Produce a shortlist of **10–50 patents/papers** that a person can actually read,
so Khyati and Bharat can judge whether the IIT Jammu acoustic drone-detection
architecture is novel — *before* we commit ₹30,000–50,000 and a week of
Bhatt & Phonche's time to a formal prior-art search.

The output is a go/no-go input, not a legal opinion. If the shortlist shows
crowded art, we park the idea cheaply. If it shows white space, the ₹50k is
well spent and B&P start from our search log rather than from zero.

### Secondary objective

Answer the business question raised in the meeting: **has anyone applied
wavelet scattering to classification in other domains?** This determines how
broadly we can claim, and whether the technique itself is a licensable asset
beyond drone detection.

---

## 2. What we already know about the portal

I probed `iprsearch.ipindia.gov.in/PublicSearch` directly. These are measured,
not assumed.

### Measured

| Finding | Detail |
|---|---|
| Search form is ASP.NET **MVC**, not WebForms | No `__VIEWSTATE`, no anti-forgery token. Plain form POST works — **no browser automation needed** |
| Form supports **16 search rows** | Each row is a `(field, text, AND/OR/NOT)` triple. The "Add" button clones them. Most users never click it |
| Searchable fields | Title, Abstract, **Complete Specification (full text)**, Applicant, Inventor, **IPC classification**, Filing office, and number fields |
| Captcha is per-**query**, not per-**result** | 200×40 PNG, 6 mixed-case alphanumerics, light dot noise. Bound to the ASP.NET session cookie |
| Paging needs **no captcha** | Server caches each query under a `QueryString` GUID |
| Abstract retrieval needs **no captcha** | `POST /PublicationSearch/PatentDetails` returns the full abstract |
| One term per row, ANDed, collapses the result set | `ACOUSTIC` + `DRONE` + `DETECTION` as 3 ANDed rows → **34 documents** |

The last two rows are what make this project cheap. One captcha buys an entire
query including all its pages and all its abstracts, so running 80 queries
costs 80 captchas, not 80,000.

### Not yet measured — Phase 0 resolves these

1. **Does a single text box OR its words?** Inferred from the ~40,000 hit count
   and from the description in the meeting. Not directly tested.
2. **Is matching substring, stemmed, or whole-word?** Determines whether
   `DETECTION` finds "detecting". Drives how many synonym variants we need.
3. **How does `LogicField{N}` bind?** There is a per-row operator *and* a
   separate top-level `LogicField`. Does row N's operator join it to the row
   above or below? Getting this wrong silently produces wrong result sets.
4. **Is `CSP` (full specification) search usable?** Full-text search is far
   more sensitive than abstract-only — but may be slow or may return
   unusably broad sets.
5. **Does `IC` (IPC) search work as expected?** IPC codes are the highest-
   precision tool in patent search.

---

## 3. The core problem is recall, not volume

This is the most important judgement in the plan, so it gets its own section.

The obvious reading of the task is "40,000 is too many, narrow it down." That
framing is a trap. Narrowing is easy — three ANDed rows already gets to 34.
The hard part is **not missing the one document that matters**.

A tight list of 34 that *looks* authoritative is more dangerous than an
obviously messy list of 4,000, because it invites false confidence. If the
portal does literal substring matching, then a query on `DRONE` silently misses
every abstract that says only `UAV`, `quadcopter`, or `unmanned aerial vehicle`
— and any one of those could be the filing that sinks ours.

**Therefore the design runs many deliberately-overlapping queries and unions
them,** rather than trying to craft one perfect query. Precision is recovered
afterwards by ranking, which is cheap and reversible. Recall lost at search
time is gone forever and we never find out.

---

## 4. Confidentiality by construction

The meeting was emphatic: putting the invention into a public tool creates
prior art against ourselves. The design enforces this structurally rather than
by asking people to be careful.

| Rule | How it is enforced |
|---|---|
| The inventive *combination* is never submitted anywhere | The query planner emits only 2–3 term combinations, each of which is generic vocabulary appearing in thousands of public papers (`WAVELET SCATTERING` + `CLASSIFICATION`). The combination exists only in the local scorer |
| Invention text never reaches an external service | The fingerprint lives in a local YAML file. Scoring is local. The optional LLM judge must point at a **self-hosted internal** endpoint |
| Captcha solving is isolated from invention data | Captcha images carry no proprietary content, so that endpoint has no such constraint |
| Nothing is published or hosted | All outputs are local files. No artifact, no cloud, no shared link |

This is also, incidentally, what your boss asked for when he said *"I should
indirectly ask those questions"* — made systematic instead of left to judgement.

---

## 5. Query strategy

Two axes, because there are two different questions.

### Axis A — novelty: "is our thing already taken?"

Anchor on the three facets that define the application, then progressively
add method facets to find increasingly close art.

```
   modality  ×  platform  ×  task          <- broad sweep, all synonym pairs
   + method_core (wavelet scattering)      <- the money query
   + method_ml  (CNN / LSTM / transformer)
   + features   (MFCC / mel / fusion)
   + deployment (edge / real-time)
   + security   (federated / privacy)
```

Each facet carries a synonym list, so `platform` expands across
`DRONE | UAV | UNMANNED AERIAL | QUADCOPTER | UAS | MULTIROTOR`. The planner
takes the cross-product within a sensible budget rather than one hand-picked
query.

### Axis B — white space: "is our method used elsewhere?"

Deliberately **drops the platform facet** and searches wavelet scattering
against any classification domain — biomedical, machinery fault, speech,
seismic. This answers the claim-breadth question and the "is this technique a
business in itself" question from the meeting.

### Axis C — IPC classification sweep

Independent of keywords entirely. Sweep the relevant IPC codes intersected
with `ACOUSTIC` / `DRONE`:

`G01S` detection & ranging · `G10L` audio analysis · `G06N` ML ·
`B64U` UAVs · `G01H` sound measurement · `G08B` alarm systems

IPC catches documents that use vocabulary we never thought of, which is
exactly the recall failure mode Axis A is vulnerable to. **This is the
cross-check on our own keyword blind spots.**

### Budget

Roughly 40–80 queries. At one captcha and ~4 seconds of politeness delay each,
a full sweep is well under an hour, unattended.

---

## 6. Ranking

Everything here is local. Three signals, combined:

1. **Facet coverage (rules).** Weighted by the profile. A document must hit all
   three *required* facets (modality, platform, task) to clear Tier 3. A hit on
   `method_core` — wavelet scattering — carries the heaviest weight, because
   that single signal matters most for our novelty question.
2. **TF-IDF similarity** between the harvested abstract and the local invention
   fingerprint. Catches conceptual closeness that keyword rules miss.
3. **LLM relevance judgement** *(only if an internal endpoint is configured).*
   Produces a per-document "why this is relevant" line and a closeness rating.
   This is what turns a ranked list into something readable.

Output tiers:

- **Tier 1 — Direct.** Acoustic + UAV + detection + ML architecture. Read every one.
- **Tier 2 — Adjacent.** Either same method different domain (Axis B), or same
  domain different method. Skim.
- **Tier 3 — Noise.** Kept in the database for audit, excluded from the shortlist.

### Sanity control

The profile carries `known_art_controls` — prior art we already know exists
(the single-feature WST + 1D-CNN work, SVM+MFCC approaches, CRNN on
mel-spectrograms). **If a run does not surface these, the pipeline is broken**,
regardless of how plausible its output looks. This is the guard against a
confident-looking empty result.

---

## 7. Deliverables

All local files, generated from one SQLite store.

1. **HTML review app** — single self-contained file, opens by double-click, no
   install. Sortable and filterable, abstract expands inline, tier colour
   coding, mark relevant/reject with notes, print to PDF. This is what Khyati
   and Bharat actually use.
2. **Markdown novelty memo** — closest art found, which facets are crowded,
   where white space appears, and a go/no-go recommendation on the ₹50k.
3. **Search log** — every query, its field/term/operator structure, its hit
   count, and the run date. This is the artefact B&P will want, and it makes
   the whole exercise reproducible and defensible.

---

## 8. Phases

| Phase | Work | Status |
|---|---|---|
| **0** | Portal semantics experiment — settle the five open questions in §2 | **Next** |
| **1** | Query planner: profile facets → tiered query plan across axes A/B/C | Not started |
| **2** | Harvester: run plan, page, fetch abstracts, store in SQLite, resumable | Client built |
| **3** | Scorer: facet rules + TF-IDF + optional LLM judge | Not started |
| **4** | Reports: HTML app, memo, search log | Not started |
| **5** | Full run, tune against controls, hand over shortlist | Not started |
| **6** | *Optional:* Google Patents / Espacenet / arXiv sources | Deferred |

### Already built and validated

- `priorart/config.yaml` — endpoints, pacing, run parameters
- `priorart/profiles/acoustic_uav.yaml` — facets, weights, IPC codes,
  fingerprint, controls. **Swap this one file to search a different invention**
  — Bharat and Vaibhav's parallel framework needs no code change
- `priorart/inpass/captcha.py` — pluggable solver; manual fallback works today,
  vision endpoint drops in via config
- `priorart/inpass/client.py` — session, captcha, search, paging, abstract
  retrieval. Tested end-to-end against the live portal

---

## 9. Risks and limits

**InPASS alone cannot establish novelty.** This is the one thing I most want on
record. It covers Indian filings only. The disclosure's own prior-art section
cites 18 academic references and not one of them is an Indian patent. The art
most likely to block this filing is a US/CN/EP patent or an IEEE/arXiv paper.
The tool is built with a pluggable source layer so Google Patents, Espacenet
and arXiv can be added without rework — but **the shortlist from Phase 5 is a
partial answer, and the memo will say so on its first page.**

| Risk | Mitigation |
|---|---|
| Silent recall failure from vocabulary blind spots | Multi-axis querying + IPC sweep + known-art controls |
| Captcha accuracy degrades or portal redesigns it | Manual solver fallback; audio captcha is a second path |
| Portal rate-limits or blocks us | Conservative pacing, retry with backoff, resumable runs |
| Abstract-only search misses detail buried in specifications | `CSP` full-text field evaluated in Phase 0 |
| Shortlist creates false confidence | Tiering, per-item rationale, explicit jurisdiction caveat in the memo |

---

## 10. Open questions

1. **Vision / LLM endpoint details** — base URL, model names, auth, API shape.
   Not blocking: manual captcha entry works today.
2. **Should Phase 6 be pulled forward?** My recommendation is yes, at least
   Google Patents, because of the jurisdiction limit in §9. Google Patents has
   no captcha and covers most jurisdictions, so it is genuinely *less* work
   than InPASS was.
3. **Does Bharat want his parallel framework searched in the same run?** It is
   one extra profile file, and the marginal cost is near zero.
4. **Who owns the review pass?** The tool produces a ranked shortlist; someone
   still has to read 30 abstracts and make the call.
