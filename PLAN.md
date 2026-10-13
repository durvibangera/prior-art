# Prior-Art Tool — Plan v2

**Supersedes** [PLAN_v1_superseded.md](PLAN_v1_superseded.md), which treated this
as a search problem. It is an *analysis* problem with a search step in the middle.

**Updated:** descoped running two inventions together (§7) in favour of a
generic upload flow — any disclosure document in, questions and a verdict out.
Bharat's framework, whenever it exists, is just a second document to upload,
not a feature to build for it specifically.

**Classification:** Internal — L&T PES. Pre-filing invention detail.

---

## 1. What changed

v1 built a query builder: you type keywords, it searches InPASS well. Reviewing
the transcript again, that is roughly the least interesting part of the ask.

What he actually described:

> *"feed your algorithm into your AI part of it, and **tell me these are the
> questions I proclaim**, like how in Google you start questioning... You start
> with one word, it completes a sentence. Whether you can help that do that"*

> *"each one I should download, preview the abstract, and then I need to
> **decipher A versus B versus C**"*

> *"**How am I different from the other?** ... You are novel, and this is your
> white space."*

> *"you can run through **this questionnaire** based on which the law firm does
> a prior art. We want the similar approach to be put into here."*

So: the tool should read the disclosure, **propose the questions itself**, run
them, and come back with a **per-claim verdict on how we differ**. A box to type
keywords into puts the human back in exactly the position the confidentiality
rule is trying to protect them from.

---

## 2. The flow

```
  B&P disclosure (.docx)
          │
          ▼
  ① INGEST ─────────── parse into questionnaire sections
          │              (prior art, deficiencies, objectives, description,
          │               inventive features, advantages)
          ▼
  ② CLAIMS ─────────── derive the claims we intend to file,
          │              each mapped to one of the four IP pillars
          ▼
  ③ QUESTIONS ──────── generate SAFE, generic search questions per claim
          │              → operator reviews exactly what will be sent
          ▼
  ④ SEARCH ─────────── InPASS, OR-boxes, union + dedupe     [built]
          │
          ▼
  ⑤ COMPARE ────────── per document: technique attributes, and
          │              per claim: does this anticipate it?
          ▼
  ⑥ VERDICT ────────── per-claim novelty, per-pillar crowding,
                         mapped back onto the B&P form sections
```

Steps ① ② ③ ⑤ ⑥ are new. Step ④ exists and works.

---

## 2b. Network reality: internet and intranet are mutually exclusive here

Discovered while wiring up endpoints, and it reshapes how this actually gets
run day to day: on this workstation, InPASS (public internet) and the OCR/LLM
endpoints (172.16.34.x, intranet) are reachable on **mutually exclusive
networks** — being set up to reach one means not reaching the other.

The two never need to overlap, once the pipeline is looked at honestly:

| Needs InPASS (internet) | Needs OCR/LLM (intranet) |
|---|---|
| ④ Search, page, fetch abstracts | ①②③ Ingest, claims, question generation |
| | ⑤⑥ Comparison, verdict |

The one place they'd otherwise collide is auto-solving the captcha mid-search.
That collision is free to avoid: the operator is already at the browser when a
search runs, so they read the captcha and type it — no network round-trip
needed. `captcha.mode` defaults to `manual` for exactly this reason.
`glm-ocr` auto-solve stays available as an option only for a machine that
happens to have both networks simultaneously (a dual-homed box or a VPN
split-tunnel), which is a nice-to-have, not a dependency.

Practical consequence — the tool is run in three passes, on whichever network
is up:

```
INTRANET  ①②③ ingest disclosure → claims → generated search questions
INTERNET  ④   web UI: paste in questions, search + harvest (type captchas)
INTRANET  ⑤⑥  compare harvested abstracts against claims → verdict
```

No architecture change was needed for this — it works because harvesting was
already designed to write everything to local SQLite before anything else
touches it. If a dual-homed machine exists somewhere (worth asking), the same
three passes can run back-to-back on one machine instead of two network
switches.

---

## 3. Confidentiality — now the central mechanism, not a side rule

In v1 the operator typed the queries, which meant the operator was the thing
standing between the invention and the search box. That is backwards.

In v2 the AI reads the confidential disclosure **locally** and emits generic
questions. The operator never composes a query, so they cannot accidentally
type the inventive combination into one.

Every generated question is shown before it is sent, with the exact portal rows
displayed. A question is blocked if it chains the distinguishing method facets
together. The disclosure, the claims and the fingerprint never leave the host;
the LLM that reads them must be the internal self-hosted endpoint.

Accuracy note: the transcript's warning was specifically about **ChatGPT and
Claude** — *"any searches that you do in chat GPT, cloud... will later come back
to the law firm"*. A query typed into the government's own patent portal is not
published. So portal searching is low-risk either way; the discipline costs
nothing and removes the judgement call.

---

## 4. The four pillars

From the meeting, with what we may claim:

| Pillar | Status | Why |
|---|---|---|
| Acoustic signal processing | **Out of scope** | No custom board or electronic card was developed |
| AI/ML classification architecture | **Claiming** | Primary area |
| Counter-drone interception | **Not ours** | Explicitly not L&T IP |
| Secure data handling | **Claiming** | Secondary area |

Output is organised by pillar, so crowding is visible per pillar rather than as
one undifferentiated pile. Art that only touches interception is recognised as
*not our fight* rather than mistaken for close prior art.

---

## 5. Per-claim comparison, not generic columns

The deliverable is a verdict per claim, not a technique table.

Draft claim set (to be confirmed with Bharat):

| ID | Claim | Pillar |
|---|---|---|
| C1 | WST as a non-trainable multi-resolution front end, replacing handcrafted acoustic features | AI/ML |
| C2 | Second-order scattering with log transform + global average pooling feeding a 1D-CNN | AI/ML |
| C3 | WST representation feeding CNN + stacked LSTM | AI/ML |
| C4 | WST with transformer in place of the recurrent stage | AI/ML |
| C5 | Statistically guided fusion across MFCC / spectral / mel domains | AI/ML |
| C6 | Multi-condition dataset across drone types, rotor speeds, flight patterns, terrains | AI/ML |
| C7 | Secure handling of captured acoustic data | Secure data |

For each shortlisted document, per claim: **anticipates / partially overlaps /
unrelated**, with the supporting sentence quoted from the abstract.

Rolled up, that gives the sentence the law firm needs: *"C1 and C3 are exposed;
C4 and C5 appear clear."*

---

## 6. Categorising results

Two orthogonal groupings, both applied:

**By relationship to us** — what a reader needs to act:

- **Same problem, same method** — threatens novelty directly
- **Same problem, different method** — competitive landscape
- **Different problem, our method** — the white-space question
- **Not relevant**

**By B&P questionnaire section** — so output drops into the law firm's process:
prior art · deficiencies in prior art · inventive feature · advantages.

The v1 "Tier 1/2/3" vocabulary is dropped. It was invented, undefined, and told
nobody anything.

---

## 7. One pipeline, any document — Bharat's framework is not a special case

Descoped for now: running two inventions in one comparison run. Not because it
isn't useful — *"he can also verify whether his idea is novel"* is a real
ask — but because building it around a specific second framework we don't have
yet would mean guessing its shape.

Instead the pipeline is generic end to end: **upload a disclosure document,
get questions, search, get a verdict.** There is no hardcoded "invention A"
and "invention B" — there is one flow that takes whatever `.docx` is handed to
it, parses it into questionnaire sections (§1), derives claims and pillars
(§2), and runs from there.

Consequence: when Bharat's framework is ready, using the tool on it is just
uploading that document and running the same pipeline again — a second,
independent run with its own claim set, its own generated questions, its own
verdict. Not a code change, not a new "slot." Each run's output is a
self-contained folder (questions, harvested documents, verdict) named after
the source document, so nothing overwrites the acoustic-UAV run when a second
one starts, and comparing the two later is just opening two reports.

This also fixes a shakier assumption in v1: that two inventions searched
together should be *assessed against both* in one pass. That only makes sense
if their claim sets overlap enough to reuse the same search. There's no reason
to assume that yet — better to keep runs independent until we know.

---

## 7b. Decided: vocabulary is auto-derived, not hand-written per invention

Confirmed 22 Sep: for a genuinely new document (Bharat's framework, or
anything else), the tool must **derive its own search facets, pillar
taxonomy, and fingerprint from the uploaded document via LLM** -- not read
them from a hand-written profile YAML the way `acoustic_uav.yaml` was built.

This was a real gap, not a detail: without it, "upload any document" was only
true in the sense that ingest's paragraph-parsing is generic -- claims.py,
questions.py and scorer.py all still depended on hand-crafted vocabulary
specific to THIS invention. Uploading a different document today would
silently search using the wrong technical vocabulary.

New step, ahead of ingest's existing claim extraction:

```
upload .docx
     |
     v
[NEW] auto-derive profile: LLM reads the disclosure and proposes
      facets (technique vocabulary), a pillar taxonomy, and a fingerprint
     |
     v
operator reviews/edits the proposed profile  <-- same review gate as claims
     |
     v
...ingest -> claims -> questions -> search -> compare -> verdict (unchanged)
```

The four pillars used for THIS invention (acoustic signal processing / AI-ML
classification / counter-drone / secure data handling) came directly from the
meeting transcript and are specific to this project -- they should not be
assumed for a different disclosure. Auto-derivation proposes a taxonomy
appropriate to whatever is actually uploaded, with acoustic_uav.yaml's
hand-written pillars kept as a worked example / fallback.

---

## 8. Generic questionnaire templates

*"This is for software-related inventions, for a design-related,
hardware-related, the questionnaires are different."*

The ingest step is template-driven, so a hardware or design disclosure uses a
different section map and a different claim-extraction prompt. Software is the
only template built now; the others are config, not code.

---

## 9. What survives from v1

| Component | Status |
|---|---|
| `inpass/client.py` — portal protocol | **Keep.** Validated end to end |
| `inpass/captcha.py` — pluggable solver | **Keep** |
| `core/store.py` — SQLite, dedupe, resumable | **Keep**, extend for claims/verdicts |
| `core/planner.py` — OR-box query planning | **Keep** as the engine under question generation |
| `core/scorer.py` — relevance ranking | **Keep** as a pre-filter; no longer the output |
| `web/` — UI | **Rework.** Query builder demoted to an advanced panel; the main flow becomes document → questions → verdict |
| Measured portal semantics | **Keep.** [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) |

---

## 10. Build order

1. **Ingest** — parse the .docx into questionnaire sections
2. **Claims + pillars** — derive claim set, operator confirms/edits
3. **Question generation** — claims → safe queries, with a visible safety check
4. **Comparison extractor** — per-document attributes, per-claim verdict
5. **Verdict + report** — per-claim novelty, per-pillar crowding, B&P mapping
6. **Re-run for another document** — confirm step 1 (ingest) genuinely works
   for any disclosure of the same template, not just this one, by treating
   this as a normal use of the tool rather than a special build step
7. **UI rework** — the above as the main flow

---

## 11. Open items

1. **Internal LLM endpoint** — base URL, model, auth, API shape. Steps ② ③ ⑤
   all degrade to keyword rules without it, and the comparison step degrades
   badly. This is now the main dependency.
2. **Claim set confirmation** — §5 is my reading of the disclosure. Bharat and
   Khyati should correct it before it drives anything.
3. **Bharat's framework.** No longer a blocker for anything — §7 means it's
   just the second document someone uploads, whenever it's ready. Nothing to
   prepare on my end in advance.
4. **Jurisdiction.** Unchanged from v1 and still the biggest limit: InPASS is
   India-only, and the art most likely to block this filing is a US/CN/EP
   patent or an IEEE paper. Recommend adding Google Patents.
