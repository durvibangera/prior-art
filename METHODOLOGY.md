# Prior-Art Analysis Tool — Methodology

**For:** Khyati, Bharat
**Status:** Stage 1 (architecture decomposition) built and verified. Stages 2–5 designed, not yet built.

---

## What problem this solves

An InPASS keyword search on "acoustic drone detection" returns tens of thousands
of hits — too broad to review, and it never tells you *how similar* a hit
actually is to our architecture. This tool replaces "does this document contain
our words" with "for each part of our invention, has anyone built the same
thing, a known alternative, or something unrelated."

## Why keyword search doesn't work here

Typing three words into one search box ORs them — a single box returns anything
containing *any* of the words, which is where the 5-lakh/40k-hit problem comes
from. ANDing three single terms across separate rows narrows that down, but
InPASS matches whole words only (`DRONE` will not find `DRONES` or `UAV`), so a
naive narrow query looks clean and quietly drops most of the relevant set. And
even a well-formed keyword search only ever answers "contains these words" —
never "is this the same method."

## The methodology

**1. Decompose the architecture, not the vocabulary.**
An internal LLM reads the disclosure and breaks it into the actual stages data
passes through — e.g. *signal acquisition → feature extraction → classification
→ detection decision*. For each stage it records three things:

- **the job** — what that stage accomplishes, described generically enough that
  it doesn't presuppose a method (e.g. "produces a multi-resolution
  representation," not "does wavelet scattering")
- **our method** — the specific technique we use for that job
- **known alternatives** — what the rest of the field commonly uses for that
  same job, pulled first from *our own disclosure's cited prior art* (the most
  reliable source — for this invention, that's 18 academic references), and
  second from the model's general knowledge as a lower-confidence hint

The alternatives field is the actual mechanism: to find real competitors you
have to search for what *they* use, not what we use. An MFCC-based detector is
our closest real competitor, and no wavelet-scattering search would ever
surface it.

**2. Generate targeted search questions from the architecture.**
Not one broad query — several per stage: has anyone done exactly this; has
anyone solved this job a different way; who uses the known alternative
methods; is this specific technique used in an unrelated field at all
(a whitespace check). Every question stays generic public vocabulary — the
actual inventive *combination* is never sent to the portal, only generic
2–3-word pairs that already exist in thousands of public documents.

**3. Search InPASS with those questions**, using the same portal mechanics
already validated (OR-within-a-box, AND-across-rows, one captcha per query,
free paging and abstract retrieval after that).

**4. Compare every retrieved document against our architecture, stage by
stage.** Not a similarity score — for each stage, a specific judgment: does
this document use the *same* method, a *known alternative*, or something
*unrelated*, backed by a quoted sentence from their document.

**5. Aggregate into a whitespace map.** Which individual stages are crowded,
and — the actual answer to "how novel are we" — which *combinations* of stages
nobody else has built. Patents are won on combinations, not single techniques;
a stage-only view would say "CNN classifiers are crowded" and miss that nobody
has chained our specific front end to one.

## A design principle worth knowing

The only AI model available for this is small (~4B parameters) and makes
specific, repeatable mistakes on this kind of task — it has been observed
answering the same question identically regardless of which stage it was
asked about, and defaulting to the invention's headline technique instead of
the actual method for a minor stage. The methodology is built around that
constraint:

- Wherever a judgment can be **measured** instead of guessed, it is — e.g.
  which technique is genuinely rare is answered by counting real search hits
  against the portal, not by asking the model which one "sounds important."
- Every claim the model makes about the document must be backed by a **quoted
  sentence verified to actually exist** in the source text before it's trusted.
  Two independent checks caught and corrected a real error this way: the model
  answered a "capture raw audio" stage with "wavelet scattering transform" —
  wrong, since that's a later stage's job — and a targeted retry against just
  the hardware section corrected it to the real answer, HBK microphones with a
  Dewesoft DAQ board.
- A claiming pillar with no supporting content in the disclosure is **reported
  as empty**, not filled in. Measured on our own document: `secure_data_handling`
  is a pillar we intend to claim, but the disclosure currently describes no
  mechanism for it — the tool says so rather than inventing one.

## What's built and verified vs. what's designed

| Stage | Status |
|---|---|
| 1. Architecture decomposition | **Built, verified** against our real disclosure |
| 2. Search question generation | Designed, not built |
| 3. InPASS search | Portal mechanics already validated (separate earlier work); not yet wired to stage 2 |
| 4. Per-document comparison | Designed, not built |
| 5. Whitespace map | Designed, not built |

## Confidentiality

The invention's architecture, fingerprint, and this document all stay local —
nothing is sent to a public AI tool or published anywhere. Search queries sent
to InPASS are always generic vocabulary; the actual technical combination
never leaves this machine.
