# Prior Art Search — InPASS

A local tool for narrowing Indian patent searches from tens of thousands of
hits down to a readable shortlist, so novelty can be assessed internally before
commissioning a formal prior-art search.

Built for the acoustic drone-detection filing (L&T PES / IIT Jammu), but the
invention is described entirely in a profile file — point it at a different
profile and it searches a different invention with no code change.

---

## Quick start — the examiner

```bash
# INTRANET: derive generic search rows from the disclosure's domain sections
python3 scripts/prepare_search.py --disclosure "<B&P form>.docx" [--site=ppubs]

# INTERNET: harvest (type the captcha for InPASS; none for USPTO)
python3 scripts/run_examiner.py <conversation_id>

# INTRANET: claims -> claim chart -> interview -> report
python3 scripts/run_examiner.py <conversation_id>
```

On the intranet run, the examiner:

1. Drafts claims from the disclosure, split into elements. You edit and accept them.
2. Reads every relevant harvested reference against every element. This builds the claim chart; every quote is verified against the source text.
3. Interviews you like a patent attorney would: one deep question at a time, citing specific references, with full memory of your answers. Type `/help` for commands.
4. Writes `priorart/out/<conversation_id>/report.md`. The report covers novelty per claim (computed from the chart), inventive-step risk, Section 3(k), white space, publishable vs patentable, and whether to commission the formal search.

Follow-up searches the examiner suggests go through a code gate that blocks any query combining two inventive features. You approve each one, and it runs on your next internet session.

The LLM endpoint must be a private or loopback address. The client refuses anything else.

---

## Why the portal needs a tool at all

Typing `acoustic drone detection` into the abstract box returns **48,979**
documents. The portal **ORs the words in a box**, so you get everything
mentioning acoustics *or* drones *or* detection.

The fix is one box per concept, ANDed together. But the portal also matches
**whole words only** — `DRONE` does not find "drones", "UAV" or "quadcopter" —
so a naive ANDed query returns a clean-looking 34 documents while silently
missing 69% of the relevant set.

This tool puts every synonym and morphological variant of a concept into one
box (ORed), and ANDs the boxes:

| Approach | Hits |
|---|---|
| One box, three words | 48,979 |
| Three boxes, one term each | 34 |
| Three boxes, full synonym sets | **109** |

See [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md) for the measurements behind this,
including the set arithmetic proving the OR behaviour.

---

## Confidentiality

Putting invention detail into a public tool can create prior art against your
own filing. The design enforces that structurally:

- **No query ever contains the inventive combination.** Each submitted query is
  generic vocabulary appearing in thousands of public papers. The planner
  refuses to emit a query chaining the distinguishing method facets together.
- **The fingerprint never leaves the machine.** Scoring is local — rules plus
  scikit-learn TF-IDF, no network call.
- **The optional LLM relevance judge must point at a self-hosted internal
  endpoint.** It sees invention text; an external API must not.
- **The captcha endpoint has no such constraint** — captcha images carry no
  proprietary content.
- **Nothing is published or hosted.** All output is local files.

The profile YAML contains pre-filing invention detail. Do not paste it into a
public tool.

---

## The two questions it answers

**Axis A — novelty.** Acoustic × UAV × detection, progressively refined by
method. Is our thing already taken?

**Axis B — white space.** Wavelet scattering against *any* domain, with the
platform facet deliberately dropped. Has anyone claimed the technique
elsewhere? This bounds how broadly we can claim, and whether the method is an
asset beyond drone detection. Results on this track are flagged **◆ WHITE
SPACE** in the UI.

**Axis C — classification.** IPC subgroup sweep, independent of keywords
entirely. Axes A and B can only find art that shares our vocabulary; Axis C is
the cross-check on our own blind spots. Note that bare section codes return
zero — full symbols including subgroup are required (`G01S13/00`, not `G01S`).

---

## Scoring

Documents are scored locally on two signals: weighted facet coverage, and
TF-IDF similarity to the invention fingerprint.

Tiering is **track-aware**. A document qualifies if it answers *either*
question — Axis A needs all required facets (acoustic + platform + task);
Axis B needs only the method core. Judging everything against Axis A would
discard every white-space result, since "wavelet scattering for bearing fault
diagnosis" contains no drone and is exactly what we want to find.

- **Tier 1** — read every one
- **Tier 2** — skim
- **Tier 3** — kept for audit, excluded from the shortlist

### Sanity control

The profile lists `known_art_controls` — prior art we already know exists. **If
a run does not surface these, the pipeline is broken**, however plausible its
output looks. Without this check, a silently-broken run and a genuine
white-space finding are indistinguishable.

---

## Layout

```
priorart/
  config.yaml              endpoints, pacing, run defaults
  profiles/
    acoustic_uav.yaml      CONFIDENTIAL — facets, IPC codes, fingerprint
  inpass/
    client.py              portal protocol: session, captcha, search, paging, detail
    captcha.py             pluggable solver (manual / vision endpoint / chain)
  core/
    planner.py             profile -> query plan across the three axes
    store.py               SQLite: dedupe, resumability, search log
    scorer.py              local scoring, track-aware tiering
  web/
    app.py                 Flask backend
    static/index.html      UI
  out/priorart.db          results
```

---

## Searching a different invention

Copy `profiles/acoustic_uav.yaml`, edit the facets and fingerprint, and run:

```bash
python3 -m priorart.web.app --profile priorart/profiles/your_invention.yaml
```

Facet vocabulary is also editable from the **Vocabulary** tab in the UI, which
writes back to the YAML.

---

## Optional: automatic captcha solving

Not required. In the web UI you type the captcha yourself, which is fine
because you are already there — and it sidesteps a real constraint: portal
sessions go stale, and a long gap between issuing a captcha and submitting
returns **HTTP 500**, not a captcha error.

For unattended batch sweeps, fill in `endpoints.vision` in `config.yaml` with
an internal vision endpoint and set `enabled: true`. Any OpenAI-compatible
server works (vLLM, Ollama, LiteLLM, TGI). An **auto** button then appears next
to the captcha.

---

## Known limits

**InPASS covers India only.** The disclosure's own prior-art section cites 18
academic references, none of them Indian patents. The art most likely to block
this filing is a US/CN/EP patent or an IEEE/arXiv paper. A shortlist from this
tool is a useful, cheap first pass — **it is not a novelty opinion**, and it is
not a substitute for the formal search.

The source layer is pluggable, so Google Patents, Espacenet and arXiv can be
added without rework. Google Patents in particular has no captcha and covers
most jurisdictions, making it less work than InPASS was.

Other limits:

- No phrase search exists — a box ORs its words, so `"WAVELET SCATTERING"`
  cannot be matched adjacently, only as two ANDed rows
- Abstract search misses detail buried in full specifications; use the
  `Complete Specification` field for precision queries
- Connections reset freely (an F5 fronts the service); the client retries with
  backoff
