"""What the examiner reads: the disclosure, and the harvested references.

Everything here is local -- TF-IDF retrieval, text splitting. Only the
selected text is later sent to the internal LLM.

The stored `abstract` column is not always an abstract: InPASS detail pages
often carry the full complete specification (measured: average ~22K chars on
InPASS, max 531K). That is better material than an abstract -- it has the
claims -- but it has to be split and searched, not pasted whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingest import parse_docx

# Content sections only. `declaration` is excluded on purpose: on the live B&P
# form the template's own worked example (a TV-subscription system) parses
# into it, and that is not our invention.
CONTENT_SECTIONS = ("title", "nature", "prior_art", "deficiencies", "purpose",
                    "technical_advancement", "description", "inventive_features",
                    "hardware", "advantages", "examples")

# Sections safe to derive the broad, generic portal search from: domain and
# purpose, never the inventive features or the detailed description.
DOMAIN_SECTIONS = ("title", "nature", "purpose")

_SECTION_LABELS = {
    "title": "Title", "nature": "Nature of invention", "prior_art": "Prior art (as the inventors describe it)",
    "deficiencies": "Deficiencies in prior art", "purpose": "Purpose / objects",
    "technical_advancement": "Technical advancement", "description": "Complete description",
    "inventive_features": "Inventive features", "hardware": "Hardware",
    "advantages": "Advantages", "examples": "Examples",
}


@dataclass
class Invention:
    path: str
    sections: dict[str, str]

    @property
    def title(self) -> str:
        return self.sections.get("title", "").strip("“”\" ")

    def text(self, keys: tuple[str, ...] = CONTENT_SECTIONS) -> str:
        parts = []
        for k in keys:
            body = self.sections.get(k, "").strip()
            if body:
                parts.append(f"## {_SECTION_LABELS.get(k, k)}\n{body}")
        return "\n\n".join(parts)


def load_invention(path: str) -> Invention:
    d = parse_docx(path)
    return Invention(path=path, sections={k: d.get(k) for k in CONTENT_SECTIONS if d.get(k)})


# ------------------------------------------------------------- references

CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_DOC = 400
HEAD_WEIGHT = 0.6          # abstract-level vs best-passage match, see retrieve()

_CLAIMS_START = re.compile(
    r"\b(what is claimed is|we claim|i claim|claims?\s*:|the claims defining the invention)",
    re.I)
_ABSTRACT_HEAD = re.compile(r"\babstract\b", re.I)


def normalize(text: str) -> str:
    text = (text or "").replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class RefText:
    application_number: str
    title: str
    site: str
    abstract: str
    claims: str
    chunks: list[str]
    full: str                      # normalized full stored text, for quote checks

    @property
    def is_full_spec(self) -> bool:
        return len(self.full) > 2500


def _chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), step)][:MAX_CHUNKS_PER_DOC]


def reference_text(doc: dict) -> RefText:
    full = normalize(doc.get("abstract") or "")
    title = normalize(doc.get("title") or "")
    abstract, claims = full, ""
    if len(full) > 2500:
        # Indian complete specifications put the ABSTRACT last; take the last
        # heading in the final 30% of the text, else fall back to the opening.
        abstract = full[:1500]
        tail_start = int(len(full) * 0.7)
        heads = [m for m in _ABSTRACT_HEAD.finditer(full) if m.start() >= tail_start]
        abs_start = heads[-1].start() if heads else None
        if abs_start is not None:
            abstract = full[heads[-1].end():heads[-1].end() + 1500].strip(" :-")
        m = _CLAIMS_START.search(full, int(len(full) * 0.3))
        if m:
            end = abs_start if abs_start and abs_start > m.start() else len(full)
            claims = full[m.start():end][:6000]
    return RefText(
        application_number=doc["application_number"], title=title,
        site=doc.get("site") or "inpass", abstract=abstract, claims=claims,
        chunks=_chunk(full), full=full)


@dataclass
class Reference:
    text: RefText
    score: float
    passages: list[str] = field(default_factory=list)   # best-matching body chunks

    @property
    def application_number(self) -> str:
        return self.text.application_number

    def digest(self, max_chars: int = 9000) -> str:
        """The block of text the model reads for this reference."""
        t = self.text
        site = "InPASS (India)" if t.site == "inpass" else "USPTO"
        parts = [f"[REF {t.application_number}] {site} -- {t.title}",
                 f"Abstract: {t.abstract}"]
        if t.claims:
            parts.append(f"Claims (excerpt): {t.claims[:3000]}")
        for p in self.passages:
            if p[:200] not in t.abstract and p[:200] not in t.claims:
                parts.append(f"Passage: {p}")
        return "\n".join(parts)[:max_chars]


def retrieve(docs: list[dict], element_queries: list[str], domain_query: str = "",
             k_per_query: int = 6, max_docs: int = 40, domain_share: float = 0.5,
             passages_per_doc: int = 3) -> list[Reference]:
    """Pick the references the examiner reads, on two tracks (the README's
    Axis A / Axis B):

    - SAME PROBLEM: documents whose title+abstract match the invention as a
      whole (`domain_query`) -- the novelty threats. Up to `domain_share` of
      `max_docs`, ranked on the abstract only, so a 500K-character
      specification cannot win on one stray passage.
    - SAME TECHNIQUE, ANY FIELD: for each claim element, the documents that
      match it best (abstract and best passage blended) -- what an
      inventive-step attack is built from, e.g. a scattering-transform
      patent for bearing faults. Filled round-robin across elements so one
      generic element ("convolutional layers") cannot take every slot.

    Measured on the live harvest: a single combined ranking let generic-ML
    InPASS specifications (deepfakes, tax anomalies) push every acoustic-UAV
    document out of the top 6.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    refs = [reference_text(d) for d in docs if (d.get("abstract") or "").strip()]
    queries = [q for q in element_queries if q.strip()]
    if not refs or not (queries or domain_query.strip()):
        return []

    rows, owner, is_body = [], [], []
    for i, r in enumerate(refs):
        rows.append(f"{r.title} {r.title} {r.abstract}")
        owner.append(i)
        is_body.append(False)
        if r.is_full_spec:
            for ch in r.chunks:
                rows.append(ch)
                owner.append(i)
                is_body.append(True)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                          min_df=1, max_df=0.6 if len(rows) >= 10 else 1.0,
                          token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9\-]+\b")
    try:
        matrix = vec.fit_transform(rows)
    except ValueError:                      # empty vocabulary
        return []
    all_queries = queries + ([domain_query] if domain_query.strip() else [])
    sims = linear_kernel(vec.transform(all_queries), matrix)   # (queries, rows)

    n_q, n_d = len(all_queries), len(refs)
    head_sim = [[0.0] * n_d for _ in range(n_q)]
    body_sim = [[0.0] * n_d for _ in range(n_q)]
    best_chunk: dict[int, dict[int, float]] = {}
    for q in range(n_q):
        for row, s in enumerate(sims[q]):
            d = owner[row]
            if not is_body[row]:
                head_sim[q][d] = s
            else:
                body_sim[q][d] = max(body_sim[q][d], s)
                if s > 0:
                    best_chunk.setdefault(d, {})
                    best_chunk[d][row] = max(s, best_chunk[d].get(row, 0.0))

    def element_score(q: int, d: int) -> float:
        if not refs[d].is_full_spec:
            return head_sim[q][d]
        return HEAD_WEIGHT * head_sim[q][d] + (1 - HEAD_WEIGHT) * body_sim[q][d]

    chosen: dict[int, float] = {}
    if domain_query.strip():
        dq = n_q - 1
        quota = int(max_docs * domain_share) if queries else max_docs
        for d in sorted(range(n_d), key=lambda d: -head_sim[dq][d])[:quota]:
            if head_sim[dq][d] > 0:
                chosen[d] = head_sim[dq][d]

    ranked = [sorted((d for d in range(n_d) if element_score(q, d) > 0),
                     key=lambda d, q=q: -element_score(q, d))[:k_per_query]
              for q in range(len(queries))]
    for depth in range(k_per_query):
        for q, lst in enumerate(ranked):
            if len(chosen) >= max_docs:
                break
            if depth < len(lst) and lst[depth] not in chosen:
                chosen[lst[depth]] = element_score(q, lst[depth])

    out = []
    for d, score in chosen.items():
        r = refs[d]
        passages = []
        if r.is_full_spec:
            ranked_rows = sorted(best_chunk.get(d, {}).items(), key=lambda kv: -kv[1])
            passages = [rows[row] for row, _ in ranked_rows][:passages_per_doc]
        out.append(Reference(text=r, score=float(score), passages=passages))
    return out


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.5)
