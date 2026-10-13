"""Parse a B&P disclosure document into named questionnaire sections.

The requirements form used for this filing is one of several templates B&P
use ("for a design-related, hardware-related, the questionnaires are
different" -- meeting transcript). Only the software template is implemented;
adding another is meant to be a new anchor list, not new parsing logic.

Anchors are matched as case-insensitive PREFIXES of a paragraph, not exact
strings, because the live document has inconsistent spacing and punctuation
around headings (e.g. "NATURE OF INVENTON:" -- note the typo in the source
form itself). A section runs from its anchor to the next matched anchor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from docx import Document

# (section_key, label, anchor prefixes to match, case-insensitive)
SOFTWARE_TEMPLATE: list[tuple[str, str, list[str]]] = [
    ("applicant", "Applicant",
     ["full name, address and nationality of the applicant"]),
    ("inventors", "Inventors",
     ["full name/s, address/s and nationality of the inventor"]),
    ("title", "Title of invention",
     ["title of your invention"]),
    ("nature", "Nature of invention",
     ["nature of inventon", "nature of invention"]),
    ("prior_art", "Prior art",
     ["prior art:"]),
    ("deficiencies", "Deficiencies in prior art",
     ["8a.", "the deficiencies or drawbacks"]),
    ("further_search", "Further search requested (8B)",
     ["8b."]),
    ("purpose", "Purpose / objects of the invention",
     ["the purpose or object of your invention"]),
    ("technical_advancement", "Technical advancement over prior art",
     ["describe how the present invention overcomes"]),
    ("description", "Complete description",
     ["complete description of your invention"]),
    ("inventive_features", "Inventive / innovative features",
     ["description of the inventive/innovative feature",
      "description of the inventive / innovative feature"]),
    ("hardware", "Hardware system description",
     ["the description of hardware system"]),
    ("advantages", "Advantages",
     ["advantages of your invention"]),
    ("examples", "Illustrations of examples",
     ["illustrations of examples", "example/s especially in"]),
    ("declaration", "Declaration",
     ["i/we hereby declare"]),
]

# Boilerplate instruction text embedded in the form itself (guidance to the
# person filling it in), not invention content. Stripped so it doesn't pollute
# what gets sent to the LLM or, transitively, influence claim extraction.
_INSTRUCTION_MARKERS = (
    "it should be brief", "i.e. existing position", "the deficiencies and drawbacks of the existing",
    "this means the design and execution", "the description should preferably begin",
    "to provide details for drafting", "- it is a broad and expanded version",
)


@dataclass
class Disclosure:
    path: str
    sections: dict[str, str] = field(default_factory=dict)
    raw_paragraphs: list[str] = field(default_factory=list)

    def get(self, key: str, default: str = "") -> str:
        return self.sections.get(key, default)

    def joined(self, *keys: str) -> str:
        return "\n\n".join(self.sections[k] for k in keys if self.sections.get(k))


def _extract_paragraphs(path: str) -> list[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def parse_docx(path: str, template: list[tuple[str, str, list[str]]] = SOFTWARE_TEMPLATE) -> Disclosure:
    paras = _extract_paragraphs(path)

    # Find where each anchor first matches, in document order.
    hits: list[tuple[int, str]] = []
    for i, para in enumerate(paras):
        low = para.lower()
        for key, _label, prefixes in template:
            if any(low.startswith(pfx) or pfx in low[:len(pfx) + 20] for pfx in prefixes):
                hits.append((i, key))
                break

    # Keep the FIRST occurrence of each key, in document order, dedupe.
    seen: set[str] = set()
    ordered: list[tuple[int, str]] = []
    for i, key in hits:
        if key not in seen:
            seen.add(key)
            ordered.append((i, key))

    sections: dict[str, str] = {}
    for idx, (start, key) in enumerate(ordered):
        end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(paras)
        body_paras = paras[start + 1:end]  # exclude the heading line itself
        # The heading line itself sometimes carries content after the
        # separator (e.g. "PRIOR ART: Drone detection on IIT Jammu dataset...",
        # or "TITLE OF YOUR INVENTION -  \u201cAn ML based framework...\u201d"
        # which uses an en-dash rather than a colon).
        heading = paras[start]
        tail = ""
        for sep in (":", "\u2013", "\u2014", "-"):
            if sep in heading:
                _, _, candidate = heading.partition(sep)
                if candidate.strip():
                    tail = candidate.strip()
                    break
        if tail:
            body_paras = [tail] + body_paras
        body = [p for p in body_paras
                if not any(p.lower().startswith(m) for m in _INSTRUCTION_MARKERS)]
        sections[key] = "\n".join(body).strip()

    return Disclosure(path=path, sections=sections, raw_paragraphs=paras)
