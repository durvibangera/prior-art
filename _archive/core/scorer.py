"""Rank harvested documents against the invention profile.

Runs entirely locally.  The invention fingerprint is proprietary pre-filing
material, so the default path uses only rules plus scikit-learn TF-IDF and
makes no network call at all.  An optional relevance judge can call an
*internal* LLM endpoint for a human-readable rationale.

Two signals:

1. Facet coverage.  Whole-word matching against the same vocabulary used to
   build the queries.  A document must hit every `required` facet to be
   considered relevant at all -- a patent about acoustic leak detection in
   pipelines matches modality and task but has no platform, so it is noise.

2. TF-IDF cosine similarity to the fingerprint.  Catches documents that are
   conceptually close while using vocabulary the facet lists do not contain,
   which is the blind spot the rules alone cannot cover.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Weight of TF-IDF similarity relative to facet score.
SIMILARITY_WEIGHT = 8.0

# A facet present in almost every harvested document tells us nothing -- it is
# measuring abstract length, not relevance. Facet weights are therefore scaled
# by how RARE the facet is in the corpus actually retrieved. `IDF_FLOOR` keeps
# a ubiquitous facet from collapsing to zero, which matters when a deliberately
# narrow query makes a genuinely important facet common.
IDF_FLOOR = 0.30

# A concept named in the TITLE is what the document is about; the same word
# buried in a boilerplate paragraph usually is not.
TITLE_BONUS = 3.0
TITLE_BONUS_CORE = 6.0

# Tiering is RANK-based, not threshold-based. An absolute cutoff does not
# generalise: the same score means different things in a 40-document corpus and
# a 4,000-document one, and the brief here is a shortlist a person will
# actually read. These are the default sizes; override per profile.
TIER1_SIZE = 15
TIER2_SIZE = 25

# A floor so a weak corpus does not get promoted just for being small.
MIN_SCORE = 3.0


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9\-]*", (text or "").upper()))


class Scorer:
    def __init__(self, profile: dict):
        self.profile = profile
        self.facets = profile["facets"]
        self.required = [n for n, f in self.facets.items() if f.get("required")]
        self.fingerprint = profile.get("fingerprint", "")

    # A document is relevant if it answers EITHER question the search asks.
    # Judging everything against the novelty requirements alone would discard
    # every white-space result, because Axis B deliberately drops the platform
    # facet -- "wavelet scattering for bearing fault diagnosis" has no drone in
    # it and is precisely the finding we are looking for.
    def _tracks(self, matched: dict) -> list[str]:
        tracks = []
        if all(r in matched for r in self.required):
            tracks.append("A_novelty")
        if "method_core" in matched:
            tracks.append("B_whitespace")
        return tracks

    # ------------------------------------------------------------------ facets

    def match_facets(self, text: str) -> dict[str, list[str]]:
        """Which facet terms appear, whole-word, mirroring portal semantics."""
        words = _word_set(text)
        upper = (text or "").upper()
        matched: dict[str, list[str]] = {}
        for name, facet in self.facets.items():
            hits = [w for w in facet.get("words", []) if w in words]
            hits += [p for p in facet.get("phrases", []) if p in upper]
            if hits:
                matched[name] = hits
        return matched

    def fit_idf(self, all_matched: list[dict]) -> None:
        """Learn how discriminating each facet is in THIS corpus.

        Measured on a real run: `security` matched 208/224 documents and
        `method_core` matched 10. Treating those as comparable evidence ranked
        an anti-poaching system above the closest real art.
        """
        import math
        n = max(len(all_matched), 1)
        self._idf = {}
        for name in self.facets:
            df = sum(1 for m in all_matched if name in m)
            idf = math.log((n + 1) / (df + 1)) / math.log(n + 1)
            self._idf[name] = IDF_FLOOR + (1 - IDF_FLOOR) * max(0.0, min(1.0, idf))

    def facet_score(self, matched: dict[str, list[str]], title: str = "") -> float:
        idf = getattr(self, "_idf", {})
        upper = (title or "").upper()
        title_words = _word_set(title)
        score = 0.0
        for name, hits in matched.items():
            w = self.facets[name]["weight"] * idf.get(name, 1.0)
            # Diminishing returns: the first hit carries the signal, further
            # synonyms of the same concept add little.
            score += w * (1 + 0.15 * (len(hits) - 1))
            # Centrality: is the concept in the title, or just mentioned?
            if any(h in title_words or h in upper for h in hits):
                score += TITLE_BONUS_CORE if name == "method_core" else TITLE_BONUS
        # counter_uas is explicitly not our claim; its presence usually marks
        # an interception patent rather than close detection art.
        if "counter_uas" in matched and "method_core" not in matched:
            score -= 1.0
        return score

    # -------------------------------------------------------------- similarity

    def similarities(self, texts: list[str]) -> list[float]:
        if not self.fingerprint or not texts:
            return [0.0] * len(texts)
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  min_df=1, max_features=20000)
            matrix = vec.fit_transform([self.fingerprint] + texts)
            return cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
        except ValueError:
            # Degenerate corpus (e.g. every abstract empty).
            return [0.0] * len(texts)

    # ------------------------------------------------------------------- score

    def score_documents(self, docs: list[dict]) -> list[dict]:
        """docs: dicts with application_number, title, abstract."""
        texts = [f"{d.get('title','')} {d.get('abstract','')}" for d in docs]
        sims = self.similarities(texts)

        # Two passes: learn facet rarity across the corpus, then score.
        all_matched = [self.match_facets(t) for t in texts]
        self.fit_idf(all_matched)

        out = []
        for doc, text, sim, matched in zip(docs, texts, sims, all_matched):
            score = self.facet_score(matched, doc.get("title", "")) + SIMILARITY_WEIGHT * sim
            out.append({
                "application_number": doc["application_number"],
                "score": round(score, 2),
                "similarity": round(sim, 4),
                "tracks": self._tracks(matched),
                "matched_facets": matched,
                "rationale": self._rationale(matched, self._tracks(matched), sim),
                "judged_by": "rules",
            })
        out.sort(key=lambda r: -r["score"])
        self._assign_tiers(out)
        return out

    def _assign_tiers(self, ranked: list[dict]) -> None:
        """Assign tiers by rank, with one override.

        Any document mentioning the distinguishing technique is promoted to at
        least Tier 2 regardless of rank. Measured on a real run, only 10 of 224
        documents mentioned wavelets at all -- they are the whole point of the
        search, and a short abstract must not be allowed to rank one out of the
        shortlist.
        """
        sizes = self.profile.get("shortlist", {})
        t1 = sizes.get("tier1", TIER1_SIZE)
        t2 = t1 + sizes.get("tier2", TIER2_SIZE)

        rank = 0
        for r in ranked:
            if not r["tracks"] or r["score"] < MIN_SCORE:
                r["tier"] = 3
                continue
            rank += 1
            r["tier"] = 1 if rank <= t1 else 2 if rank <= t2 else 3

        for r in ranked:
            if "method_core" in r["matched_facets"] and r["tracks"] and r["tier"] == 3:
                r["tier"] = 2
                r["rationale"] = "Mentions wavelet scattering. " + r["rationale"]

    def _rationale(self, matched, tracks, sim) -> str:
        if not tracks:
            missing = [r for r in self.required if r not in matched]
            return (f"Not close art: missing {', '.join(missing)} "
                    f"and no wavelet-scattering method")
        parts = []
        if tracks == ["B_whitespace"]:
            parts.append("WHITE SPACE: our method in another domain")
        if "method_core" in matched:
            parts.append(f"uses {', '.join(matched['method_core'][:3]).lower()}")
        for name in ("method_ml", "features", "deployment", "security"):
            if name in matched:
                parts.append(f"{self.facets[name]['label'].lower()}: "
                             f"{', '.join(matched[name][:2]).lower()}")
        if sim > 0.10:
            parts.append(f"textual similarity {sim:.2f}")
        return "; ".join(parts) if parts else "matches core facets only"
