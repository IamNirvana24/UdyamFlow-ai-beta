"""
Retrieval layer for the UdyamFlow AI compliance copilot.

Deliberately uses TF-IDF rather than sentence-transformers:
  - scikit-learn is already a project dependency, so deployment adds nothing
  - the corpus is 12 small, vocabulary-distinct documents, where lexical
    matching performs well
  - no torch download, no model weights to ship, no cold-start latency

Documents are split into section-level chunks (one per markdown heading) so
that a question about documents required for a Fire NOC retrieves the
documents section, not the whole file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

# Maps identifiers coming from api.py to knowledge base slugs.
#
# api.py's target_cols keys (e.g. "trade_license", "fire_noc") already equal
# the knowledge base slugs exactly, so those pass through unchanged. The
# display names in api.py's LICENSE_LABELS dict differ from this project's
# original short-form names ("Pollution Control Board NOC" vs "Pollution
# NOC"), so both the exact api.py display names and a few common variants are
# mapped here to stay robust as label wording changes.
LABEL_TO_SLUG = {
    # snake_case keys from api.py's target_cols / LICENSE_LABELS dict keys —
    # identity mapping, since these already match the knowledge base slugs
    "trade_license": "trade_license",
    "gst_registration": "gst_registration",
    "factory_license": "factory_license",
    "labour_registration": "labour_registration",
    "fire_noc": "fire_noc",
    "pollution_noc": "pollution_noc",
    "environmental_clearance": "environmental_clearance",
    "fssai_license": "fssai_license",
    "boiler_license": "boiler_license",
    "explosives_license": "explosives_license",
    "electricity_noc": "electricity_noc",
    "building_plan_approval": "building_plan_approval",
    # exact display names from api.py's LICENSE_LABELS values
    "Trade License": "trade_license",
    "GST Registration": "gst_registration",
    "Factory License": "factory_license",
    "Labour / Shops & Establishment Registration": "labour_registration",
    "Fire NOC": "fire_noc",
    "Pollution Control Board NOC": "pollution_noc",
    "Environmental Clearance": "environmental_clearance",
    "FSSAI License": "fssai_license",
    "Boiler License": "boiler_license",
    "Explosives/Hazardous Storage License (PESO)": "explosives_license",
    "Electricity Load Sanction / NOC": "electricity_noc",
    "Building Plan Approval": "building_plan_approval",
    # common shorthand variants, kept for robustness if wording changes again
    "GST": "gst_registration",
    "Pollution NOC": "pollution_noc",
    "FSSAI": "fssai_license",
    "Explosives License": "explosives_license",
    "Electricity NOC": "electricity_noc",
    "Labour Registration": "labour_registration",
}


# Distinctive terms that, when present in a query, strongly indicate a specific
# document. Needed because tokens like "NOC", "licence", and "registration" are
# shared across the corpus and dilute pure TF-IDF matching.
SLUG_KEYWORDS = {
    "trade_license": ["trade licence", "trade license", "municipal licence", "shop licence"],
    "gst_registration": ["gst", "gstin", "input tax", "tax invoice", "turnover threshold"],
    "factory_license": ["factory licence", "factory license", "factories act", "occupier", "chief inspector of factories"],
    "fire_noc": ["fire", "fire noc", "hydrant", "sprinkler", "escape route", "fire safety"],
    "pollution_noc": ["pollution", "ppcb", "consent to establish", "consent to operate", "cte", "cto", "effluent", "red category", "orange category"],
    "environmental_clearance": ["environmental clearance", "eia", "public hearing", "terms of reference", "appraisal"],
    "fssai_license": ["fssai", "food", "foscos", "food safety"],
    "boiler_license": ["boiler", "steam", "pressure vessel", "boiler attendant"],
    "explosives_license": ["explosive", "peso", "lpg", "solvent", "compressed gas", "petroleum", "safety distance"],
    "electricity_noc": ["electricity", "power connection", "pspcl", "sanctioned load", "high tension", "substation", "transformer"],
    "labour_registration": ["labour", "epf", "esi", "provident fund", "contract labour", "shops and establishments", "professional tax"],
    "building_plan_approval": ["building plan", "land use", "zoning", "setback", "floor area ratio", "occupancy certificate", "puda", "agricultural land", "change of land use", "clu", "master plan", "construction"],
    "approval_sequence": ["first", "order", "sequence", "which comes", "what order", "parallel", "takes longest", "start with", "dependency", "roadmap", "timeline"],
}


# When the query is asking specifically for the documents list, boost that
# section within whichever document(s) matched — otherwise "Why it is
# required" or "What it is" chunks of the SAME document can outrank the
# actual documents list, since they share the document's title terms.
_DOCUMENT_INTENT_KEYWORDS = [
    "document", "documents", "papers", "paperwork",
    "what do i need", "required documents", "documents needed",
    "documents required", "what's needed", "whats needed",
]


@dataclass
class Chunk:
    slug: str
    title: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title} — {self.section}"


def _parse_front_matter(raw: str) -> dict:
    """Pull the loose `key: value` lines that sit under the H1 heading."""
    meta = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("##"):
            break
        match = re.match(r"^([a-z_]+):\s*(.+)$", line.strip())
        if match:
            meta[match.group(1)] = match.group(2).strip()
    return meta


def _chunk_document(raw: str, fallback_slug: str) -> List[Chunk]:
    """Split one markdown file into one chunk per `##` section."""
    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fallback_slug

    meta = _parse_front_matter(raw)
    slug = meta.get("slug", fallback_slug)

    # The preamble (everything before the first ## section) becomes an
    # "Overview" chunk so authority/applicability questions are retrievable.
    parts = re.split(r"^##\s+", raw, flags=re.MULTILINE)
    chunks: List[Chunk] = []

    preamble = parts[0]
    overview_bits = [f"{k}: {v}" for k, v in meta.items()]
    if overview_bits:
        chunks.append(
            Chunk(
                slug=slug,
                title=title,
                section="Overview",
                text=f"{title}\n" + "\n".join(overview_bits),
            )
        )

    for part in parts[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        section = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                slug=slug,
                title=title,
                section=section,
                # Prepending the title and section makes the chunk's own
                # identity part of what TF-IDF matches against.
                text=f"{title}. {section}. {body}",
            )
        )

    return chunks


class ComplianceRetriever:
    """Loads the knowledge base once at import time and serves top-k chunks."""

    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks: List[Chunk] = []
        self._vectorizer = None
        self._matrix = None
        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base directory not found: {self.kb_dir}")

        for filename in sorted(os.listdir(self.kb_dir)):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(self.kb_dir, filename)
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            self.chunks.extend(_chunk_document(raw, filename[:-3]))

        if not self.chunks:
            raise ValueError("Knowledge base loaded but contained no usable chunks.")

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = 4, approval_filter: str = None):
        """
        Return [(Chunk, score)] sorted by relevance.

        approval_filter: optional slug. When the user is asking from inside a
        specific approval card in the UI, pass it to bias retrieval to that
        document rather than relying on the query wording alone.
        """
        if not query or not query.strip():
            return []

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()

        # Keyword boost: if the query explicitly names a document's distinctive
        # terms, lift that document's chunks. Prevents "fire NOC" from losing to
        # "electricity NOC" on the shared token "NOC".
        lowered = query.lower()
        matched_slugs = {
            slug
            for slug, keywords in SLUG_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        }
        if matched_slugs:
            for idx, chunk in enumerate(self.chunks):
                if chunk.slug in matched_slugs:
                    scores[idx] += 0.30

        # Section-intent boost: "what documents do I need" should surface the
        # "Typical documents required" section over other sections of the
        # same document, which otherwise compete on shared title terms.
        if any(kw in lowered for kw in _DOCUMENT_INTENT_KEYWORDS):
            for idx, chunk in enumerate(self.chunks):
                if chunk.section == "Typical documents required":
                    scores[idx] += 0.35

        if approval_filter:
            # Small boost only — the frontend's "Ask about this" button already
            # writes the approval's name into the question text itself, so this
            # is a tie-breaker for genuinely vague phrasing ("what about this?"),
            # not the primary signal. Kept small so it cannot, by itself, turn
            # an unrelated word into a false match (see the raised threshold
            # below, which is calibrated against this value).
            for idx, chunk in enumerate(self.chunks):
                if chunk.slug == approval_filter:
                    scores[idx] += 0.06

        ranked = scores.argsort()[::-1][:top_k]
        # Threshold calibrated against real queries: genuine approval questions
        # score 0.19-0.67, greetings/small talk/unrelated topics score 0.0-0.14.
        # Set comfortably between the two so boosts alone can't cross it.
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0.15]

    def get_document(self, slug: str) -> List[Chunk]:
        return [c for c in self.chunks if c.slug == slug]

    @property
    def stats(self) -> dict:
        return {
            "documents": len({c.slug for c in self.chunks}),
            "chunks": len(self.chunks),
            "vocabulary": len(self._vectorizer.vocabulary_),
        }


# Module-level singleton — built once on import, reused across requests.
retriever = ComplianceRetriever()
