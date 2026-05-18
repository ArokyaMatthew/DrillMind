"""
DrillMind — Drilling-Specific Document Chunker
================================================
Splits DDR documents into semantically meaningful chunks for embedding.

Chunking strategy:
1. Split by timestamped activity blocks (natural DDR structure)
2. Group short activities into chunks of ~500 tokens
3. Preserve metadata: well name, report index, operation type
4. Add summary as a separate chunk (for retrieval of high-level context)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from drillmind.parsers.ddr_parser import DDRDocument


@dataclass
class DDRChunk:
    """A chunk of DDR text ready for embedding."""
    chunk_id: str                   # Unique identifier
    doc_id: str                     # Parent document ID
    text: str                       # Chunk text for embedding
    well_name: str                  # Well identifier
    report_index: int               # DDR sequential index
    chunk_type: str                 # "activity" or "summary"
    operations: list[str]           # Operation types in this chunk
    depths: list[float]             # Depths mentioned in this chunk
    token_count: int                # Approximate token count

    @property
    def metadata(self) -> dict:
        """Metadata for ChromaDB."""
        return {
            "doc_id": self.doc_id,
            "well_name": self.well_name,
            "report_index": self.report_index,
            "chunk_type": self.chunk_type,
            "operations": ",".join(self.operations) if self.operations else "",
            "has_depth": len(self.depths) > 0,
            "min_depth": min(self.depths) if self.depths else -1.0,
            "max_depth": max(self.depths) if self.depths else -1.0,
        }


# Approximate token count (words / 0.75)
def _approx_tokens(text: str) -> int:
    return int(len(text.split()) / 0.75)


# Split DDR text by time entries
_RE_TIME_SPLIT = re.compile(r"(?=\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*:)")

# Depth extraction
_RE_DEPTH = re.compile(r"(\d{3,5})\s*m(?:\s*(?:MD|TVD|dpm))?", re.IGNORECASE)

# Operation keywords
_OPERATION_KEYWORDS = {
    "drilling": re.compile(r"\b(?:drill(?:ed|ing)?|ream(?:ed|ing)?)\b", re.I),
    "tripping": re.compile(r"\b(?:TIH|POOH|RIH|trip)\b", re.I),
    "casing": re.compile(r"\b(?:cas(?:ed|ing)|cement)\b", re.I),
    "testing": re.compile(r"\b(?:test(?:ed|ing)?|BOP|FIT|LOT)\b", re.I),
    "circulation": re.compile(r"\b(?:circulat|displace?)\b", re.I),
    "completion": re.compile(r"\b(?:complet|perfora|tubing)\b", re.I),
    "well_control": re.compile(r"\b(?:kick|shut.?in|kill|well\s*control)\b", re.I),
}


def _detect_ops(text: str) -> list[str]:
    """Detect operation types in chunk text."""
    return [name for name, pat in _OPERATION_KEYWORDS.items() if pat.search(text)]


def _extract_depths(text: str) -> list[float]:
    """Extract depth values from text."""
    return [float(m.group(1)) for m in _RE_DEPTH.finditer(text)]


def chunk_ddr(
    doc: DDRDocument,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[DDRChunk]:
    """
    Split a DDR document into chunks for embedding.

    Strategy:
    1. Split by timestamped activity entries
    2. Group adjacent entries until ~max_tokens
    3. Add summary as separate chunk
    """
    chunks: list[DDRChunk] = []
    chunk_idx = 0

    # --- Activity chunks ---
    activity_blocks = _RE_TIME_SPLIT.split(doc.activities_text)
    activity_blocks = [b.strip() for b in activity_blocks if b.strip()]

    if not activity_blocks:
        # Single block if no time entries found
        if doc.activities_text.strip():
            activity_blocks = [doc.activities_text.strip()]

    # Group blocks into chunks
    current_text = ""
    current_tokens = 0

    for block in activity_blocks:
        block_tokens = _approx_tokens(block)

        if current_tokens + block_tokens > max_tokens and current_text:
            # Emit current chunk
            chunks.append(DDRChunk(
                chunk_id=f"{doc.doc_id}-act-{chunk_idx:02d}",
                doc_id=doc.doc_id,
                text=current_text.strip(),
                well_name=doc.well_name,
                report_index=doc.report_index,
                chunk_type="activity",
                operations=_detect_ops(current_text),
                depths=_extract_depths(current_text),
                token_count=current_tokens,
            ))
            chunk_idx += 1

            # Overlap: keep last block
            if overlap_tokens > 0 and block_tokens < overlap_tokens * 2:
                current_text = block + "\n"
                current_tokens = block_tokens
            else:
                current_text = ""
                current_tokens = 0

        current_text += block + "\n"
        current_tokens += block_tokens

    # Emit remaining
    if current_text.strip():
        chunks.append(DDRChunk(
            chunk_id=f"{doc.doc_id}-act-{chunk_idx:02d}",
            doc_id=doc.doc_id,
            text=current_text.strip(),
            well_name=doc.well_name,
            report_index=doc.report_index,
            chunk_type="activity",
            operations=_detect_ops(current_text),
            depths=_extract_depths(current_text),
            token_count=current_tokens,
        ))

    # --- Summary chunk ---
    if doc.summary and len(doc.summary.strip()) > 10:
        chunks.append(DDRChunk(
            chunk_id=f"{doc.doc_id}-sum-00",
            doc_id=doc.doc_id,
            text=f"[DDR Summary] {doc.summary.strip()}",
            well_name=doc.well_name,
            report_index=doc.report_index,
            chunk_type="summary",
            operations=_detect_ops(doc.summary),
            depths=_extract_depths(doc.summary),
            token_count=_approx_tokens(doc.summary),
        ))

    return chunks


def chunk_all_ddrs(
    documents: list[DDRDocument],
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[DDRChunk]:
    """
    Chunk all DDR documents.

    Returns
    -------
    list[DDRChunk]
        All chunks from all documents, ready for embedding.
    """
    all_chunks: list[DDRChunk] = []

    for doc in documents:
        doc_chunks = chunk_ddr(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        all_chunks.extend(doc_chunks)

    # Statistics
    activity_chunks = sum(1 for c in all_chunks if c.chunk_type == "activity")
    summary_chunks = sum(1 for c in all_chunks if c.chunk_type == "summary")
    avg_tokens = sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1)

    logger.info(
        f"Chunking complete: {len(all_chunks)} chunks from {len(documents)} DDRs "
        f"({activity_chunks} activity, {summary_chunks} summary, "
        f"avg {avg_tokens:.0f} tokens/chunk)"
    )

    return all_chunks
