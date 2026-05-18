"""
DrillMind — Daily Drilling Report (DDR) Parser
================================================
Loads DDRs from HuggingFace ``bengsoon/volve_alpaca`` or raw WITSML XML.

Each DDR is parsed into a ``DDRDocument`` dataclass that carries:
- well name (inferred from content heuristics)
- report date (inferred from sequential ordering)
- activity lines (timestamped operations)
- summary (LLM-generated in the Alpaca dataset)
- metadata for RAG indexing
"""

from __future__ import annotations

import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# DDR Document dataclass
# ---------------------------------------------------------------------------

@dataclass
class DDRActivity:
    """A single timestamped activity from a DDR."""
    time_start: str          # e.g. "00:00"
    time_end: str            # e.g. "06:00"
    description: str         # Raw activity text
    depth_md: Optional[float] = None   # Extracted depth (m MD) if mentioned
    operation: Optional[str] = None    # Classified operation type


@dataclass
class DDRDocument:
    """A parsed Daily Drilling Report."""
    doc_id: str                            # Unique hash for dedup
    well_name: str                         # e.g. "15/9-F-9 A"
    report_index: int                      # Sequential index in dataset
    activities_text: str                    # Full raw activity text
    summary: str                           # Summarized version
    activities: list[DDRActivity] = field(default_factory=list)
    depths_mentioned: list[float] = field(default_factory=list)
    mud_weights: list[float] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict:
        """Metadata dict for ChromaDB storage."""
        return {
            "well_name": self.well_name,
            "report_index": self.report_index,
            "has_depth": len(self.depths_mentioned) > 0,
            "has_mud_weight": len(self.mud_weights) > 0,
            "operations": ",".join(set(self.operations)) if self.operations else "",
            "char_count": len(self.activities_text),
        }


# ---------------------------------------------------------------------------
# Regex patterns for field extraction from DDR text
# ---------------------------------------------------------------------------

# Time range pattern: "00:00 - 06:00:"
_RE_TIME_RANGE = re.compile(
    r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*:\s*(.*?)(?=\d{1,2}:\d{2}\s*-|\Z)",
    re.DOTALL,
)

# Depth mentions: "2202 m", "3450m MD", "1041 m MD"
_RE_DEPTH = re.compile(r"(\d{3,5})\s*m(?:\s*(?:MD|TVD|dpm))?", re.IGNORECASE)

# Mud weight: "1.54 SG", "1,03 SG", "1.20 sg"
_RE_MUD_WEIGHT = re.compile(r"(\d[.,]\d{1,3})\s*SG", re.IGNORECASE)

# Operation classification keywords
_OPERATION_PATTERNS = {
    "drilling": re.compile(r"\b(?:drill(?:ed|ing)?|ream(?:ed|ing)?|ROP)\b", re.I),
    "tripping": re.compile(r"\b(?:TIH|POOH|RIH|trip(?:ped|ping)?|pull(?:ed|ing)?\s*out)\b", re.I),
    "casing": re.compile(r"\b(?:cas(?:ed|ing)|cement(?:ed|ing)?|shoe\s*track)\b", re.I),
    "testing": re.compile(r"\b(?:test(?:ed|ing)?|pressure\s*test|BOP\s*test|FIT|LOT)\b", re.I),
    "circulation": re.compile(r"\b(?:circulat(?:ed|ing|e)|displace?(?:ed|ing)?|condition(?:ed|ing)?)\b", re.I),
    "logging": re.compile(r"\b(?:logg(?:ed|ing)|MWD|LWD|wireline|survey)\b", re.I),
    "completion": re.compile(r"\b(?:complet(?:ed|ion|ing)|perfora(?:ted|tion|ting)|tubing)\b", re.I),
    "bha": re.compile(r"\b(?:BHA|bit|whipstock|stabilizer|MU\s+BHA)\b", re.I),
    "well_control": re.compile(r"\b(?:kick|shut.?in|kill|well\s*control|gas\s*influx|flow\s*check)\b", re.I),
    "maintenance": re.compile(r"\b(?:repair(?:ed|ing)?|maint(?:enance|ained)|wait(?:ed|ing)?|suspend)\b", re.I),
}


def _extract_activities(text: str) -> list[DDRActivity]:
    """Parse timestamped activity lines from DDR text."""
    activities = []
    for match in _RE_TIME_RANGE.finditer(text):
        t_start, t_end, desc = match.group(1), match.group(2), match.group(3).strip()

        # Extract depth if mentioned
        depth_match = _RE_DEPTH.search(desc)
        depth = float(depth_match.group(1)) if depth_match else None

        # Classify operation
        op = None
        for op_name, pattern in _OPERATION_PATTERNS.items():
            if pattern.search(desc):
                op = op_name
                break

        activities.append(DDRActivity(
            time_start=t_start,
            time_end=t_end,
            description=desc,
            depth_md=depth,
            operation=op,
        ))
    return activities


def _extract_depths(text: str) -> list[float]:
    """Extract all depth mentions from DDR text."""
    return [float(m.group(1)) for m in _RE_DEPTH.finditer(text)]


def _extract_mud_weights(text: str) -> list[float]:
    """Extract mud weight values (SG) from DDR text."""
    weights = []
    for m in _RE_MUD_WEIGHT.finditer(text):
        val = m.group(1).replace(",", ".")
        try:
            weights.append(float(val))
        except ValueError:
            continue
    return weights


def _classify_operations(text: str) -> list[str]:
    """Identify all operation types present in DDR text."""
    ops = []
    for op_name, pattern in _OPERATION_PATTERNS.items():
        if pattern.search(text):
            ops.append(op_name)
    return ops


def _make_doc_id(index: int, text: str) -> str:
    """Generate deterministic document ID."""
    h = hashlib.sha256(f"{index}:{text[:200]}".encode()).hexdigest()[:16]
    return f"ddr-{index:04d}-{h}"


# ---------------------------------------------------------------------------
# Well name inference from DDR content
# ---------------------------------------------------------------------------

# Volve well patterns
_RE_WELL = re.compile(r"15/9-F-\d+\s*[A-Z]*", re.IGNORECASE)
_VOLVE_WELLS = [
    "15/9-F-1 C", "15/9-F-4", "15/9-F-5", "15/9-F-9 A",
    "15/9-F-10", "15/9-F-11", "15/9-F-12", "15/9-F-14", "15/9-F-15 D",
]


def _infer_well_name(text: str, index: int) -> str:
    """Try to infer well name from DDR content. Falls back to 'Volve Unknown'."""
    match = _RE_WELL.search(text)
    if match:
        return match.group(0).strip()
    # Default — the Alpaca dataset is sequentially ordered per well
    return "Volve (unspecified)"


# ---------------------------------------------------------------------------
# Public API: Load DDRs
# ---------------------------------------------------------------------------

def load_ddrs_from_huggingface(
    dataset_name: str = "bengsoon/volve_alpaca",
    split: str = "train",
    max_docs: int = 0,
) -> list[DDRDocument]:
    """
    Load DDRs from HuggingFace ``bengsoon/volve_alpaca``.

    Parameters
    ----------
    dataset_name : str
        HuggingFace dataset identifier.
    split : str
        Dataset split to load (``train`` or ``test``).
    max_docs : int
        Maximum number of documents to load. 0 = all.

    Returns
    -------
    list[DDRDocument]
        Parsed DDR documents with extracted fields.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Install 'datasets' package: pip install datasets")
        return []

    logger.info(f"Loading DDRs from HuggingFace: {dataset_name} ({split})")
    ds = load_dataset(dataset_name, split=split)

    if max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    documents: list[DDRDocument] = []
    skipped = 0

    for i, row in enumerate(ds):
        activities_text = row.get("input", "")
        summary = row.get("output", "")

        # Skip empty rows
        if not activities_text or len(activities_text.strip()) < 20:
            skipped += 1
            continue

        doc = DDRDocument(
            doc_id=_make_doc_id(i, activities_text),
            well_name=_infer_well_name(activities_text, i),
            report_index=i,
            activities_text=activities_text,
            summary=summary or "",
            activities=_extract_activities(activities_text),
            depths_mentioned=_extract_depths(activities_text),
            mud_weights=_extract_mud_weights(activities_text),
            operations=_classify_operations(activities_text),
        )
        documents.append(doc)

    logger.info(
        f"DDR loading complete: {len(documents)} documents parsed, "
        f"{skipped} skipped (empty), "
        f"{sum(len(d.depths_mentioned) > 0 for d in documents)} with depths, "
        f"{sum(len(d.mud_weights) > 0 for d in documents)} with mud weights"
    )
    return documents


def load_ddrs_from_xml(xml_dir: str | Path) -> list[DDRDocument]:
    """
    Load DDRs from raw WITSML drillReport XML files.

    Parameters
    ----------
    xml_dir : Path
        Directory containing ``*.xml`` DDR files.

    Returns
    -------
    list[DDRDocument]
        Parsed DDR documents.

    Notes
    -----
    Expects WITSML ``drillReport`` schema. Falls back gracefully
    if XML structure is unexpected.
    """
    import xml.etree.ElementTree as ET

    xml_path = Path(xml_dir)
    if not xml_path.exists():
        logger.warning(f"DDR XML directory not found: {xml_path}")
        return []

    xml_files = sorted(xml_path.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} DDR XML files in {xml_path}")

    documents: list[DDRDocument] = []

    # WITSML namespace
    ns = {"witsml": "http://www.witsml.org/schemas/1series"}

    for idx, xml_file in enumerate(xml_files):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Try to extract well name
            well_el = root.find(".//witsml:nameWell", ns)
            well_name = well_el.text if well_el is not None else "Unknown"

            # Extract activity text from all activity elements
            activity_els = root.findall(".//witsml:activity", ns)
            activities_text = "\n".join(
                el.text.strip() for el in activity_els if el.text
            )

            if not activities_text:
                # Try general text content
                activities_text = ET.tostring(root, encoding="unicode", method="text")

            if len(activities_text.strip()) < 20:
                continue

            doc = DDRDocument(
                doc_id=_make_doc_id(idx, activities_text),
                well_name=well_name,
                report_index=idx,
                activities_text=activities_text,
                summary="",
                activities=_extract_activities(activities_text),
                depths_mentioned=_extract_depths(activities_text),
                mud_weights=_extract_mud_weights(activities_text),
                operations=_classify_operations(activities_text),
            )
            documents.append(doc)

        except ET.ParseError as e:
            logger.warning(f"Failed to parse {xml_file.name}: {e}")
            continue

    logger.info(f"DDR XML loading complete: {len(documents)} documents parsed")
    return documents
