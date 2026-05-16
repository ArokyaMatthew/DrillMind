"""
Time-Log Parser
================
Reads the time-indexed real-time drilling CSV from the Volve dataset.

Source file : Norway-NA-15_47_9-F-9 A time.csv
Rows        : ~420 000
Columns     : 239  (we select only the RTOC-critical subset)
Interval    : ~4-5 seconds
Range       : 2009-06-27 → 2009-07-16

Design decisions
----------------
1.  We load ONLY the columns listed in column_registry.yaml.
    This avoids OOM on the 428 MB file and keeps the DataFrame lean.
2.  DateTime is parsed from the "DateTime parsed" column (ISO 8601 with tz).
3.  The returned DataFrame uses standardized column names from the registry.
4.  This module NEVER hardcodes raw CSV column names — everything comes
    from the registry.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from drillmind.config import ColumnRegistry, get_column_registry, get_settings


# Categories that correspond to time-indexed RTOC data
_TIME_LOG_CATEGORIES = [
    "index",       # datetime
    "depth",
    "rop",
    "weight",
    "torque",
    "pressure",
    "flow",
    "rotary",
    "mud",
    "pit",
    "gas",
    "directional",
    "downhole",
    "state",
    "pumps",
    "meta",
]


def _get_usecols_and_rename(registry: ColumnRegistry) -> tuple[list[str], dict[str, str]]:
    """
    Build the list of raw CSV columns to read and the rename mapping.

    Returns
    -------
    usecols : list[str]
        Raw CSV headers to pass to ``pd.read_csv(usecols=...)``.
    rename_map : dict[str, str]
        Mapping from raw header → standardized name.
    """
    usecols: list[str] = []
    rename_map: dict[str, str] = {}

    for cat in _TIME_LOG_CATEGORIES:
        for key in registry.get_keys_for_category(cat):
            col_def = registry.by_key[key]
            usecols.append(col_def.raw)
            rename_map[col_def.raw] = col_def.name

    # De-duplicate (some columns might appear in multiple categories — shouldn't happen,
    # but defensive coding).
    seen: set[str] = set()
    deduped: list[str] = []
    for c in usecols:
        if c not in seen:
            deduped.append(c)
            seen.add(c)

    return deduped, rename_map


def load_time_log(
    filepath: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load and clean the time-indexed drilling log.

    Parameters
    ----------
    filepath : str | Path | None
        Path to the CSV.  If ``None``, uses the path from settings.yaml.
    nrows : int | None
        Limit rows (for development / testing).  ``None`` → load all.

    Returns
    -------
    pd.DataFrame
        DateTime-indexed DataFrame with standardized column names.
    """
    if filepath is None:
        filepath = get_settings().data.time_log

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Time log file not found: {filepath}")

    registry = get_column_registry()
    usecols, rename_map = _get_usecols_and_rename(registry)

    # Filter to only columns that actually exist in the file header.
    # (Some columns in the registry may belong to the depth file only.)
    header_line = filepath.open("r", encoding="utf-8").readline()
    available = set(header_line.strip().split(","))

    usecols_present = [c for c in usecols if c in available]
    missing = set(usecols) - set(usecols_present)
    if missing:
        logger.warning(
            "Columns in registry but not in time log ({}): {}",
            len(missing),
            sorted(missing),
        )

    logger.info(
        "Loading time log: {} | selecting {}/{} columns | nrows={}",
        filepath.name,
        len(usecols_present),
        len(available),
        nrows,
    )

    df = pd.read_csv(
        filepath,
        usecols=usecols_present,
        nrows=nrows,
        low_memory=False,
    )

    # Rename to standardized names
    rename_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_filtered, inplace=True)

    # Parse datetime index
    datetime_col = registry.by_key["time_index"].name  # "datetime"
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Drop rows where index is NaT
        nat_count = df.index.isna().sum()
        if nat_count > 0:
            logger.warning("Dropping {} rows with NaT datetime index", nat_count)
            df = df[df.index.notna()]

    # Convert numeric columns — coerce errors to NaN
    for col in df.columns:
        if df[col].dtype == object:
            # Attempt numeric conversion; leave as-is if it's truly text
            converted = pd.to_numeric(df[col], errors="coerce")
            # If > 50% converted successfully, treat as numeric
            if converted.notna().mean() > 0.5:
                df[col] = converted

    logger.info(
        "Time log loaded: {} rows × {} columns | {} → {}",
        len(df),
        len(df.columns),
        df.index.min(),
        df.index.max(),
    )

    return df
