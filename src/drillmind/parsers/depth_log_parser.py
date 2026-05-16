"""
Depth-Log Parser
=================
Reads the depth-indexed drilling log from the Volve dataset.

Source file : Norway-NA-15_47_9-F-9 A depth.csv
Columns     : 115
Index       : Measured Depth (m)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from drillmind.config import get_column_registry, get_settings


def load_depth_log(
    filepath: str | Path | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load the depth-indexed drilling log.

    Parameters
    ----------
    filepath : str | Path | None
        Path to the depth CSV.  Defaults to settings.yaml path.
    nrows : int | None
        Row limit for dev/testing.

    Returns
    -------
    pd.DataFrame
        Measured-depth-indexed DataFrame with standardized column names
        where available, raw names otherwise.
    """
    if filepath is None:
        filepath = get_settings().data.depth_log

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Depth log not found: {filepath}")

    registry = get_column_registry()

    logger.info("Loading depth log: {} | nrows={}", filepath.name, nrows)

    df = pd.read_csv(filepath, nrows=nrows, low_memory=False)

    # Drop the unnamed index columns that come from the original WITSML→CSV conversion
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
        logger.debug("Dropped {} unnamed index columns", len(unnamed_cols))

    # Rename columns that exist in the registry
    rename_map = {col_def.raw: col_def.name for col_def in registry.by_key.values()}
    rename_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_filtered, inplace=True)

    # Set measured depth as index
    depth_col = registry.by_key["depth_index"].name  # "measured_depth"
    if depth_col in df.columns:
        df.set_index(depth_col, inplace=True)
        df.sort_index(inplace=True)

    # Coerce numeric
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.5:
                df[col] = converted

    logger.info(
        "Depth log loaded: {} rows × {} columns | depth {} → {} m",
        len(df),
        len(df.columns),
        df.index.min(),
        df.index.max(),
    )
    return df
