"""
Production Data Parser
=======================
Reads the Volve production data Excel file.

Source file : Volve production data.xlsx
Columns     : 24
Index       : DATEPRD (daily)
Range       : 2007 – 2016 (multiple wells)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from drillmind.config import get_column_registry, get_settings


def load_production_data(
    filepath: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load and clean the Volve production data.

    Parameters
    ----------
    filepath : str | Path | None
        Path to the Excel file.  Defaults to settings.yaml path.

    Returns
    -------
    pd.DataFrame
        Date-indexed production DataFrame with standardized column names.
    """
    if filepath is None:
        filepath = get_settings().data.production

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Production data not found: {filepath}")

    registry = get_column_registry()

    logger.info("Loading production data: {}", filepath.name)

    df = pd.read_excel(filepath, engine="openpyxl")

    # Rename columns via registry
    rename_map = {col_def.raw: col_def.name for col_def in registry.by_key.values()}
    rename_filtered = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_filtered, inplace=True)

    # Parse date index
    date_col = registry.by_key["production_date"].name  # "date"
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.set_index(date_col, inplace=True)
        df.sort_index(inplace=True)

    # Fix encoding issues in string columns (e.g. M�RSK → MÆRSK)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.encode("latin-1").decode("utf-8", errors="replace")
            if isinstance(x, str)
            else x
        )

    # Get unique wells
    wellbore_col = registry.by_key.get("wellbore_code")
    if wellbore_col and wellbore_col.name in df.columns:
        wells = df[wellbore_col.name].unique()
        logger.info("Wells in production data: {}", list(wells))

    logger.info(
        "Production data loaded: {} rows × {} columns | {} → {}",
        len(df),
        len(df.columns),
        df.index.min(),
        df.index.max(),
    )
    return df
