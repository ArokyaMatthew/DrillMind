"""
ROP & Petrophysics Parser
==========================
Reads the ROP data file from the Volve dataset.

Source file : ROP data .csv
Rows        : 152
Columns     : 8
Index       : Depth (m)

Columns (verified from file inspection):
- Depth       : Measured depth (m)
- WOB         : Weight on bit (units in raw file: likely daN or N)
- SURF_RPM    : Surface RPM
- ROP_AVG     : Average rate of penetration
- PHIF        : Formation porosity (fractional, 0-1)
- VSH         : Volume of shale (fractional, 0-1)
- SW          : Water saturation (fractional, 0-1)
- KLOGH       : Logarithmic permeability estimate (mD)

This data integrates petrophysical formation properties with drilling
parameters — useful for drilling optimization models that correlate
ROP to formation hardness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from drillmind.config import get_settings


# Standardized column names for the ROP file
_ROP_RENAME_MAP = {
    "Depth": "depth_m",
    "WOB": "wob_rop",
    "SURF_RPM": "surface_rpm",
    "ROP_AVG": "rop_avg",
    "PHIF": "porosity",
    "VSH": "volume_shale",
    "SW": "water_saturation",
    "KLOGH": "perm_log",
}


def load_rop_data(
    filepath: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the ROP + petrophysics dataset.

    Parameters
    ----------
    filepath : str | Path | None
        Path to the CSV file.  Defaults to settings.data.rop_log.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by depth (m) with standardized column names.
    """
    if filepath is None:
        settings = get_settings()
        filepath = settings.data.rop_log

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ROP data file not found: {filepath}")

    logger.info("Loading ROP data: {}", filepath.name)

    df = pd.read_csv(filepath, encoding="utf-8")

    # Verify expected columns are present
    expected_cols = set(_ROP_RENAME_MAP.keys())
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        logger.warning("Missing expected columns in ROP data: {}", missing)

    # Rename columns that exist
    rename = {k: v for k, v in _ROP_RENAME_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)

    # Set depth as index
    if "depth_m" in df.columns:
        df.set_index("depth_m", inplace=True)
        df.index.name = "depth_m"

    # Coerce numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        "ROP data loaded: {} rows x {} columns | depth {:.0f} -> {:.0f} m",
        len(df),
        len(df.columns),
        df.index.min(),
        df.index.max(),
    )

    return df
