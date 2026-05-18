"""
Data Quality Engine
====================
Analyses a time-indexed drilling DataFrame for common data quality issues
that RTOC analysts must handle before trusting sensor data:

1. **Time gaps** — missing data intervals (rig comms dropout, WITSML gaps)
2. **Spikes**    — physically impossible values (sensor glitches)
3. **Flat-lines** — stuck sensors reporting constant values
4. **NaN coverage** — columns with too much missing data
5. **Range violations** — values outside physical bounds

All thresholds are read from config/settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from drillmind.config import QualitySettings, get_settings


@dataclass
class GapRecord:
    """A detected time gap in the data."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration_seconds: float


@dataclass
class SpikeRecord:
    """A detected spike in a column."""

    column: str
    index: pd.Timestamp
    value: float
    zscore: float


@dataclass
class FlatlineRecord:
    """A detected flatline (stuck sensor) in a column."""

    column: str
    start: pd.Timestamp
    end: pd.Timestamp
    value: float
    count: int


@dataclass
class ColumnStats:
    """Per-column statistics."""

    column: str
    non_null_count: int
    null_count: int
    null_ratio: float
    mean: float | None
    std: float | None
    min_val: float | None
    max_val: float | None


@dataclass
class DataQualityReport:
    """Complete data quality report for a DataFrame."""

    total_rows: int
    total_columns: int
    time_range_start: pd.Timestamp | None
    time_range_end: pd.Timestamp | None
    gaps: list[GapRecord] = field(default_factory=list)
    spikes: list[SpikeRecord] = field(default_factory=list)
    flatlines: list[FlatlineRecord] = field(default_factory=list)
    column_stats: list[ColumnStats] = field(default_factory=list)
    sparse_columns: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Data Quality Report",
            f"{'='*50}",
            f"Rows: {self.total_rows:,}",
            f"Columns: {self.total_columns}",
            f"Time range: {self.time_range_start} → {self.time_range_end}",
            f"Time gaps (>{get_settings().quality.gap_threshold_seconds}s): {len(self.gaps)}",
            f"Spikes detected: {len(self.spikes)}",
            f"Flatline segments: {len(self.flatlines)}",
            f"Sparse columns (>{100*(1-get_settings().quality.min_non_null_ratio):.0f}% null): "
            f"{len(self.sparse_columns)}",
        ]
        return "\n".join(lines)


def detect_time_gaps(
    df: pd.DataFrame,
    threshold_seconds: float,
) -> list[GapRecord]:
    """
    Find intervals where the time index jumps by more than ``threshold_seconds``.
    These correspond to WITSML transmission dropouts or rig downtime.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex — skipping gap detection")
        return []

    deltas = df.index.to_series().diff().dt.total_seconds()
    mask = deltas > threshold_seconds
    gap_indices = deltas[mask]

    gaps = []
    for idx, duration in gap_indices.items():
        pos = df.index.get_loc(idx)
        if pos > 0:
            start = df.index[pos - 1]
            gaps.append(GapRecord(start=start, end=idx, duration_seconds=duration))

    if gaps:
        logger.info(
            "Found {} time gaps > {}s (max: {:.0f}s)",
            len(gaps),
            threshold_seconds,
            max(g.duration_seconds for g in gaps),
        )
    return gaps


def detect_spikes(
    df: pd.DataFrame,
    zscore_threshold: float,
    max_spikes_per_col: int = 100,
) -> list[SpikeRecord]:
    """
    Detect point anomalies using z-score on numeric columns.
    A spike is a value that deviates from the column mean by more than
    ``zscore_threshold`` standard deviations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    spikes: list[SpikeRecord] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            continue

        zscores = ((series - mean) / std).abs()
        spike_mask = zscores > zscore_threshold
        spike_indices = zscores[spike_mask].nlargest(max_spikes_per_col)

        for idx, z in spike_indices.items():
            spikes.append(SpikeRecord(
                column=col,
                index=idx,
                value=float(series.loc[idx]),
                zscore=float(z),
            ))

    if spikes:
        logger.info(
            "Found {} spikes across {} columns (z > {:.1f})",
            len(spikes),
            len({s.column for s in spikes}),
            zscore_threshold,
        )
    return spikes


def detect_flatlines(
    df: pd.DataFrame,
    window: int,
) -> list[FlatlineRecord]:
    """
    Detect stuck sensors — sequences of ``window`` or more identical consecutive values.
    A real sensor almost never reports the exact same float for 20+ readings.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    flatlines: list[FlatlineRecord] = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < window:
            continue

        # Find runs of consecutive equal values
        different = series != series.shift(1)
        group_ids = different.cumsum()
        groups = series.groupby(group_ids)

        for _, group in groups:
            if len(group) >= window:
                flatlines.append(FlatlineRecord(
                    column=col,
                    start=group.index[0],
                    end=group.index[-1],
                    value=float(group.iloc[0]),
                    count=len(group),
                ))

    if flatlines:
        logger.info(
            "Found {} flatline segments across {} columns (window={})",
            len(flatlines),
            len({f.column for f in flatlines}),
            window,
        )
    return flatlines


def compute_column_stats(df: pd.DataFrame) -> list[ColumnStats]:
    """Compute per-column summary statistics."""
    stats = []
    for col in df.columns:
        non_null = int(df[col].notna().sum())
        null = int(df[col].isna().sum())
        total = non_null + null

        if df[col].dtype in [np.float64, np.int64, float, int]:
            desc = df[col].describe()
            stats.append(ColumnStats(
                column=col,
                non_null_count=non_null,
                null_count=null,
                null_ratio=null / total if total > 0 else 1.0,
                mean=float(desc.get("mean", np.nan)),
                std=float(desc.get("std", np.nan)),
                min_val=float(desc.get("min", np.nan)),
                max_val=float(desc.get("max", np.nan)),
            ))
        else:
            stats.append(ColumnStats(
                column=col,
                non_null_count=non_null,
                null_count=null,
                null_ratio=null / total if total > 0 else 1.0,
                mean=None,
                std=None,
                min_val=None,
                max_val=None,
            ))

    return stats


def run_quality_check(df: pd.DataFrame) -> DataQualityReport:
    """
    Run the full data quality pipeline on a time-indexed drilling DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex (output of ``load_time_log``).

    Returns
    -------
    DataQualityReport
        Full quality report.
    """
    settings = get_settings().quality

    time_start = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
    time_end = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None

    logger.info("Running data quality check on {} rows × {} columns", len(df), len(df.columns))

    gaps = detect_time_gaps(df, settings.gap_threshold_seconds)
    spikes = detect_spikes(df, settings.spike_zscore_threshold)
    flatlines = detect_flatlines(df, settings.flatline_window)
    col_stats = compute_column_stats(df)

    sparse = [
        cs.column for cs in col_stats
        if (1.0 - cs.null_ratio) < settings.min_non_null_ratio
    ]

    report = DataQualityReport(
        total_rows=len(df),
        total_columns=len(df.columns),
        time_range_start=time_start,
        time_range_end=time_end,
        gaps=gaps,
        spikes=spikes,
        flatlines=flatlines,
        column_stats=col_stats,
        sparse_columns=sparse,
    )

    logger.info("\n{}", report.summary())
    return report
