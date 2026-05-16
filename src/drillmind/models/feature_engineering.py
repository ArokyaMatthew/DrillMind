"""
Feature Engineering Pipeline
=============================
Transforms raw time-indexed drilling telemetry into ML-ready features.

This module computes the exact features an RTOC analyst watches for:
- Rolling statistics (mean, std, min, max over configurable windows)
- First derivatives (rate-of-change — are parameters trending up/down?)
- Cross-channel ratios (flow_in vs flow_out, hookload vs WOB)
- Lagged features (what happened N seconds ago?)

Design Principle
----------------
Features are computed ONLY on columns that actually have data.
The feature list is deterministic — given the same input DataFrame shape,
the output will always have the same columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    # Rolling window sizes (in number of rows, ~4-5s per row)
    rolling_windows: tuple[int, ...] = (6, 30, 60)  # ~30s, ~2.5min, ~5min

    # Columns to compute features on — these are the standardized names
    # from column_registry.yaml that have >50% coverage in the Volve data
    target_columns: tuple[str, ...] = (
        "spp",                # Standpipe Pressure — primary kick indicator
        "weight_on_hook",     # Hook Load — stuck pipe indicator
        "torque_averaged",    # Surface Torque — stuck pipe / bit dysfunction
        "rpm_avg",            # Rotary Speed — drilling state
        "flow_pumps",         # Flow Pumps — lost circulation / kick
        "mud_weight_in",      # Mud Weight In — reference
        "mud_weight_out",     # Mud Weight Out — gas cut mud indicator
        "pit_volume_active",  # Pit Volume — kick / lost circ indicator
        "pit_volume_change",  # Pit Volume Change — most sensitive kick indicator
        "casing_pressure",    # Casing Pressure — well control
        "gas_total",          # Total Gas — gas influx
        "wob_avg",            # Weight on Bit — drilling state
        "bit_depth",          # Bit Depth — operational context
        "tvd",                # True Vertical Depth — operational context
        "mud_temp_in",        # Mud Temp In — thermal monitoring
        "mud_temp_out",       # Mud Temp Out — returns temperature
    )


def compute_rolling_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Compute rolling window statistics for target columns.

    For each (column, window) pair, computes:
    - mean, std, min, max over the window
    - This captures whether a parameter is stable, trending, or volatile.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame from ``load_time_log()``.
    config : FeatureConfig | None
        Feature configuration.  Defaults to standard config.

    Returns
    -------
    pd.DataFrame
        Original columns plus rolling features.
    """
    if config is None:
        config = FeatureConfig()

    available = [c for c in config.target_columns if c in df.columns]
    features: dict[str, pd.Series] = {}

    for col in available:
        series = df[col]
        for window in config.rolling_windows:
            prefix = f"{col}_w{window}"
            rolling = series.rolling(window=window, min_periods=1)
            features[f"{prefix}_mean"] = rolling.mean()
            features[f"{prefix}_std"] = rolling.std()
            features[f"{prefix}_min"] = rolling.min()
            features[f"{prefix}_max"] = rolling.max()

    result = pd.DataFrame(features, index=df.index)
    logger.debug(
        "Rolling features: {} columns x {} windows = {} features",
        len(available),
        len(config.rolling_windows),
        len(result.columns),
    )
    return result


def compute_derivative_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Compute first-order derivatives (rate-of-change) for target columns.

    A sudden change in the derivative of SPP, hookload, or pit volume
    is the primary way RTOC analysts detect kicks and stuck pipe.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    config : FeatureConfig | None
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        Derivative features (d/dt approximation via differencing).
    """
    if config is None:
        config = FeatureConfig()

    available = [c for c in config.target_columns if c in df.columns]
    features: dict[str, pd.Series] = {}

    for col in available:
        series = df[col]
        # First difference (instantaneous rate of change)
        features[f"{col}_diff1"] = series.diff(1)
        # 6-sample difference (~30 seconds rate of change)
        features[f"{col}_diff6"] = series.diff(6)
        # 30-sample difference (~2.5 minutes rate of change)
        features[f"{col}_diff30"] = series.diff(30)

    result = pd.DataFrame(features, index=df.index)
    logger.debug("Derivative features: {} columns", len(result.columns))
    return result


def compute_cross_channel_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cross-channel ratios and differences.

    These are domain-specific features that encode physical relationships:
    - mud_weight_out - mud_weight_in: positive means gas-cut mud (kick indicator)
    - spp / flow_pumps: hydraulics ratio
    - torque / rpm: mechanical specific energy proxy

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.

    Returns
    -------
    pd.DataFrame
        Cross-channel features.
    """
    features: dict[str, pd.Series] = {}

    # Mud weight differential — gas cut mud detection
    if "mud_weight_out" in df.columns and "mud_weight_in" in df.columns:
        features["mud_weight_diff"] = df["mud_weight_out"] - df["mud_weight_in"]

    # Temperature differential — formation fluid influx
    if "mud_temp_out" in df.columns and "mud_temp_in" in df.columns:
        features["mud_temp_diff"] = df["mud_temp_out"] - df["mud_temp_in"]

    # SPP per unit flow — hydraulics efficiency
    if "spp" in df.columns and "flow_pumps" in df.columns:
        flow_safe = df["flow_pumps"].replace(0, np.nan)
        features["spp_per_flow"] = df["spp"] / flow_safe

    # Torque per RPM — mechanical specific energy proxy
    if "torque_averaged" in df.columns and "rpm_avg" in df.columns:
        rpm_safe = df["rpm_avg"].replace(0, np.nan)
        features["torque_per_rpm"] = df["torque_averaged"] / rpm_safe

    # WOB normalized by hookload — effective drilling weight
    if "wob_avg" in df.columns and "weight_on_hook" in df.columns:
        hook_safe = df["weight_on_hook"].replace(0, np.nan)
        features["wob_hookload_ratio"] = df["wob_avg"] / hook_safe

    result = pd.DataFrame(features, index=df.index)
    logger.debug("Cross-channel features: {} columns", len(result.columns))
    return result


def build_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for anomaly detection.

    Concatenates:
    1. Raw target columns
    2. Rolling statistics
    3. Derivatives
    4. Cross-channel features

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed drilling DataFrame.
    config : FeatureConfig | None
        Feature configuration.
    drop_warmup : bool
        If True, drops the first ``max(rolling_windows)`` rows where
        rolling features are incomplete.

    Returns
    -------
    pd.DataFrame
        Complete feature matrix, NaN-free (filled forward then backward).
    """
    if config is None:
        config = FeatureConfig()

    available = [c for c in config.target_columns if c in df.columns]
    raw = df[available].copy()

    rolling = compute_rolling_features(df, config)
    derivatives = compute_derivative_features(df, config)
    cross = compute_cross_channel_features(df)

    feature_matrix = pd.concat([raw, rolling, derivatives, cross], axis=1)

    # Drop warmup period where rolling features are incomplete
    if drop_warmup:
        max_window = max(config.rolling_windows)
        feature_matrix = feature_matrix.iloc[max_window:]

    # Fill remaining NaN: forward-fill then backward-fill, then zero
    feature_matrix = feature_matrix.ffill().bfill().fillna(0)

    # Replace infinities with NaN then fill
    feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_matrix = feature_matrix.fillna(0)

    logger.info(
        "Feature matrix built: {} rows x {} features",
        len(feature_matrix),
        len(feature_matrix.columns),
    )
    return feature_matrix
