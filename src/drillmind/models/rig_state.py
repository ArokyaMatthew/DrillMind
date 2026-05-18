"""
Drilling Activity State Classifier
====================================
Classifies the rig's operational state at each time sample based on
the sensor signature. This is fundamental RTOC analysis — every
real-time monitoring system must know what the rig is *doing* before
it can assess whether something is *wrong*.

States are classified using the IADC standard activity codes, adapted
to the available sensor channels in the Volve time-log dataset.

The classification uses decision-tree logic on the following channels
(all verified present in the Volve time.csv with >99% coverage):

    bit_depth       — is the string moving?
    wob_avg         — is weight being applied?
    rpm_avg         — is the string rotating?
    spp             — are pumps running?
    flow_pumps      — flow rate
    weight_on_hook  — hookload (string in/out of hole indicator)
    torque_averaged — is there rotational resistance?
    pit_volume_active — total active pit volume

Decision logic is based on the physical relationships:
- DRILLING: RPM > 0 AND WOB > threshold AND flow > 0 AND depth increasing
- REAMING:  RPM > 0 AND WOB > threshold AND flow > 0 AND depth NOT increasing
- CIRCULATING: RPM = 0 AND flow > 0 AND string stationary
- CONNECTION: String stationary AND flow = 0 AND depth was recently increasing
- TRIPPING IN: Depth increasing, no RPM, no WOB, no flow
- TRIPPING OUT: Depth decreasing, no RPM, no WOB, no flow
- STATIC: Everything off
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class RigState(str, Enum):
    """IADC-aligned rig activity states."""

    DRILLING = "drilling"
    REAMING = "reaming"
    CIRCULATING = "circulating"
    CONNECTION = "connection"
    TRIPPING_IN = "tripping_in"
    TRIPPING_OUT = "tripping_out"
    WASHING = "washing"          # Rotating + flow but no WOB (backreaming)
    STATIC = "static"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StateThresholds:
    """
    Thresholds for state classification.

    These are calibrated from the actual Volve F-9 A data statistics:
    - rpm_avg mean=13.46, but 0 when not drilling (bimodal)
    - flow_pumps mean=397, but 0 when off
    - wob_avg is near 0 except when drilling
    - spp mean=1737 when pumps on, ~100-200 when off

    All thresholds are intentionally conservative to minimize
    misclassification.
    """

    rpm_active: float = 5.0         # RPM above this = rotating
    flow_active: float = 50.0       # flow_pumps above this = pumping
    spp_active: float = 300.0       # SPP above this = pumps running
    wob_on_bottom: float = 1.0      # wob_avg above this = weight on bit
    depth_change_rate: float = 0.01 # m/sample: bit moving
    hookload_tripping: float = 10.0 # kkgf threshold for tripping detection


def classify_rig_state(
    df: pd.DataFrame,
    thresholds: StateThresholds | None = None,
) -> pd.Series:
    """
    Classify rig activity state for each row in the time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame from ``load_time_log()`` with standardized
        column names.
    thresholds : StateThresholds | None
        Classification thresholds.  Defaults to values calibrated from
        the Volve data distribution.

    Returns
    -------
    pd.Series
        Series of ``RigState`` values, same index as ``df``.

    Raises
    ------
    KeyError
        If required columns are missing from the DataFrame.
    """
    if thresholds is None:
        thresholds = StateThresholds()

    # Verify required columns exist
    required = ["rpm_avg", "flow_pumps", "spp", "wob_avg", "bit_depth", "weight_on_hook"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot classify rig state: missing columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Extract channels
    rpm = df["rpm_avg"].fillna(0)
    flow = df["flow_pumps"].fillna(0)
    spp = df["spp"].fillna(0)
    wob = df["wob_avg"].fillna(0)
    depth = df["bit_depth"].ffill().fillna(0)
    hookload = df["weight_on_hook"].ffill().fillna(0)

    # Compute depth rate of change (m/sample, ~4-5 seconds per sample)
    depth_diff = depth.diff().fillna(0)

    # Binary conditions
    is_rotating = rpm > thresholds.rpm_active
    is_pumping = (flow > thresholds.flow_active) | (spp > thresholds.spp_active)
    has_wob = wob > thresholds.wob_on_bottom
    depth_increasing = depth_diff > thresholds.depth_change_rate
    depth_decreasing = depth_diff < -thresholds.depth_change_rate
    depth_stable = ~depth_increasing & ~depth_decreasing

    # Classify using hierarchical decision tree with exclusive priority
    # Priority order (first match wins):
    # DRILLING > REAMING > WASHING > TRIPPING_IN > TRIPPING_OUT >
    # CIRCULATING > CONNECTION > STATIC > UNKNOWN

    drilling_mask = is_rotating & is_pumping & has_wob & depth_increasing
    reaming_mask = is_rotating & is_pumping & has_wob & ~depth_increasing
    washing_mask = is_rotating & is_pumping & ~has_wob
    trip_in_mask = depth_increasing & ~is_rotating & ~is_pumping
    trip_out_mask = depth_decreasing & ~is_rotating & ~is_pumping
    circ_mask = ~is_rotating & is_pumping & depth_stable

    # CONNECTION: Not rotating + Not pumping + Depth stable + recent drilling
    # (look back 30 samples ~2.5 min for recent drilling activity)
    recent_drilling = drilling_mask.rolling(30, min_periods=1).max().fillna(0).astype(bool)
    conn_mask = ~is_rotating & ~is_pumping & depth_stable & recent_drilling
    static_mask = ~is_rotating & ~is_pumping & depth_stable & ~recent_drilling

    # np.select applies conditions in order — first match wins, no overwrites
    conditions = [
        drilling_mask, reaming_mask, washing_mask,
        trip_in_mask, trip_out_mask,
        circ_mask, conn_mask, static_mask,
    ]
    choices = [
        RigState.DRILLING, RigState.REAMING, RigState.WASHING,
        RigState.TRIPPING_IN, RigState.TRIPPING_OUT,
        RigState.CIRCULATING, RigState.CONNECTION, RigState.STATIC,
    ]
    state_values = np.select(conditions, choices, default=RigState.UNKNOWN)
    states = pd.Series(state_values, index=df.index, dtype=object)

    # Log summary
    from collections import Counter
    counts = Counter(states)
    total = len(states)
    logger.info("Rig state classification: {} samples", total)
    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        logger.info("  {:15s}: {:>7,d} ({:5.1f}%)", state.value if hasattr(state, 'value') else str(state), count, pct)

    return states


def compute_state_transitions(states: pd.Series) -> pd.DataFrame:
    """
    Identify state transitions and compute duration of each state segment.

    Parameters
    ----------
    states : pd.Series
        Output from ``classify_rig_state()``.

    Returns
    -------
    pd.DataFrame
        Columns: start, end, state, duration_samples, duration_seconds
    """
    transitions: list[dict] = []
    current_state = states.iloc[0]
    segment_start = states.index[0]
    segment_start_idx = 0

    for i in range(1, len(states)):
        if states.iloc[i] != current_state:
            segment_end = states.index[i - 1]
            duration_samples = i - segment_start_idx

            # Compute duration in seconds if index is datetime
            if hasattr(segment_end, 'total_seconds'):
                duration_s = (segment_end - segment_start).total_seconds()
            elif hasattr(segment_start, 'timestamp'):
                duration_s = (segment_end - segment_start).total_seconds()
            else:
                duration_s = duration_samples * 4.5  # approximate

            transitions.append({
                "start": segment_start,
                "end": segment_end,
                "state": current_state.value if hasattr(current_state, 'value') else str(current_state),
                "duration_samples": duration_samples,
                "duration_seconds": round(duration_s, 1),
            })

            current_state = states.iloc[i]
            segment_start = states.index[i]
            segment_start_idx = i

    # Append the last segment
    segment_end = states.index[-1]
    duration_samples = len(states) - segment_start_idx
    if hasattr(segment_end, 'timestamp'):
        duration_s = (segment_end - segment_start).total_seconds()
    else:
        duration_s = duration_samples * 4.5
    transitions.append({
        "start": segment_start,
        "end": segment_end,
        "state": current_state.value if hasattr(current_state, 'value') else str(current_state),
        "duration_samples": duration_samples,
        "duration_seconds": round(duration_s, 1),
    })

    result = pd.DataFrame(transitions)
    logger.info(
        "State transitions: {} segments from {} unique states",
        len(result),
        result["state"].nunique(),
    )
    return result
