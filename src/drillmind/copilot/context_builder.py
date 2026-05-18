"""
Copilot Context Builder
========================
Constructs structured, data-grounded context from the live
application state — sensor readings, anomaly events, rig state,
KPIs, and well metadata.

This is the core differentiator: the copilot doesn't hallucinate
answers — every claim is backed by a specific sensor reading,
anomaly score, or calculated KPI from the running pipeline.

The context builder produces a structured dict that can be
serialized into a prompt template for any LLM backend.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def build_current_snapshot(
    time_df: pd.DataFrame,
    row_index: int = -1,
) -> dict[str, Any]:
    """
    Build a snapshot of the current drilling state from sensor readings.

    Parameters
    ----------
    time_df : pd.DataFrame
        Time-indexed DataFrame from load_time_log().
    row_index : int
        Row to use as "current" (-1 = last row).

    Returns
    -------
    dict
        Keys are standardized column names, values are floats.
    """
    row = time_df.iloc[row_index]
    timestamp = time_df.index[row_index]

    snapshot = {
        "timestamp": str(timestamp),
        "row_index": row_index if row_index >= 0 else len(time_df) + row_index,
    }

    # Include all numeric sensor values
    for col in time_df.columns:
        val = row[col]
        if isinstance(val, (int, float, np.integer, np.floating)):
            if pd.notna(val) and np.isfinite(val):
                snapshot[col] = round(float(val), 4)
            else:
                snapshot[col] = None
        elif isinstance(val, str):
            snapshot[col] = val

    return snapshot


def build_anomaly_context(
    events: list,
    anomaly_details: dict[str, np.ndarray],
    features: pd.DataFrame,
    n_recent: int = 5,
) -> dict[str, Any]:
    """
    Build anomaly detection context: recent events, overall stats,
    and current anomaly score.

    Parameters
    ----------
    events : list
        List of AnomalyEvent objects from event_classifier.
    anomaly_details : dict
        Output of ensemble.score_with_details().
    features : pd.DataFrame
        Feature matrix (for index alignment).
    n_recent : int
        Number of most recent events to include.

    Returns
    -------
    dict
        Anomaly context for prompt construction.
    """
    total_anomalous = int(anomaly_details["is_anomaly"].sum())
    total_samples = len(anomaly_details["is_anomaly"])
    anomaly_rate = total_anomalous / max(total_samples, 1)

    # Current (latest) anomaly score
    current_score = float(anomaly_details["combined"][-1])
    is_anomaly_now = bool(anomaly_details["is_anomaly"][-1])

    # Event type breakdown
    type_counts = Counter(e.event_type.value for e in events)
    severity_counts = Counter(e.severity.value for e in events)

    # Recent events (last N)
    recent = []
    for ev in events[-n_recent:]:
        recent.append({
            "type": ev.event_type.value,
            "severity": ev.severity.value,
            "timestamp": str(ev.timestamp),
            "score": round(ev.score, 4),
            "description": ev.description,
            "recommended_action": ev.recommended_action,
        })

    return {
        "current_anomaly_score": round(current_score, 4),
        "is_anomaly_now": is_anomaly_now,
        "total_events": len(events),
        "total_anomalous_samples": total_anomalous,
        "total_samples": total_samples,
        "anomaly_rate_pct": round(anomaly_rate * 100, 2),
        "event_types": dict(type_counts),
        "severity_breakdown": dict(severity_counts),
        "recent_events": recent,
    }


def build_rig_state_context(
    rig_states: pd.Series,
    transitions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Build rig state context: current state, time in state, breakdown.

    Parameters
    ----------
    rig_states : pd.Series
        Output of classify_rig_state().
    transitions : pd.DataFrame | None
        Output of compute_state_transitions().

    Returns
    -------
    dict
        Rig state context.
    """
    current_state = rig_states.iloc[-1]
    state_value = current_state.value if hasattr(current_state, "value") else str(current_state)

    # State breakdown
    counts = Counter(s.value if hasattr(s, "value") else str(s) for s in rig_states)
    total = len(rig_states)
    breakdown = {
        state: {"count": count, "pct": round(100 * count / total, 1)}
        for state, count in sorted(counts.items(), key=lambda x: -x[1])
    }

    result = {
        "current_state": state_value,
        "total_samples": total,
        "state_breakdown": breakdown,
    }

    if transitions is not None and len(transitions) > 0:
        last_transition = transitions.iloc[-1]
        result["last_transition"] = {
            "from_state": transitions.iloc[-2]["state"] if len(transitions) > 1 else "unknown",
            "to_state": last_transition["state"],
            "duration_seconds": last_transition["duration_seconds"],
        }
        result["total_transitions"] = len(transitions)

    return result


def build_kpi_context(
    kpi_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build drilling KPI context.

    Parameters
    ----------
    kpi_df : pd.DataFrame
        Output of compute_drilling_kpis().

    Returns
    -------
    dict
        KPI context with current values and statistics.
    """
    result = {}
    for col in kpi_df.columns:
        series = kpi_df[col].dropna()
        if len(series) == 0:
            result[col] = {"available": False, "reason": "No valid values (rig not drilling)"}
            continue

        result[col] = {
            "available": True,
            "current": round(float(series.iloc[-1]), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "percentile_25": round(float(series.quantile(0.25)), 4),
            "percentile_75": round(float(series.quantile(0.75)), 4),
        }

    return result


def build_well_context(settings) -> dict[str, Any]:
    """Build well metadata context."""
    return {
        "well_name": settings.well,
        "field": settings.field_name,
        "operator": settings.operator,
        "bit_diameter_inches": 12.25,
        "section": "12¼ inch",
        "location": "Norwegian North Sea, Block 15/9",
    }


def build_full_context(
    time_df: pd.DataFrame,
    events: list,
    anomaly_details: dict,
    features: pd.DataFrame,
    rig_states: pd.Series,
    transitions: pd.DataFrame | None,
    kpi_df: pd.DataFrame,
    settings,
) -> dict[str, Any]:
    """
    Build the complete copilot context from all data sources.

    This is the single entry point used by the copilot engine.

    Returns
    -------
    dict
        Full context with keys:
        - well: well metadata
        - snapshot: current sensor readings
        - anomalies: anomaly detection results
        - rig_state: current rig activity
        - kpis: drilling performance metrics
    """
    return {
        "well": build_well_context(settings),
        "snapshot": build_current_snapshot(time_df),
        "anomalies": build_anomaly_context(events, anomaly_details, features),
        "rig_state": build_rig_state_context(rig_states, transitions),
        "kpis": build_kpi_context(kpi_df),
    }
