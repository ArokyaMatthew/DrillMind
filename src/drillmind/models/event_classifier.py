"""
Anomaly Event Classifier
=========================
Takes raw anomaly scores from the ensemble detector and classifies them
into drilling problem categories based on which sensor channels are
contributing most to the anomaly.

Drilling Problem Categories
----------------------------
1. **KICK / GAS INFLUX**: Pit gain + flow increase + SPP change + gas increase
2. **LOST CIRCULATION**: Pit loss + flow decrease + SPP drop
3. **STUCK PIPE**: Torque spike + hookload change + no ROP
4. **BIT DYSFUNCTION**: Erratic torque + ROP drop at constant WOB
5. **WASHOUT**: Gradual SPP decrease + flow rate change
6. **UNKNOWN**: Anomaly detected but doesn't match known patterns

Classification is rule-based with configurable thresholds — not ML.
This is intentional: in well control, you need explainable decisions.
An RTOC analyst must be able to trace WHY the system flagged something.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class DrillingEvent(str, Enum):
    """Drilling problem categories."""

    KICK = "kick"
    LOST_CIRCULATION = "lost_circulation"
    STUCK_PIPE = "stuck_pipe"
    BIT_DYSFUNCTION = "bit_dysfunction"
    WASHOUT = "washout"
    CONNECTION_GAS = "connection_gas"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyEvent:
    """A classified anomaly event."""

    timestamp: datetime
    event_type: DrillingEvent
    severity: Severity
    score: float
    contributing_channels: dict[str, float]  # channel -> deviation
    description: str
    recommended_action: str
    duration_rows: int = 1

    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "score": round(self.score, 4),
            "contributing_channels": {
                k: round(v, 4) for k, v in self.contributing_channels.items()
            },
            "description": self.description,
            "recommended_action": self.recommended_action,
            "duration_rows": self.duration_rows,
        }


# Channel groups for pattern matching
_KICK_CHANNELS = {"pit_volume_active", "pit_volume_change", "flow_pumps", "gas_total", "spp"}
_LOST_CIRC_CHANNELS = {"pit_volume_active", "pit_volume_change", "flow_pumps", "spp"}
_STUCK_PIPE_CHANNELS = {"torque_averaged", "weight_on_hook", "wob_avg", "rpm_avg"}
_BIT_DYS_CHANNELS = {"torque_averaged", "wob_avg", "rpm_avg"}


def _compute_channel_deviations(
    feature_row: pd.Series,
    feature_means: pd.Series,
    feature_stds: pd.Series,
    base_columns: list[str],
) -> dict[str, float]:
    """
    Compute z-score deviations for base columns (raw features, not rolling).

    Returns {channel_name: z_score} for channels present in the feature row.
    """
    deviations: dict[str, float] = {}
    for col in base_columns:
        if col in feature_row.index and col in feature_means.index:
            std = feature_stds.get(col, 1.0)
            if std > 0 and not np.isnan(std):
                z = (feature_row[col] - feature_means[col]) / std
                deviations[col] = float(z)
    return deviations


def _classify_single(
    deviations: dict[str, float],
    score: float,
) -> tuple[DrillingEvent, Severity, str, str]:
    """
    Classify a single anomaly based on channel deviations.

    Returns (event_type, severity, description, recommended_action).
    """
    # Check for KICK pattern:
    # - Pit volume increasing (positive deviation)
    # - Gas increasing
    # - SPP may change (increase or decrease depending on kick type)
    pit_dev = deviations.get("pit_volume_change", 0)
    gas_dev = deviations.get("gas_total", 0)
    spp_dev = deviations.get("spp", 0)
    flow_dev = deviations.get("flow_pumps", 0)
    torque_dev = deviations.get("torque_averaged", 0)
    hookload_dev = deviations.get("weight_on_hook", 0)
    wob_dev = deviations.get("wob_avg", 0)
    rpm_dev = deviations.get("rpm_avg", 0)

    # KICK: pit gain + (gas up OR flow up)
    if pit_dev > 2.0 and (gas_dev > 2.0 or flow_dev > 2.0):
        severity = Severity.CRITICAL if pit_dev > 3.0 else Severity.HIGH
        return (
            DrillingEvent.KICK,
            severity,
            f"Possible kick: pit volume increasing (z={pit_dev:.1f}), "
            f"gas={gas_dev:.1f}, flow={flow_dev:.1f}",
            "STOP DRILLING. Check pit levels. Monitor gas readings. "
            "Prepare to shut in well if pit gain continues.",
        )

    # LOST CIRCULATION: pit loss + flow decrease + SPP drop
    if pit_dev < -2.0 and spp_dev < -1.5:
        severity = Severity.HIGH if pit_dev < -3.0 else Severity.MEDIUM
        return (
            DrillingEvent.LOST_CIRCULATION,
            severity,
            f"Possible lost circulation: pit volume decreasing (z={pit_dev:.1f}), "
            f"SPP dropping (z={spp_dev:.1f})",
            "Reduce pump rate. Monitor losses. Prepare LCM (Lost Circulation Material). "
            "Consider pulling off bottom.",
        )

    # STUCK PIPE: torque spike + hookload change
    if abs(torque_dev) > 3.0 and abs(hookload_dev) > 2.0:
        severity = Severity.HIGH if abs(torque_dev) > 4.0 else Severity.MEDIUM
        return (
            DrillingEvent.STUCK_PIPE,
            severity,
            f"Possible stuck pipe: torque deviation (z={torque_dev:.1f}), "
            f"hookload deviation (z={hookload_dev:.1f})",
            "Work pipe gently. DO NOT apply excessive overpull. "
            "Consider circulating spotting fluid. Monitor free-point indicators.",
        )

    # BIT DYSFUNCTION: erratic torque at constant WOB
    if abs(torque_dev) > 2.5 and abs(wob_dev) < 1.0:
        severity = Severity.MEDIUM
        return (
            DrillingEvent.BIT_DYSFUNCTION,
            severity,
            f"Possible bit dysfunction: torque erratic (z={torque_dev:.1f}) "
            f"with stable WOB (z={wob_dev:.1f})",
            "Check ROP trend. Consider changing drilling parameters. "
            "Monitor for whirl or bit bounce. POOH for bit inspection if persistent.",
        )

    # WASHOUT: gradual SPP decrease
    if spp_dev < -2.0 and abs(flow_dev) < 1.0:
        severity = Severity.MEDIUM
        return (
            DrillingEvent.WASHOUT,
            severity,
            f"Possible washout: SPP decreasing (z={spp_dev:.1f}) "
            f"with stable flow (z={flow_dev:.1f})",
            "Monitor SPP trend. Consider POOH for drill string inspection. "
            "Check for cuttings at shakers.",
        )

    # CONNECTION GAS: gas spike without pit gain
    if gas_dev > 3.0 and abs(pit_dev) < 1.0:
        severity = Severity.LOW
        return (
            DrillingEvent.CONNECTION_GAS,
            severity,
            f"Connection gas: gas spike (z={gas_dev:.1f}) without pit gain",
            "Monitor. Likely connection gas from formation exposure during connection. "
            "Note depth and formation for geological record.",
        )

    # UNKNOWN: anomaly detected but doesn't match known patterns
    top_channels = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    top_str = ", ".join(f"{k}={v:.1f}" for k, v in top_channels)
    severity = Severity.MEDIUM if score > 0.5 else Severity.LOW
    return (
        DrillingEvent.UNKNOWN,
        severity,
        f"Anomaly detected: top deviations: {top_str}",
        "Investigate sensor readings. Check for operational context "
        "(connection, tripping, parameter change).",
    )


def classify_anomalies(
    features: pd.DataFrame,
    anomaly_scores: np.ndarray,
    anomaly_mask: np.ndarray,
    base_columns: list[str] | None = None,
    merge_window: int = 30,
) -> list[AnomalyEvent]:
    """
    Classify detected anomalies into drilling problem categories.

    Parameters
    ----------
    features : pd.DataFrame
        The feature matrix (from ``build_feature_matrix``).
    anomaly_scores : np.ndarray
        Combined anomaly scores per row.
    anomaly_mask : np.ndarray
        Binary array: 1 = anomaly, 0 = normal.
    base_columns : list[str] | None
        Raw column names to use for deviation analysis.
        Defaults to the RTOC-critical columns.
    merge_window : int
        Anomalies within this many rows of each other are merged
        into a single event.

    Returns
    -------
    list[AnomalyEvent]
        Classified and merged anomaly events.
    """
    if base_columns is None:
        base_columns = [
            "spp", "weight_on_hook", "torque_averaged", "rpm_avg",
            "flow_pumps", "mud_weight_in", "mud_weight_out",
            "pit_volume_active", "pit_volume_change", "casing_pressure",
            "gas_total", "wob_avg", "bit_depth", "tvd",
        ]
    base_columns = [c for c in base_columns if c in features.columns]

    # Compute reference statistics from non-anomalous data
    normal_mask = anomaly_mask == 0
    if normal_mask.sum() < 100:
        logger.warning("Too few normal samples ({}) for reliable classification", normal_mask.sum())
        feature_means = features[base_columns].mean()
        feature_stds = features[base_columns].std()
    else:
        feature_means = features.loc[normal_mask, base_columns].mean()
        feature_stds = features.loc[normal_mask, base_columns].std()

    # Find contiguous anomaly segments
    anomaly_indices = np.where(anomaly_mask == 1)[0]
    if len(anomaly_indices) == 0:
        return []

    # Merge nearby anomalies into segments
    segments: list[list[int]] = []
    current_segment: list[int] = [anomaly_indices[0]]

    for idx in anomaly_indices[1:]:
        if idx - current_segment[-1] <= merge_window:
            current_segment.append(idx)
        else:
            segments.append(current_segment)
            current_segment = [idx]
    segments.append(current_segment)

    # Classify each segment
    events: list[AnomalyEvent] = []
    for segment in segments:
        peak_idx = segment[np.argmax(anomaly_scores[segment])]
        peak_score = float(anomaly_scores[peak_idx])
        peak_row = features.iloc[peak_idx]
        timestamp = features.index[peak_idx]

        deviations = _compute_channel_deviations(
            peak_row, feature_means, feature_stds, base_columns
        )

        event_type, severity, description, action = _classify_single(deviations, peak_score)

        events.append(AnomalyEvent(
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            score=peak_score,
            contributing_channels=deviations,
            description=description,
            recommended_action=action,
            duration_rows=len(segment),
        ))

    # Sort by severity then score
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    events.sort(key=lambda e: (severity_order[e.severity], -e.score))

    logger.info(
        "Classified {} anomaly events from {} raw detections across {} segments",
        len(events),
        int(anomaly_mask.sum()),
        len(segments),
    )

    # Log summary by type
    from collections import Counter
    type_counts = Counter(e.event_type.value for e in events)
    for event_type, count in type_counts.most_common():
        logger.info("  {}: {} events", event_type, count)

    return events
