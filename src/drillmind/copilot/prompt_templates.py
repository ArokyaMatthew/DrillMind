"""
Copilot Prompt Templates
=========================
Domain-specific prompts for the drilling query engine.

The system prompt encodes fundamental drilling engineering knowledge
that grounds the LLM's reasoning. The user prompt injects the live
operational context built by context_builder.py.

This is NOT a generic chatbot prompt. Every rule here reflects
real RTOC operational procedures.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """\
You are DrillMind, a drilling data analysis system for Real-Time Operations Center (RTOC) analysts \
monitoring drilling operations on the Norwegian Continental Shelf.

You are currently monitoring well {well_name} in the {field} field, operated by \
{operator}. The well is being drilled with a {bit_diameter}" bit in the {section} section.

## Your Role
- Answer questions about the current drilling state using ONLY the real-time data provided.
- Cite specific sensor values, timestamps, and scores in your answers.
- When discussing anomalies, always reference the anomaly score, event type, and recommended action.
- When discussing drilling performance, reference MSE, d-exponent, and their trends.
- Never speculate beyond what the data shows. If data is unavailable, say so explicitly.

## Drilling Engineering Knowledge
Use this knowledge to interpret sensor readings:

### Anomaly Interpretation
- **Kick indicators**: pit volume gain + gas increase + flow change. CRITICAL — well control event.
- **Lost circulation**: pit volume loss + SPP decrease. Prepare LCM treatment.
- **Stuck pipe**: sudden torque increase + hookload change. Work pipe immediately.
- **Bit dysfunction**: erratic torque with stable WOB. Consider POOH for bit inspection.
- **Connection gas**: gas spike during connection without pit gain. Normal — note depth for geological record.

### KPI Interpretation
- **MSE (Mechanical Specific Energy)**: Energy input per volume of rock. \
  Normal range: 10-100 MPa for North Sea sandstones. High MSE (>200 MPa) = inefficient drilling. \
  MSE > 3x rock UCS = bit is not drilling efficiently.
- **d-exponent**: Normalized drilling rate. Decreasing trend at constant mud weight = increasing pore pressure. \
  Normal range: 0.5-1.5. Sudden drops warrant pore pressure verification.
- **Corrected d-exponent**: d-exp adjusted for mud weight. More reliable pore pressure indicator.

### Rig State Interpretation
- **Drilling**: RPM > 0, WOB applied, pumps on, depth increasing. Active rock destruction.
- **Reaming**: RPM > 0, WOB applied, pumps on, depth NOT increasing. Hole conditioning.
- **Circulating**: Pumps on, no rotation. Hole cleaning or conditioning mud.
- **Connection**: Everything stopped. Making/breaking pipe connection.
- **Tripping in/out**: Running pipe in/out of hole. Monitor hookload for overpull.
- **Static**: Everything off. Monitor for well flow (kick while static).

## Response Format
- Be concise but thorough.
- Always cite the specific data value you're referencing (e.g., "SPP is 158 kPa, which is...").
- If an anomaly is detected, lead with the severity and recommended action.
- Use bullet points for multi-part answers.
"""


def build_system_prompt(well_context: dict[str, Any]) -> str:
    """Fill the system prompt with well metadata."""
    return SYSTEM_PROMPT.format(
        well_name=well_context.get("well_name", "Unknown"),
        field=well_context.get("field", "Unknown"),
        operator=well_context.get("operator", "Unknown"),
        bit_diameter=well_context.get("bit_diameter_inches", "Unknown"),
        section=well_context.get("section", "Unknown"),
    )


USER_PROMPT_TEMPLATE = """\
## Current Drilling Context

### Sensor Snapshot (as of {timestamp})
{sensor_table}

### Anomaly Detection Status
- Current anomaly score: {anomaly_score} (threshold: 0.30)
- Anomaly active: {is_anomaly}
- Total events detected: {total_events}
- Anomaly rate: {anomaly_rate}%

### Recent Anomaly Events
{recent_events}

### Rig State
- Current state: **{rig_state}**
- Total transitions: {total_transitions}

### Drilling KPIs
{kpi_summary}

### State Breakdown
{state_breakdown}

---

## Analyst Question
{question}
"""


def _format_sensor_table(snapshot: dict[str, Any]) -> str:
    """Format sensor readings as a compact table."""
    # Key sensors to show (using verified column names)
    key_sensors = {
        "bit_depth": ("Bit Depth", "m"),
        "tvd": ("TVD", "m"),
        "spp": ("Standpipe Pressure", "kPa"),
        "weight_on_hook": ("Hookload", "kkgf"),
        "torque_averaged": ("Surface Torque", "kN·m"),
        "rpm_avg": ("RPM", "rpm"),
        "wob_avg": ("Weight on Bit", ""),
        "flow_pumps": ("Flow Rate", "L/min"),
        "mud_weight_in": ("Mud Weight In", "sg"),
        "mud_weight_out": ("Mud Weight Out", "sg"),
        "pit_volume_active": ("Active Pit Volume", "m³"),
        "gas_total": ("Total Gas", "%"),
        "casing_pressure": ("Casing Pressure", "kPa"),
    }

    lines = []
    for col, (label, unit) in key_sensors.items():
        val = snapshot.get(col)
        if val is not None:
            lines.append(f"- {label}: {val} {unit}")
        else:
            lines.append(f"- {label}: N/A")

    return "\n".join(lines)


def _format_recent_events(events: list[dict]) -> str:
    """Format recent anomaly events."""
    if not events:
        return "No recent anomaly events."

    lines = []
    for ev in events:
        lines.append(
            f"- [{ev['severity'].upper()}] {ev['type'].replace('_', ' ').title()} "
            f"at {ev['timestamp'][:19]} (score: {ev['score']}) — {ev['description']}"
        )
    return "\n".join(lines)


def _format_kpi_summary(kpis: dict[str, Any]) -> str:
    """Format KPI values."""
    lines = []
    kpi_labels = {
        "mse_mpa": "MSE (Mechanical Specific Energy)",
        "d_exponent": "d-Exponent",
        "d_exponent_corrected": "Corrected d-Exponent",
    }
    for key, label in kpi_labels.items():
        data = kpis.get(key, {})
        if not data.get("available", False):
            reason = data.get("reason", "Not available")
            lines.append(f"- {label}: {reason}")
        else:
            lines.append(
                f"- {label}: current={data['current']}, "
                f"mean={data['mean']}, range=[{data['min']}, {data['max']}]"
            )
    return "\n".join(lines)


def _format_state_breakdown(state_data: dict[str, Any]) -> str:
    """Format rig state breakdown."""
    breakdown = state_data.get("state_breakdown", {})
    lines = []
    for state, info in breakdown.items():
        lines.append(f"- {state}: {info['pct']}% ({info['count']} samples)")
    return "\n".join(lines)


def build_user_prompt(
    context: dict[str, Any],
    question: str,
) -> str:
    """
    Build the complete user prompt with live operational context.

    Parameters
    ----------
    context : dict
        Output of build_full_context() from context_builder.py.
    question : str
        The analyst's natural language question.

    Returns
    -------
    str
        Fully formatted prompt with data context and question.
    """
    snapshot = context.get("snapshot", {})
    anomalies = context.get("anomalies", {})
    rig_state = context.get("rig_state", {})
    kpis = context.get("kpis", {})

    return USER_PROMPT_TEMPLATE.format(
        timestamp=snapshot.get("timestamp", "Unknown"),
        sensor_table=_format_sensor_table(snapshot),
        anomaly_score=anomalies.get("current_anomaly_score", "N/A"),
        is_anomaly="YES ⚠️" if anomalies.get("is_anomaly_now") else "No",
        total_events=anomalies.get("total_events", 0),
        anomaly_rate=anomalies.get("anomaly_rate_pct", 0),
        recent_events=_format_recent_events(anomalies.get("recent_events", [])),
        rig_state=rig_state.get("current_state", "unknown"),
        total_transitions=rig_state.get("total_transitions", 0),
        kpi_summary=_format_kpi_summary(kpis),
        state_breakdown=_format_state_breakdown(rig_state),
        question=question,
    )
