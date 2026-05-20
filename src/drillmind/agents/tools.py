"""
DrillMind — Domain Tools
============================================
Eight tools that the agent orchestrator can invoke to gather evidence
for answering drilling questions.  Each tool is a pure function that
takes the application state dict and optional parameters, and returns
a structured evidence dict.

Inspired by TADI (Lu, 2026) but adapted for DrillMind's real-time
architecture — every tool can access live sensor data, anomaly events,
and the RAG store.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, parameters: dict | None = None):
    """Decorator to register a tool function."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
            "function": fn,
        }
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@register_tool(
    name="get_current_sensors",
    description=(
        "Get the latest sensor readings from the drilling rig. "
        "Returns current values for SPP, hookload, torque, RPM, WOB, "
        "bit depth, mud weight, gas levels, and pit volume."
    ),
)
def get_current_sensors(state: dict, **kwargs) -> dict:
    """Return latest sensor snapshot from time_df."""
    time_df = state.get("time_df")
    if time_df is None or time_df.empty:
        return {"error": "No telemetry data loaded"}

    last = time_df.iloc[-1]
    sensor_cols = [
        "spp", "weight_on_hook", "torque_averaged", "rpm_avg",
        "wob_avg", "bit_depth", "tvd", "mud_weight_in", "mud_weight_out",
        "gas_total", "pit_volume_active", "flow_pumps", "rop",
    ]

    readings = {}
    for col in sensor_cols:
        if col in last.index:
            val = last[col]
            readings[col] = round(float(val), 3) if not _is_nan(val) else None

    # Timestamp
    ts_col = "timestamp" if "timestamp" in time_df.columns else None
    if ts_col:
        readings["timestamp"] = str(last[ts_col])
    elif time_df.index.name and "time" in time_df.index.name.lower():
        readings["timestamp"] = str(last.name)

    return {"sensors": readings, "total_rows": len(time_df)}


@register_tool(
    name="get_anomaly_status",
    description=(
        "Get the current anomaly detection status including the latest "
        "anomaly score, whether an anomaly is active, and the most recent "
        "classified anomaly events with their types and severities."
    ),
)
def get_anomaly_status(state: dict, **kwargs) -> dict:
    """Return anomaly status from detection pipeline."""
    anomaly_details = state.get("anomaly_details")  # dict[str, np.ndarray]
    events = state.get("events", [])  # list[AnomalyEvent]

    result = {"anomaly_active": False, "score": 0.0, "recent_events": []}

    if anomaly_details is not None and "combined" in anomaly_details:
        score = float(anomaly_details["combined"][-1])
        result["score"] = round(score, 4)
        result["anomaly_active"] = bool(anomaly_details["is_anomaly"][-1])

    # Top 5 most recent events (events are AnomalyEvent dataclass instances)
    if events:
        sorted_events = sorted(events, key=lambda e: e.score, reverse=True)
        for evt in sorted_events[:5]:
            result["recent_events"].append({
                "type": evt.event_type.value if hasattr(evt.event_type, 'value') else str(evt.event_type),
                "severity": evt.severity.value if hasattr(evt.severity, 'value') else str(evt.severity),
                "score": round(evt.score, 4),
                "timestamp": str(evt.timestamp),
                "description": evt.description,
                "action": evt.recommended_action,
            })

    result["total_events"] = len(events)
    return result


@register_tool(
    name="get_rig_state",
    description=(
        "Get the current rig state classification and time breakdown. "
        "Returns the current state (drilling, circulating, tripping, static, etc.) "
        "and percentage of time spent in each state."
    ),
)
def get_rig_state(state: dict, **kwargs) -> dict:
    """Return rig state classification results."""
    rig_states = state.get("rig_states")
    transitions = state.get("transitions")

    result = {"current_state": "unknown", "breakdown": {}}

    if rig_states is not None and not rig_states.empty:
        result["current_state"] = str(rig_states.iloc[-1])

        # Percentage breakdown
        counts = rig_states.value_counts()
        total = len(rig_states)
        for s, cnt in counts.items():
            result["breakdown"][str(s)] = {
                "count": int(cnt),
                "pct": round(100 * cnt / total, 1),
            }

    if transitions is not None and not transitions.empty:
        result["total_transitions"] = len(transitions)
        # Last 5 transitions
        recent = transitions.tail(5).to_dict("records")
        result["recent_transitions"] = [
            {k: str(v) for k, v in rec.items()} for rec in recent
        ]

    return result


@register_tool(
    name="get_drilling_kpis",
    description=(
        "Get drilling performance KPIs: Mechanical Specific Energy (MSE), "
        "d-exponent, and corrected d-exponent. These indicate bit efficiency "
        "and pore pressure trends."
    ),
)
def get_drilling_kpis(state: dict, **kwargs) -> dict:
    """Return drilling KPI values."""
    kpi_df = state.get("kpi_df")

    result = {"mse": None, "d_exponent": None, "d_exponent_corrected": None}

    if kpi_df is None or kpi_df.empty:
        return {"kpis": result, "note": "KPI data not available (no drilling activity in loaded window)"}

    import numpy as np

    for col, key in [("mse_mpa", "mse"), ("d_exponent", "d_exponent"), ("d_exponent_corrected", "d_exponent_corrected")]:
        if col in kpi_df.columns:
            valid = kpi_df[col].dropna()
            if len(valid) > 0:
                result[key] = {
                    "current": round(float(valid.iloc[-1]), 3),
                    "mean": round(float(valid.mean()), 3),
                    "std": round(float(valid.std()), 3),
                    "min": round(float(valid.min()), 3),
                    "max": round(float(valid.max()), 3),
                    "valid_count": int(len(valid)),
                }

    return {"kpis": result}


@register_tool(
    name="search_ddr",
    description=(
        "Search Daily Drilling Reports (DDRs) for historical information. "
        "Use this to find past drilling events, mud weights used, BHA descriptions, "
        "casing operations, well control incidents, and operational history. "
        "Provide a natural language query describing what you're looking for."
    ),
    parameters={"query": "Search query string", "well_filter": "Optional well name filter"},
)
def search_ddr(state: dict, query: str = "", well_filter: str = None, **kwargs) -> dict:
    """Search DDR vector store."""
    rag_store = state.get("rag_store")

    if rag_store is None:
        return {"error": "DDR RAG store not initialized", "results": []}

    if not query:
        return {"error": "No search query provided", "results": []}

    results = rag_store.search(query=query, top_k=5, well_filter=well_filter)

    return {
        "query": query,
        "results": [r.to_dict() for r in results],
        "total_ddrs_indexed": rag_store.count,
    }


@register_tool(
    name="query_production",
    description=(
        "Query production data for Volve field wells. "
        "Returns oil/gas/water production volumes by well and date range."
    ),
    parameters={"well": "Optional well name filter"},
)
def query_production(state: dict, well: str = None, **kwargs) -> dict:
    """Query production data."""
    prod_df = state.get("production_df")

    if prod_df is None or prod_df.empty:
        return {"error": "Production data not loaded"}

    result = {"wells": [], "total_records": len(prod_df)}

    # Well list
    well_col = None
    for candidate in ["WELL_BORE_CODE", "wellbore", "well"]:
        if candidate in prod_df.columns:
            well_col = candidate
            break

    if well_col:
        wells = prod_df[well_col].unique().tolist()
        result["wells"] = [str(w) for w in wells]

    # If specific well requested, filter
    if well and well_col:
        filtered = prod_df[prod_df[well_col].str.contains(well, case=False, na=False)]
        if not filtered.empty:
            result["filtered_records"] = len(filtered)
            # Summary stats for numeric columns
            numeric = filtered.select_dtypes(include="number")
            if not numeric.empty:
                result["summary"] = {
                    col: {"mean": round(float(numeric[col].mean()), 2),
                          "total": round(float(numeric[col].sum()), 2)}
                    for col in numeric.columns[:6]  # Limit columns
                }

    return result


@register_tool(
    name="get_data_quality",
    description=(
        "Get the data quality report for the current telemetry. "
        "Shows detected gaps, spikes, flatlines, and sparse columns."
    ),
)
def get_data_quality(state: dict, **kwargs) -> dict:
    """Return data quality report."""
    quality = state.get("quality_report")  # DataQualityReport dataclass

    if quality is None:
        return {"error": "Quality report not available"}

    # Extract key metrics (quality is a DataQualityReport dataclass, not a dict)
    return {
        "total_rows": quality.total_rows,
        "total_columns": quality.total_columns,
        "time_gaps": len(quality.gaps),
        "spikes_detected": len(quality.spikes),
        "flatline_segments": len(quality.flatlines),
        "sparse_columns": len(quality.sparse_columns),
    }


@register_tool(
    name="compare_wells",
    description=(
        "Compare drilling parameters or production across different Volve wells. "
        "Useful for offset well analysis."
    ),
    parameters={"metric": "What to compare (production, depth, etc.)"},
)
def compare_wells(state: dict, metric: str = "production", **kwargs) -> dict:
    """Cross-well comparison."""
    prod_df = state.get("production_df")

    if prod_df is None or prod_df.empty:
        return {"error": "No multi-well data available for comparison"}

    # Find well column
    well_col = None
    for candidate in ["WELL_BORE_CODE", "wellbore", "well"]:
        if candidate in prod_df.columns:
            well_col = candidate
            break

    if not well_col:
        return {"error": "Cannot identify well column in production data"}

    wells = prod_df[well_col].unique()
    comparison = {}

    for w in wells:
        well_data = prod_df[prod_df[well_col] == w]
        numeric = well_data.select_dtypes(include="number")
        if not numeric.empty:
            comparison[str(w)] = {
                "records": len(well_data),
                "metrics": {
                    col: round(float(numeric[col].mean()), 2)
                    for col in numeric.columns[:4]
                },
            }

    return {"comparison": comparison, "total_wells": len(wells)}


@register_tool(
    name="get_depth_log",
    description=(
        "Get depth-indexed LWD/MWD log data. Returns formation evaluation "
        "measurements at each depth: gamma ray, resistivity, density, neutron, "
        "and other downhole measurements. Use for formation analysis queries."
    ),
    parameters={"depth_min": "Optional minimum depth (m)", "depth_max": "Optional maximum depth (m)"},
)
def get_depth_log(state: dict, depth_min: float = None, depth_max: float = None, **kwargs) -> dict:
    """Return depth-indexed log data."""
    depth_df = state.get("depth_df")

    if depth_df is None or depth_df.empty:
        return {"error": "Depth log not loaded"}

    import numpy as np

    df = depth_df
    if depth_min is not None:
        df = df[df.index >= depth_min]
    if depth_max is not None:
        df = df[df.index <= depth_max]

    result = {
        "total_rows": len(df),
        "depth_range": {"min": round(float(df.index.min()), 2), "max": round(float(df.index.max()), 2)},
        "columns": list(df.columns[:20]),  # Limit to first 20 for context size
    }

    # Summary stats for key columns
    for col in df.columns[:10]:
        series = df[col].dropna()
        if len(series) > 0:
            result[col] = {
                "mean": round(float(series.mean()), 3),
                "min": round(float(series.min()), 3),
                "max": round(float(series.max()), 3),
            }

    return result


@register_tool(
    name="get_rop_formation",
    description=(
        "Get ROP and formation properties data: rate of penetration correlated "
        "with porosity, permeability (KLOGH), shale volume (VSH), and water "
        "saturation (SW). Use for drilling optimization and formation analysis."
    ),
)
def get_rop_formation(state: dict, **kwargs) -> dict:
    """Return ROP and petrophysics data."""
    rop_df = state.get("rop_df")

    if rop_df is None or rop_df.empty:
        return {"error": "ROP data not loaded"}

    import numpy as np

    result = {
        "total_rows": len(rop_df),
        "depth_range": {"min": round(float(rop_df.index.min()), 2), "max": round(float(rop_df.index.max()), 2)},
        "columns": list(rop_df.columns),
    }

    # Summary of each column
    for col in rop_df.columns:
        series = rop_df[col].dropna()
        if len(series) > 0:
            result[col] = {
                "mean": round(float(series.mean()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "std": round(float(series.std()), 4),
            }

    return result


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, state: dict, **params) -> dict:
    """
    Execute a registered tool by name.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute.
    state : dict
        Application state containing DataFrames and stores.
    **params
        Additional parameters for the tool.

    Returns
    -------
    dict
        Tool execution result.
    """
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        result = tool["function"](state, **params)
        return result
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        return {"error": f"Tool execution failed: {str(e)}"}


def get_tool_descriptions() -> str:
    """
    Get formatted tool descriptions for LLM system prompt.

    Returns
    -------
    str
        Formatted tool list for LLM consumption.
    """
    lines = []
    for name, tool in TOOL_REGISTRY.items():
        params = ""
        if tool["parameters"]:
            param_list = [f"{k}: {v}" for k, v in tool["parameters"].items()]
            params = f" Parameters: {', '.join(param_list)}"
        lines.append(f"- {name}: {tool['description']}{params}")
    return "\n".join(lines)


def _is_nan(val) -> bool:
    """Check if value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False
