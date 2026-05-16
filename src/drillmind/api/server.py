"""
DrillMind REST API
===================
FastAPI backend that serves:

1. Drilling data endpoints (well info, time series, depth log)
2. Anomaly detection results (events, scores, timeline)
3. Data quality reports
4. WebSocket endpoint for real-time streaming

This is the integration layer — connects parsers, models, and the
streaming engine into a unified API surface.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from drillmind.config import get_settings
from drillmind.data.quality import run_quality_check
from drillmind.models.anomaly_detection import (
    AutoencoderConfig,
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestConfig,
    IsolationForestDetector,
)
from drillmind.models.event_classifier import classify_anomalies
from drillmind.models.feature_engineering import build_feature_matrix
from drillmind.parsers.production_parser import load_production_data
from drillmind.parsers.time_log_parser import load_time_log

# ---------------------------------------------------------------------------
# Application State (loaded once at startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


def _serialize_value(val: Any) -> Any:
    """Safely serialize numpy/pandas types to JSON-compatible Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if np.isnan(val) else round(float(val), 6)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, pd.Timestamp):
        return str(val)
    if pd.isna(val):
        return None
    return val


def _df_to_records(df: pd.DataFrame, max_rows: int = 10000) -> list[dict]:
    """Convert a DataFrame to a list of dicts with safe serialization."""
    records = []
    for idx, row in df.head(max_rows).iterrows():
        record = {"timestamp": str(idx)} if isinstance(idx, pd.Timestamp) else {"index": idx}
        for col in df.columns:
            record[col] = _serialize_value(row[col])
        records.append(record)
    return records


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and train models at startup."""
    logger.info("=== DrillMind API starting ===")
    settings = get_settings()

    # Load time log (limit rows via DRILLMIND_MAX_ROWS env var for dev)
    import os
    max_rows = int(os.environ.get("DRILLMIND_MAX_ROWS", "0")) or None
    logger.info("Loading time log... (max_rows={})", max_rows or "ALL")
    time_df = load_time_log(nrows=max_rows)
    _state["time_df"] = time_df
    logger.info("Time log: {} rows", len(time_df))

    # Build features
    logger.info("Building feature matrix...")
    features = build_feature_matrix(time_df)
    _state["features"] = features
    logger.info("Features: {} rows x {} cols", *features.shape)

    # Train anomaly detection
    logger.info("Training autoencoder...")
    ae = AutoencoderDetector(AutoencoderConfig(epochs=30, batch_size=512))
    ae.fit(features)
    _state["ae"] = ae

    logger.info("Training isolation forest...")
    ifo = IsolationForestDetector(IsolationForestConfig(n_estimators=200))
    ifo.fit(features)
    _state["ifo"] = ifo

    logger.info("Calibrating ensemble...")
    ensemble = EnsembleDetector(ae, ifo)
    ensemble.calibrate(features)
    _state["ensemble"] = ensemble

    # Score and classify
    logger.info("Scoring and classifying anomalies...")
    details = ensemble.score_with_details(features)
    _state["anomaly_details"] = details

    events = classify_anomalies(
        features=features,
        anomaly_scores=details["combined"],
        anomaly_mask=details["is_anomaly"],
    )
    _state["events"] = events

    # Run quality check
    logger.info("Running data quality check...")
    quality_report = run_quality_check(time_df)
    _state["quality_report"] = quality_report

    # Load production data
    try:
        prod_df = load_production_data()
        _state["production_df"] = prod_df
    except Exception as e:
        logger.warning("Could not load production data: {}", e)

    logger.info("=== DrillMind API ready ===")
    yield
    logger.info("=== DrillMind API shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DrillMind API",
    description="Real-Time Drilling Operations AI Copilot",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard static files
_dashboard_dir = get_settings().data.raw_dir.replace("data/raw", "dashboard")
# Resolve relative to project root
from drillmind.config import get_project_root
_dashboard_path = get_project_root() / "dashboard"
if _dashboard_path.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")


# ---------------------------------------------------------------------------
# Well Info
# ---------------------------------------------------------------------------
@app.get("/api/well/info")
async def well_info():
    """Return well metadata."""
    settings = get_settings()
    time_df: pd.DataFrame = _state["time_df"]
    return {
        "well": settings.well,
        "field": settings.field_name,
        "operator": settings.operator,
        "total_rows": len(time_df),
        "time_start": str(time_df.index.min()),
        "time_end": str(time_df.index.max()),
        "columns": list(time_df.columns),
        "n_features": _state["features"].shape[1],
        "n_events": len(_state["events"]),
    }


# ---------------------------------------------------------------------------
# Time Series Data
# ---------------------------------------------------------------------------
@app.get("/api/data/timeseries")
async def get_timeseries(
    start: int = Query(0, ge=0, description="Start row index"),
    limit: int = Query(1000, ge=1, le=10000, description="Number of rows"),
    columns: str = Query(None, description="Comma-separated column names"),
):
    """Return time-indexed drilling data."""
    time_df: pd.DataFrame = _state["time_df"]
    end = min(start + limit, len(time_df))
    subset = time_df.iloc[start:end]

    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip() in time_df.columns]
        if cols:
            subset = subset[cols]

    return {
        "start": start,
        "end": end,
        "total": len(time_df),
        "columns": list(subset.columns),
        "data": _df_to_records(subset),
    }


# ---------------------------------------------------------------------------
# Anomaly Events
# ---------------------------------------------------------------------------
@app.get("/api/anomalies/events")
async def get_anomaly_events(
    severity: str = Query(None, description="Filter by severity: low, medium, high, critical"),
    event_type: str = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Return classified anomaly events."""
    events = _state["events"]

    if severity:
        events = [e for e in events if e.severity.value == severity]
    if event_type:
        events = [e for e in events if e.event_type.value == event_type]

    return {
        "total": len(events),
        "events": [e.to_dict() for e in events[:limit]],
    }


@app.get("/api/anomalies/scores")
async def get_anomaly_scores(
    start: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Return anomaly scores for the time series."""
    details = _state["anomaly_details"]
    features: pd.DataFrame = _state["features"]
    end = min(start + limit, len(features))

    scores = []
    for i in range(start, end):
        scores.append({
            "timestamp": str(features.index[i]),
            "combined": round(float(details["combined"][i]), 4),
            "autoencoder": round(float(details["autoencoder_norm"][i]), 4),
            "isolation_forest": round(float(details["isolation_forest_norm"][i]), 4),
            "is_anomaly": int(details["is_anomaly"][i]),
        })

    return {
        "start": start,
        "end": end,
        "total": len(features),
        "scores": scores,
    }


@app.get("/api/anomalies/summary")
async def get_anomaly_summary():
    """Return a summary of all detected anomalies."""
    events = _state["events"]
    details = _state["anomaly_details"]

    from collections import Counter

    type_counts = Counter(e.event_type.value for e in events)
    severity_counts = Counter(e.severity.value for e in events)

    return {
        "total_events": len(events),
        "total_anomalous_samples": int(details["is_anomaly"].sum()),
        "total_samples": len(details["is_anomaly"]),
        "anomaly_rate": round(float(details["is_anomaly"].mean()), 4),
        "by_type": dict(type_counts),
        "by_severity": dict(severity_counts),
    }


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------
@app.get("/api/quality/report")
async def get_quality_report():
    """Return data quality report."""
    report = _state["quality_report"]
    return {
        "total_rows": report.total_rows,
        "total_columns": report.total_columns,
        "time_range_start": str(report.time_range_start),
        "time_range_end": str(report.time_range_end),
        "n_gaps": len(report.gaps),
        "n_spikes": len(report.spikes),
        "n_flatlines": len(report.flatlines),
        "n_sparse_columns": len(report.sparse_columns),
        "sparse_columns": report.sparse_columns,
        "gaps": [
            {
                "start": str(g.start),
                "end": str(g.end),
                "duration_seconds": g.duration_seconds,
            }
            for g in report.gaps[:50]
        ],
    }


# ---------------------------------------------------------------------------
# Production Data
# ---------------------------------------------------------------------------
@app.get("/api/data/production")
async def get_production_data(
    well: str = Query(None, description="Filter by wellbore code"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Return production data."""
    if "production_df" not in _state:
        return JSONResponse(status_code=404, content={"error": "Production data not loaded"})

    prod_df: pd.DataFrame = _state["production_df"]

    if well:
        prod_df = prod_df[prod_df["wellbore_code"].str.contains(well, case=False, na=False)]

    return {
        "total": len(prod_df),
        "wells": list(prod_df["wellbore_code"].unique()) if "wellbore_code" in prod_df.columns else [],
        "data": _df_to_records(prod_df.head(limit)),
    }


# ---------------------------------------------------------------------------
# WebSocket for real-time streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """Stream drilling data in real-time via WebSocket."""
    await ws.accept()
    time_df: pd.DataFrame = _state["time_df"]
    details = _state["anomaly_details"]
    features: pd.DataFrame = _state["features"]
    settings = get_settings()

    speed = settings.replay.speed_multiplier
    logger.info("WebSocket client connected, streaming at {}x", speed)

    try:
        import json

        # Send metadata first
        await ws.send_json({
            "type": "meta",
            "well": settings.well,
            "total_rows": len(time_df),
            "speed": speed,
        })

        for i in range(len(time_df)):
            row = time_df.iloc[i]
            data = {
                "type": "data",
                "index": i,
                "timestamp": str(time_df.index[i]),
            }

            # Add sensor values
            for col in time_df.columns:
                data[col] = _serialize_value(row[col])

            # Add anomaly score if available
            if i < len(features):
                feat_idx = features.index.get_loc(time_df.index[i]) if time_df.index[i] in features.index else None
                if feat_idx is not None and isinstance(feat_idx, int):
                    data["anomaly_score"] = round(float(details["combined"][feat_idx]), 4)
                    data["is_anomaly"] = int(details["is_anomaly"][feat_idx])

            await ws.send_json(data)

            # Compute sleep from time delta
            if i + 1 < len(time_df):
                delta = (time_df.index[i + 1] - time_df.index[i]).total_seconds()
                delta = max(0, min(delta, 60))
                sleep = delta / speed
                if sleep > 0:
                    await asyncio.sleep(sleep)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: {}", e)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/")
async def root():
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard/index.html")
