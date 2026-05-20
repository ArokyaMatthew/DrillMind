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
from pathlib import Path
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
from drillmind.models.rig_state import classify_rig_state, compute_state_transitions
from drillmind.models.drilling_kpis import compute_drilling_kpis
from drillmind.copilot.engine import CopilotEngine
from drillmind.parsers.production_parser import load_production_data
from drillmind.agents.orchestrator import AgentOrchestrator
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

    # ---- Model training or loading from checkpoint ----
    model_dir = Path(settings.data.processed_dir).parent / "models"
    force_retrain = os.environ.get("DRILLMIND_RETRAIN", "0") == "1"
    checkpoint_exists = (model_dir / "autoencoder_weights.pt").exists()

    if checkpoint_exists and not force_retrain:
        # ---- FAST PATH: Load saved models (~2 seconds) ----
        logger.info("Loading saved models from {} ...", model_dir)

        ae = AutoencoderDetector(AutoencoderConfig())
        ae.load(model_dir)
        _state["ae"] = ae

        ifo = IsolationForestDetector(IsolationForestConfig())
        ifo.load(model_dir)
        _state["ifo"] = ifo

        ensemble = EnsembleDetector(ae, ifo)
        ensemble.load(model_dir)
        _state["ensemble"] = ensemble

        try:
            from drillmind.models.lstm_detector import LSTMDetector, LSTMConfig
            if (model_dir / "lstm_weights.pt").exists():
                lstm = LSTMDetector(LSTMConfig())
                lstm.load(model_dir)
                _state["lstm"] = lstm
                logger.info("LSTM loaded (ensemble weights: AE {:.0%} + IF {:.0%} + LSTM {:.0%})",
                    ensemble.config.autoencoder_weight,
                    ensemble.config.isolation_forest_weight,
                    ensemble.config.lstm_weight,
                )
        except Exception as e:
            logger.warning("LSTM load failed (non-fatal): {}", e)

        logger.info("All models loaded from checkpoint")
    else:
        # ---- SLOW PATH: Train from scratch (~40 min) ----
        if force_retrain:
            logger.info("DRILLMIND_RETRAIN=1 — forcing full retrain")

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

        # Train LSTM temporal model (3rd ensemble member)
        try:
            from drillmind.models.lstm_detector import LSTMDetector, LSTMConfig

            logger.info("Training LSTM temporal model...")
            lstm = LSTMDetector(LSTMConfig(seq_len=60, epochs=20))
            lstm_result = lstm.fit(time_df)

            if lstm_result.get("status") == "trained":
                lstm_scores = lstm.score(time_df)
                offset = len(time_df) - len(features)
                lstm_aligned = lstm_scores[offset:] if offset > 0 else lstm_scores[:len(features)]
                ensemble.attach_lstm(lstm_aligned)
                ensemble.calibrate(features)
                _state["lstm"] = lstm
                logger.info("LSTM integrated into ensemble (AE 50% + IF 30% + LSTM 20%)")
            else:
                logger.info("LSTM skipped: {}", lstm_result.get("reason", "unknown"))
        except Exception as e:
            logger.warning("LSTM training failed (non-fatal, using AE+IF only): {}", e)

        # Save all models to disk
        logger.info("Saving models to {} ...", model_dir)
        ae.save(model_dir)
        ifo.save(model_dir)
        ensemble.save(model_dir)
        if "lstm" in _state:
            _state["lstm"].save(model_dir)
        logger.info("Models saved — next startup will load from checkpoint")

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

    # Classify rig state
    logger.info("Classifying rig states...")
    rig_states = classify_rig_state(time_df)
    _state["rig_states"] = rig_states
    transitions = compute_state_transitions(rig_states)
    _state["transitions"] = transitions

    # Compute drilling KPIs
    logger.info("Computing drilling KPIs...")
    kpi_df = compute_drilling_kpis(time_df)
    _state["kpi_df"] = kpi_df

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

    # Load depth-indexed log (LWD/MWD data, 86 rows × 115 columns)
    try:
        from drillmind.parsers.depth_log_parser import load_depth_log
        depth_df = load_depth_log()
        _state["depth_df"] = depth_df
    except Exception as e:
        logger.warning("Could not load depth log (non-fatal): {}", e)
        _state["depth_df"] = None

    # Load ROP + petrophysics data (152 rows × 8 columns)
    try:
        from drillmind.parsers.rop_parser import load_rop_data
        rop_df = load_rop_data()
        _state["rop_df"] = rop_df
    except Exception as e:
        logger.warning("Could not load ROP data (non-fatal): {}", e)
        _state["rop_df"] = None

    # --- DDR + RAG Pipeline ---
    try:
        from drillmind.parsers.ddr_parser import load_ddrs_from_huggingface
        from drillmind.rag.chunker import chunk_all_ddrs
        from drillmind.rag.store import DDRVectorStore

        logger.info("Loading DDRs from HuggingFace...")
        ddrs = load_ddrs_from_huggingface()
        chunks = chunk_all_ddrs(ddrs)

        rag_store = DDRVectorStore(persist_dir="data/chromadb")
        if rag_store.count == 0:
            logger.info("Indexing DDR chunks into ChromaDB...")
            rag_store.index_chunks(chunks)
        else:
            logger.info(f"ChromaDB already has {rag_store.count} docs, skipping re-index")
        _state["rag_store"] = rag_store
        _state["ddrs"] = ddrs
    except Exception as e:
        logger.warning(f"DDR/RAG initialization failed (non-fatal): {e}")
        _state["rag_store"] = None
        _state["ddrs"] = []

    # Initialize copilot engine
    copilot_provider = os.environ.get("DRILLMIND_LLM_PROVIDER", "fallback")
    copilot_model = os.environ.get("DRILLMIND_LLM_MODEL", None)
    _state["copilot"] = CopilotEngine(provider=copilot_provider, model=copilot_model)

    logger.info("=== DrillMind API ready ===")
    yield
    logger.info("=== DrillMind API shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DrillMind API",
    description="Real-time drilling analytics and monitoring API",
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
# Depth Log (LWD/MWD)
# ---------------------------------------------------------------------------
@app.get("/api/data/depth")
async def get_depth_data(
    limit: int = Query(500, ge=1, le=5000),
):
    """Return depth-indexed LWD/MWD log data."""
    depth_df = _state.get("depth_df")
    if depth_df is None:
        return JSONResponse(status_code=404, content={"error": "Depth log not loaded"})

    return {
        "total": len(depth_df),
        "columns": list(depth_df.columns),
        "depth_range": {
            "min": round(float(depth_df.index.min()), 2),
            "max": round(float(depth_df.index.max()), 2),
        },
        "data": _df_to_records(depth_df.head(limit)),
    }


# ---------------------------------------------------------------------------
# ROP + Petrophysics
# ---------------------------------------------------------------------------
@app.get("/api/data/rop")
async def get_rop_data(
    limit: int = Query(500, ge=1, le=5000),
):
    """Return ROP and petrophysical formation properties (porosity, perm, shale volume)."""
    rop_df = _state.get("rop_df")
    if rop_df is None:
        return JSONResponse(status_code=404, content={"error": "ROP data not loaded"})

    return {
        "total": len(rop_df),
        "columns": list(rop_df.columns),
        "depth_range": {
            "min": round(float(rop_df.index.min()), 2),
            "max": round(float(rop_df.index.max()), 2),
        },
        "data": _df_to_records(rop_df.head(limit)),
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

            # Add anomaly score if available (use offset-based lookup)
            offset = len(time_df) - len(features)
            feat_idx = i - offset
            if 0 <= feat_idx < len(details["combined"]):
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
# Rig State
# ---------------------------------------------------------------------------
@app.get("/api/rig/state")
async def get_rig_state(
    start: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Return rig state classification for each time sample."""
    states = _state["rig_states"]
    end = min(start + limit, len(states))
    time_df = _state["time_df"]

    data = []
    for i in range(start, end):
        val = states.iloc[i]
        data.append({
            "timestamp": str(time_df.index[i]),
            "state": val.value if hasattr(val, "value") else str(val),
        })

    return {"start": start, "end": end, "total": len(states), "data": data}


@app.get("/api/rig/summary")
async def get_rig_summary():
    """Return rig state time breakdown."""
    from collections import Counter
    states = _state["rig_states"]
    counts = Counter(s.value if hasattr(s, "value") else str(s) for s in states)
    total = len(states)
    return {
        "total_samples": total,
        "states": {
            state: {"count": count, "pct": round(100 * count / total, 2)}
            for state, count in sorted(counts.items(), key=lambda x: -x[1])
        },
    }


@app.get("/api/rig/transitions")
async def get_rig_transitions(
    limit: int = Query(100, ge=1, le=1000),
):
    """Return rig state transition log."""
    trans = _state["transitions"]
    records = []
    for _, row in trans.tail(limit).iterrows():
        records.append({
            "start": str(row["start"]),
            "end": str(row["end"]),
            "state": row["state"],
            "duration_samples": int(row["duration_samples"]),
            "duration_seconds": float(row["duration_seconds"]),
        })
    return {"total": len(trans), "transitions": records}


# ---------------------------------------------------------------------------
# Drilling KPIs
# ---------------------------------------------------------------------------
@app.get("/api/kpi/values")
async def get_kpi_values(
    start: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
):
    """Return drilling KPI values."""
    kpi_df = _state["kpi_df"]
    end = min(start + limit, len(kpi_df))
    subset = kpi_df.iloc[start:end]

    data = []
    for idx, row in subset.iterrows():
        record = {"timestamp": str(idx)}
        for col in kpi_df.columns:
            val = row[col]
            record[col] = round(float(val), 4) if pd.notna(val) and np.isfinite(val) else None
        data.append(record)

    return {"start": start, "end": end, "total": len(kpi_df), "data": data}


@app.get("/api/kpi/summary")
async def get_kpi_summary():
    """Return KPI summary statistics."""
    kpi_df = _state["kpi_df"]
    summary = {}
    for col in kpi_df.columns:
        series = kpi_df[col].dropna()
        if len(series) == 0:
            summary[col] = {"available": False}
        else:
            finite = series[np.isfinite(series)]
            summary[col] = {
                "available": True,
                "count": len(finite),
                "mean": round(float(finite.mean()), 4),
                "std": round(float(finite.std()), 4),
                "min": round(float(finite.min()), 4),
                "max": round(float(finite.max()), 4),
                "median": round(float(finite.median()), 4),
            }
    return summary


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------
from pydantic import BaseModel


class CopilotQuery(BaseModel):
    question: str


@app.post("/api/copilot/query")
async def copilot_query(query: CopilotQuery):
    """Process a natural language question using tool-based query orchestration."""
    # Wire the stored CopilotEngine's LLM into the orchestrator if configured
    copilot: CopilotEngine = _state.get("copilot")
    llm_fn = None
    if copilot and copilot._llm.name != "fallback":
        llm_fn = copilot._llm.generate

    agent = AgentOrchestrator(
        state={
            "time_df": _state["time_df"],
            "events": _state["events"],
            "anomaly_details": _state["anomaly_details"],
            "features": _state["features"],
            "rig_states": _state["rig_states"],
            "transitions": _state["transitions"],
            "kpi_df": _state["kpi_df"],
            "production_df": _state.get("production_df"),
            "quality_report": _state.get("quality_report"),
            "rag_store": _state.get("rag_store"),
            "depth_df": _state.get("depth_df"),
            "rop_df": _state.get("rop_df"),
        },
        llm_fn=llm_fn,
    )
    result = await agent.query(query.question)
    provider_name = copilot._llm.name if copilot else "fallback"
    model_name = copilot._llm.model_name if copilot and llm_fn else "rule-based-v2"
    return {
        "answer": result.answer,
        "provider": provider_name,
        "model": model_name,
        "grounded": result.grounded,
        "context_summary": {
            "intent": result.intent,
            "tools_called": result.tools_called,
            "evidence_count": len(result.evidence),
            "total_time_ms": round(result.total_time * 1000),
        },
    }


class RAGSearchQuery(BaseModel):
    query: str
    top_k: int = 5
    well_filter: str | None = None


@app.post("/api/rag/search")
async def rag_search(search: RAGSearchQuery):
    """Search Daily Drilling Reports via semantic similarity."""
    rag_store = _state.get("rag_store")
    if rag_store is None:
        return {"error": "RAG store not initialized", "results": []}

    results = rag_store.search(
        query=search.query,
        top_k=search.top_k,
        well_filter=search.well_filter,
    )
    return {
        "query": search.query,
        "results": [r.to_dict() for r in results],
        "total_indexed": rag_store.count,
    }


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.3.0",
        "features": {
            "anomaly_detection": True,
            "rig_state": True,
            "drilling_kpis": True,
            "rag_ddr": _state.get("rag_store") is not None,
            "agent_orchestrator": True,
        },
    }


@app.get("/")
async def root():
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard/index.html")
