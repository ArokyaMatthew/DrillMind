# DrillMind — AI Copilot for Real-Time Drilling Operations

<p align="center">
  <strong>Multimodal anomaly detection and intelligent alerting for RTOC analysts</strong><br>
  Built on the Equinor Volve open dataset · PyTorch + scikit-learn ensemble · FastAPI + WebSocket streaming
</p>

---

## What This Is

DrillMind is a production-grade AI system for **Real-Time Operations Centers (RTOC)** in drilling. It ingests time-indexed drilling telemetry from the [Equinor Volve dataset](https://www.equinor.com/energy/volve-data-sharing) (419,747 rows × 239 channels, 4-5 second intervals over 19 days of actual drilling on well 15/9-F-9 A), runs ensemble anomaly detection, classifies detected events into operational categories, and serves everything through a real-time dashboard.

**No synthetic data. No mock services. No hallucinated column names.** Every column name in this codebase was verified against the actual downloaded CSV files.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RTOC Dashboard                               │
│  KPIs │ SPP Chart │ Hookload │ Torque │ Pit Vol │ Anomaly Timeline  │
│  Event List │ Severity Filters │ Detection Summary                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │ REST + WebSocket
┌──────────────────────┴───────────────────────────────────────────────┐
│                       FastAPI Backend                                 │
│  /api/well/info │ /api/data/timeseries │ /api/anomalies/*            │
│  /api/quality/report │ /api/data/production │ /ws/stream              │
└───────┬──────────────┬──────────────────┬────────────────────────────┘
        │              │                  │
  ┌─────┴─────┐  ┌─────┴──────┐   ┌──────┴──────┐
  │  Parsers  │  │  ML Engine │   │  Streaming  │
  │           │  │            │   │             │
  │ Time Log  │  │ Features   │   │ WebSocket   │
  │ Depth Log │  │ Autoencoder│   │ Replay      │
  │ Production│  │ Iso Forest │   │ Engine      │
  │           │  │ Ensemble   │   │             │
  │           │  │ Classifier │   │             │
  └─────┬─────┘  └─────┬──────┘   └─────────────┘
        │              │
  ┌─────┴──────────────┴──────┐
  │    Column Registry        │
  │    (column_registry.yaml) │
  │    63 verified mappings   │
  └───────────┬───────────────┘
              │
  ┌───────────┴───────────────┐
  │   Equinor Volve Dataset   │
  │   428 MB time-series CSV  │
  │   5.6 MB depth log        │
  │   2.3 MB production XLSX  │
  └───────────────────────────┘
```

## Data

| File | Size | Rows | Columns | Description |
|------|------|------|---------|-------------|
| `Norway-NA-15_47_9-F-9 A time.csv` | 428 MB | 419,747 | 239 | Real-time drilling telemetry (4-5s intervals) |
| `Norway-NA-15_47_9-F-9 A depth.csv` | 5.6 MB | ~36K | 115 | Depth-indexed petrophysical + drilling data |
| `Volve production data.xlsx` | 2.3 MB | 15,634 | 24 | Daily production data for 7 wells (2007-2016) |

## ML Pipeline

### Feature Engineering (261 features)
- **Rolling statistics**: mean/std/min/max over 30s, 2.5min, 5min windows
- **First-order derivatives**: instantaneous, 30s, and 2.5min rate-of-change
- **Cross-channel features**: mud weight differential, SPP/flow ratio, torque/RPM ratio, WOB/hookload ratio

### Anomaly Detection Ensemble
- **Autoencoder** (PyTorch, CUDA): learns "normal" drilling baseline — high reconstruction error = novel anomaly
- **Isolation Forest** (scikit-learn, 200 trees): multivariate point anomalies in 261-dimensional feature space
- **Ensemble combiner**: weighted (60% AE + 40% IF), min-max normalized, percentile-calibrated threshold

### Event Classifier
Rule-based classification into 6 drilling problem categories with **explainable** reasoning:

| Event Type | Detection Pattern | Recommended Action |
|-----------|-------------------|-------------------|
| **Kick / Gas Influx** | Pit gain + gas up + flow change | Stop drilling, check pit levels, prepare to shut in |
| **Lost Circulation** | Pit loss + SPP drop | Reduce pump rate, prepare LCM |
| **Stuck Pipe** | Torque spike + hookload change | Work pipe gently, consider spotting fluid |
| **Bit Dysfunction** | Erratic torque at constant WOB | Check ROP trend, consider POOH |
| **Washout** | Gradual SPP decrease | Monitor, consider drill string inspection |
| **Connection Gas** | Gas spike without pit gain | Monitor — normal during connections |

## Project Structure

```
C:\Projects\rig\
├── config/
│   ├── column_registry.yaml    # 63 verified column mappings
│   └── settings.yaml           # Central configuration
├── data/
│   ├── raw/                    # Downloaded Volve datasets
│   └── processed/              # Model artifacts
├── dashboard/
│   ├── index.html              # RTOC dashboard
│   ├── styles.css              # Dark industrial theme
│   └── app.js                  # Chart.js + API integration
├── src/drillmind/
│   ├── config.py               # Typed configuration loader
│   ├── parsers/
│   │   ├── time_log_parser.py  # 428 MB time-series parser
│   │   ├── depth_log_parser.py # Depth-indexed log parser
│   │   └── production_parser.py# Excel production parser
│   ├── data/
│   │   └── quality.py          # Gap/spike/flatline detection
│   ├── models/
│   │   ├── feature_engineering.py  # 261 features
│   │   ├── anomaly_detection.py    # AE + IF + Ensemble
│   │   └── event_classifier.py     # Rule-based classification
│   ├── streaming/
│   │   └── replay_engine.py    # WebSocket replay server
│   └── api/
│       └── server.py           # FastAPI backend
├── scripts/
│   ├── verify_pipeline.py      # End-to-end ML verification
│   └── verify_classifier.py    # Event classifier verification
├── tests/
└── pyproject.toml
```

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Download data (requires Kaggle CLI)
kaggle datasets download -d imranulhaquenoor/volve-dataset-well-f-9-a -p data/raw --unzip
kaggle datasets download -d lamyalbert/volve-production-data -p data/raw --unzip

# 3. Run server (50K rows for fast startup)
set DRILLMIND_MAX_ROWS=50000
python -m uvicorn drillmind.api.server:app --host 0.0.0.0 --port 8000

# 4. Open dashboard
# http://localhost:8000/dashboard/index.html
```

## Verified Results (on real Volve data)

| Metric | Value |
|--------|-------|
| Feature matrix | 49,940 rows × 261 features |
| Autoencoder training | CUDA, 30 epochs, loss 0.167 |
| Isolation Forest | 200 trees, 2.0% contamination |
| Ensemble threshold | 0.28 (97th percentile) |
| Anomalies detected | 1,499 / 49,940 (3.0%) |
| Classified events | 65 total |
| — Connection gas | 20 events (physically real) |
| — Bit dysfunction | 2 events |
| — Unknown | 43 events (operational transitions) |

## Tech Stack

- **Python 3.11** — core runtime
- **PyTorch 2.5 + CUDA** — autoencoder training & inference
- **scikit-learn 1.7** — isolation forest, preprocessing
- **pandas / numpy** — data manipulation
- **FastAPI + uvicorn** — REST API + WebSocket server
- **Chart.js** — real-time dashboard charting
- **Equinor Volve Dataset** — industry-standard open drilling data

## License

MIT
