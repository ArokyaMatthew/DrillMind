"""
Data Preprocessor
==================
Reads the raw 428 MB time-series CSV, applies data quality filters,
computes features, trains models, and saves everything to Parquet +
model artifacts for fast startup in production.

This script is run ONCE after data download. After this, the API server
can load pre-processed data in seconds instead of minutes.

Usage:
    python scripts/preprocess.py [--max-rows N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger

from drillmind.config import get_project_root, get_settings
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
from drillmind.parsers.time_log_parser import load_time_log


def main() -> None:
    parser = argparse.ArgumentParser(description="DrillMind Data Preprocessor")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows (0=all)")
    parser.add_argument("--ae-epochs", type=int, default=50, help="Autoencoder epochs")
    args = parser.parse_args()

    root = get_project_root()
    output_dir = root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # 1. Load raw data
    logger.info("Step 1/6: Loading time log...")
    nrows = args.max_rows or None
    df = load_time_log(nrows=nrows)
    logger.info("Loaded: {} rows x {} cols", len(df), len(df.columns))

    # 2. Quality check
    logger.info("Step 2/6: Running data quality check...")
    report = run_quality_check(df)

    # Save quality report
    report_path = output_dir / "quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.summary())
        f.write("\n\nSparse columns:\n")
        for col in report.sparse_columns:
            f.write(f"  - {col}\n")
    logger.info("Quality report saved to {}", report_path)

    # 3. Save cleaned time series as Parquet
    logger.info("Step 3/6: Saving cleaned time series to Parquet...")
    parquet_path = output_dir / "time_log_cleaned.parquet"
    df.to_parquet(parquet_path, engine="pyarrow" if "pyarrow" in sys.modules else "fastparquet")
    logger.info(
        "Parquet saved: {} ({:.1f} MB)",
        parquet_path,
        parquet_path.stat().st_size / 1e6,
    )

    # 4. Build features
    logger.info("Step 4/6: Building feature matrix...")
    features = build_feature_matrix(df)
    features_path = output_dir / "features.parquet"
    features.to_parquet(features_path)
    logger.info("Features saved: {} rows x {} cols", *features.shape)

    # 5. Train models
    logger.info("Step 5/6: Training anomaly detection models...")

    ae = AutoencoderDetector(AutoencoderConfig(
        epochs=args.ae_epochs,
        batch_size=512,
    ))
    ae_metrics = ae.fit(features)
    ae.save(str(model_dir / "autoencoder"))

    ifo = IsolationForestDetector(IsolationForestConfig(n_estimators=200))
    ifo_metrics = ifo.fit(features)

    ensemble = EnsembleDetector(ae, ifo)
    cal_metrics = ensemble.calibrate(features)

    # 6. Score and classify
    logger.info("Step 6/6: Scoring and classifying...")
    details = ensemble.score_with_details(features)
    events = classify_anomalies(
        features=features,
        anomaly_scores=details["combined"],
        anomaly_mask=details["is_anomaly"],
    )

    # Save events
    import json
    events_path = output_dir / "anomaly_events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)
    logger.info("Events saved: {} events to {}", len(events), events_path)

    # Summary
    elapsed = time.time() - t0
    logger.info(
        "\n{'='*60}\n"
        "PREPROCESSING COMPLETE in {:.1f}s\n"
        "  Rows processed:     {:,}\n"
        "  Features:           {}\n"
        "  AE final loss:      {:.6f}\n"
        "  Ensemble threshold: {:.4f}\n"
        "  Anomalies:          {:,} ({:.1f}%)\n"
        "  Events classified:  {}\n"
        "  Output directory:   {}\n"
        "{'='*60}",
        elapsed,
        len(df),
        features.shape[1],
        ae_metrics["train_losses"][-1],
        cal_metrics["threshold"],
        int(details["is_anomaly"].sum()),
        100 * details["is_anomaly"].mean(),
        len(events),
        output_dir,
    )


if __name__ == "__main__":
    main()
