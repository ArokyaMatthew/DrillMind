"""End-to-end verification of the ML pipeline on real Volve data."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from drillmind.parsers.time_log_parser import load_time_log
from drillmind.models.feature_engineering import build_feature_matrix
from drillmind.models.anomaly_detection import (
    AutoencoderConfig,
    AutoencoderDetector,
    EnsembleConfig,
    EnsembleDetector,
    IsolationForestConfig,
    IsolationForestDetector,
)


def main() -> None:
    # 1. Load real data
    print("=== LOADING DATA ===")
    df = load_time_log(nrows=50000)

    # 2. Build feature matrix
    print("\n=== BUILDING FEATURES ===")
    features = build_feature_matrix(df)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature columns (first 10): {list(features.columns[:10])}")
    print(f"NaN count: {features.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(features.values).sum()}")

    # 3. Train autoencoder (fewer epochs for verification)
    print("\n=== TRAINING AUTOENCODER ===")
    ae = AutoencoderDetector(AutoencoderConfig(epochs=20, batch_size=512))
    ae_metrics = ae.fit(features)
    print(f"Final train loss: {ae_metrics['train_losses'][-1]:.6f}")
    print(f"Final val loss: {ae_metrics['val_losses'][-1]:.6f}")
    print(f"Threshold: {ae_metrics['threshold']:.6f}")

    # 4. Train isolation forest
    print("\n=== TRAINING ISOLATION FOREST ===")
    ifo = IsolationForestDetector(IsolationForestConfig(n_estimators=100))
    ifo_metrics = ifo.fit(features)
    print(f"Anomalies flagged: {ifo_metrics['n_anomalies_train']}")

    # 5. Ensemble
    print("\n=== ENSEMBLE SCORING ===")
    ensemble = EnsembleDetector(ae, ifo)
    cal = ensemble.calibrate(features)
    print(f"Ensemble threshold: {cal['threshold']:.4f}")
    print(f"Score P95: {cal['score_p95']:.4f}")
    print(f"Score P99: {cal['score_p99']:.4f}")

    # 6. Score the data
    details = ensemble.score_with_details(features)
    n_anomalies = int(details["is_anomaly"].sum())
    total = len(features)
    print(f"\nTotal anomalies detected: {n_anomalies} / {total} ({100*n_anomalies/total:.1f}%)")

    # 7. Show when anomalies occur
    anomaly_idx = np.where(details["is_anomaly"] == 1)[0]
    if len(anomaly_idx) > 0:
        print("\nFirst 10 anomaly timestamps:")
        for i in anomaly_idx[:10]:
            ts = features.index[i]
            score = details["combined"][i]
            ae_s = details["autoencoder_norm"][i]
            ifo_s = details["isolation_forest_norm"][i]
            print(f"  {ts}  score={score:.4f}  ae={ae_s:.4f}  ifo={ifo_s:.4f}")

    # 8. Save model
    print("\n=== SAVING MODEL ===")
    ae.save("data/processed/models/autoencoder")
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
