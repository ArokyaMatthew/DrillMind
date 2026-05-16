"""Verify the anomaly event classifier on real Volve data."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from drillmind.parsers.time_log_parser import load_time_log
from drillmind.models.feature_engineering import build_feature_matrix
from drillmind.models.anomaly_detection import (
    AutoencoderConfig,
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestConfig,
    IsolationForestDetector,
)
from drillmind.models.event_classifier import classify_anomalies


def main() -> None:
    # Load data and build features
    df = load_time_log(nrows=50000)
    features = build_feature_matrix(df)

    # Train models
    ae = AutoencoderDetector(AutoencoderConfig(epochs=20, batch_size=512))
    ae.fit(features)

    ifo = IsolationForestDetector(IsolationForestConfig(n_estimators=100))
    ifo.fit(features)

    ensemble = EnsembleDetector(ae, ifo)
    ensemble.calibrate(features)

    # Score and classify
    details = ensemble.score_with_details(features)

    events = classify_anomalies(
        features=features,
        anomaly_scores=details["combined"],
        anomaly_mask=details["is_anomaly"],
    )

    # Print classified events
    print(f"\n{'='*80}")
    print(f"CLASSIFIED ANOMALY EVENTS: {len(events)}")
    print(f"{'='*80}")

    for i, event in enumerate(events[:20]):
        print(f"\n--- Event {i+1} ---")
        print(f"  Time:        {event.timestamp}")
        print(f"  Type:        {event.event_type.value}")
        print(f"  Severity:    {event.severity.value}")
        print(f"  Score:       {event.score:.4f}")
        print(f"  Duration:    {event.duration_rows} samples")
        print(f"  Description: {event.description}")
        print(f"  Action:      {event.recommended_action}")

    print(f"\n{'='*80}")
    print("CLASSIFICATION COMPLETE")


if __name__ == "__main__":
    main()
