"""
Data Quality Engine Tests
==========================
"""

import pandas as pd
import numpy as np

from drillmind.data.quality import (
    DataQualityReport,
    detect_flatlines,
    detect_spikes,
    detect_time_gaps,
    run_quality_check,
)


class TestTimeGaps:
    def test_no_gaps_in_continuous_data(self):
        """Evenly spaced data should have no gaps."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({"a": range(100)}, index=idx)
        gaps = detect_time_gaps(df, threshold_seconds=30)
        assert len(gaps) == 0

    def test_detects_gap(self):
        """A 2-minute gap should be detected with 30s threshold."""
        idx = pd.DatetimeIndex([
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:05",
            "2024-01-01 00:02:05",  # 2 min gap
            "2024-01-01 00:02:10",
        ], tz="UTC")
        df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
        gaps = detect_time_gaps(df, threshold_seconds=30)
        assert len(gaps) == 1
        assert gaps[0].duration_seconds == 120.0


class TestSpikes:
    def test_detects_spike(self):
        """An extreme outlier should be flagged."""
        np.random.seed(42)
        values = np.random.normal(100, 5, 1000)
        values[500] = 999  # spike
        idx = pd.date_range("2024-01-01", periods=1000, freq="5s", tz="UTC")
        df = pd.DataFrame({"sensor": values}, index=idx)
        spikes = detect_spikes(df, zscore_threshold=4.0)
        assert len(spikes) >= 1
        assert any(s.column == "sensor" for s in spikes)

    def test_no_spikes_in_uniform_data(self):
        """Uniform data should produce no spikes."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({"a": np.ones(100)}, index=idx)
        spikes = detect_spikes(df, zscore_threshold=4.0)
        assert len(spikes) == 0


class TestFlatlines:
    def test_detects_flatline(self):
        """A sequence of 30 identical values should be flagged with window=20."""
        values = list(range(50)) + [99.0] * 30 + list(range(50))
        idx = pd.date_range("2024-01-01", periods=len(values), freq="5s", tz="UTC")
        df = pd.DataFrame({"sensor": values}, index=idx)
        flatlines = detect_flatlines(df, window=20)
        assert len(flatlines) >= 1
        assert flatlines[0].value == 99.0
        assert flatlines[0].count >= 30

    def test_no_flatline_in_varying_data(self):
        """Unique values should produce no flatlines."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({"a": np.arange(100, dtype=float)}, index=idx)
        flatlines = detect_flatlines(df, window=20)
        assert len(flatlines) == 0


class TestFullReport:
    def test_report_on_real_data(self, time_df):
        """Quality check on real data should produce a valid report."""
        report = run_quality_check(time_df)
        assert isinstance(report, DataQualityReport)
        assert report.total_rows == len(time_df)
        assert report.total_columns == len(time_df.columns)
        assert report.time_range_start is not None
        assert report.time_range_end is not None

    def test_summary_string(self, time_df):
        """summary() should return a non-empty string."""
        report = run_quality_check(time_df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Rows:" in summary
