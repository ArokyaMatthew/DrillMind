"""
Feature Engineering Tests
==========================
"""

import numpy as np
import pandas as pd

from drillmind.models.feature_engineering import build_feature_matrix


class TestFeatureMatrix:

    def test_returns_dataframe(self, time_df_large):
        features = build_feature_matrix(time_df_large)
        assert isinstance(features, pd.DataFrame)

    def test_feature_count(self, time_df_large):
        """Feature matrix should have 261 columns (verified in Phase 2)."""
        features = build_feature_matrix(time_df_large)
        assert features.shape[1] == 261

    def test_no_nans(self, time_df_large):
        """Feature matrix must have zero NaN values."""
        features = build_feature_matrix(time_df_large)
        nan_count = features.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in feature matrix"

    def test_no_infs(self, time_df_large):
        """Feature matrix must have zero Inf values."""
        features = build_feature_matrix(time_df_large)
        inf_count = np.isinf(features.values).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values in feature matrix"

    def test_fewer_rows_than_input(self, time_df_large):
        """Rolling windows drop the first N rows, so output < input."""
        features = build_feature_matrix(time_df_large)
        assert len(features) < len(time_df_large)

    def test_same_datetime_index(self, time_df_large):
        """Feature matrix must have DatetimeIndex from the source data."""
        features = build_feature_matrix(time_df_large)
        assert isinstance(features.index, pd.DatetimeIndex)

    def test_feature_names_not_empty(self, time_df_large):
        """Every column should have a non-empty name."""
        features = build_feature_matrix(time_df_large)
        for col in features.columns:
            assert col and isinstance(col, str), f"Invalid column name: {col}"
