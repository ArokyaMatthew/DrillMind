"""
Parser Tests
=============
Verify all 4 parsers against real Volve data.
"""

import pandas as pd
import numpy as np


class TestTimeLogParser:
    """Tests for the time-indexed drilling log parser."""

    def test_returns_dataframe(self, time_df):
        assert isinstance(time_df, pd.DataFrame)

    def test_row_count(self, time_df):
        assert len(time_df) == 1000

    def test_datetime_index(self, time_df):
        assert isinstance(time_df.index, pd.DatetimeIndex)
        assert time_df.index.name == "datetime"

    def test_no_nat_in_index(self, time_df):
        assert time_df.index.isna().sum() == 0

    def test_index_is_sorted(self, time_df):
        assert time_df.index.is_monotonic_increasing

    def test_critical_columns_exist(self, time_df):
        """Verify the columns we use in KPIs and anomaly detection exist."""
        required = [
            "spp", "weight_on_hook", "torque_averaged", "rpm_avg",
            "wob_avg", "bit_depth", "flow_pumps", "pit_volume_active",
            "mud_weight_in", "mud_weight_out", "tvd", "gas_total",
        ]
        for col in required:
            assert col in time_df.columns, f"Missing critical column: {col}"

    def test_numeric_columns_are_numeric(self, time_df):
        """Sensor channels must be numeric, not object dtype."""
        numeric_expected = ["spp", "weight_on_hook", "torque_averaged", "rpm_avg"]
        for col in numeric_expected:
            assert pd.api.types.is_numeric_dtype(time_df[col]), (
                f"{col} is {time_df[col].dtype}, expected numeric"
            )

    def test_standardized_column_names(self, time_df, registry):
        """Column names should be the standardized registry names, not raw CSV headers."""
        # No raw CSV header should appear as a column name
        for col in time_df.columns:
            # If it's in the registry by_name, it's been renamed correctly
            if col in registry.by_raw:
                # This means the column was NOT renamed — it's still raw
                assert False, f"Column '{col}' is a raw header, should be standardized"


class TestDepthLogParser:
    """Tests for the depth-indexed log parser."""

    def test_returns_dataframe(self, depth_df):
        assert isinstance(depth_df, pd.DataFrame)

    def test_row_count(self, depth_df):
        assert len(depth_df) == 1000

    def test_depth_index(self, depth_df):
        assert depth_df.index.name == "measured_depth"

    def test_depth_is_numeric(self, depth_df):
        assert pd.api.types.is_numeric_dtype(depth_df.index)

    def test_depth_is_positive(self, depth_df):
        assert (depth_df.index >= 0).all()


class TestProductionParser:
    """Tests for the production data parser."""

    def test_returns_dataframe(self, production_df):
        assert isinstance(production_df, pd.DataFrame)

    def test_has_rows(self, production_df):
        assert len(production_df) > 10000  # Full dataset is ~15K rows

    def test_date_index(self, production_df):
        assert isinstance(production_df.index, pd.DatetimeIndex)

    def test_wellbore_column(self, production_df):
        assert "wellbore_code" in production_df.columns

    def test_multiple_wells(self, production_df):
        wells = production_df["wellbore_code"].nunique()
        assert wells >= 5, f"Expected multiple wells, got {wells}"


class TestROPParser:
    """Tests for the ROP + petrophysics parser."""

    def test_returns_dataframe(self, rop_df):
        assert isinstance(rop_df, pd.DataFrame)

    def test_depth_index(self, rop_df):
        assert rop_df.index.name == "depth_m"

    def test_expected_columns(self, rop_df):
        expected = ["wob_rop", "surface_rpm", "rop_avg", "porosity",
                     "volume_shale", "water_saturation", "perm_log"]
        for col in expected:
            assert col in rop_df.columns, f"Missing: {col}"

    def test_depth_range(self, rop_df):
        assert rop_df.index.min() >= 3000
        assert rop_df.index.max() <= 5000

    def test_porosity_range(self, rop_df):
        """Porosity must be between 0 and 1."""
        valid = rop_df["porosity"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()
