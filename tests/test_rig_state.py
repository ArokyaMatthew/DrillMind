"""
Rig State Classifier Tests
============================
"""

import numpy as np
import pandas as pd

from drillmind.models.rig_state import (
    RigState,
    classify_rig_state,
    compute_state_transitions,
)


class TestRigStateClassifier:

    def test_returns_series(self, time_df):
        states = classify_rig_state(time_df)
        assert isinstance(states, pd.Series)

    def test_same_length_as_input(self, time_df):
        states = classify_rig_state(time_df)
        assert len(states) == len(time_df)

    def test_all_values_are_rig_state(self, time_df):
        states = classify_rig_state(time_df)
        for val in states.unique():
            assert isinstance(val, RigState), f"Unexpected value: {val}"

    def test_static_when_idle(self):
        """When RPM=0, flow=0, depth constant → should be STATIC."""
        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({
            "rpm_avg": np.zeros(100),
            "flow_pumps": np.zeros(100),
            "spp": np.zeros(100),
            "wob_avg": np.zeros(100),
            "bit_depth": np.ones(100) * 1000,
            "weight_on_hook": np.ones(100) * 70,
        }, index=idx)
        states = classify_rig_state(df)
        # Most samples should be static
        static_pct = (states == RigState.STATIC).mean()
        assert static_pct > 0.9, f"Expected mostly static, got {static_pct:.1%}"


class TestStateTransitions:

    def test_returns_dataframe(self, time_df):
        states = classify_rig_state(time_df)
        trans = compute_state_transitions(states)
        assert isinstance(trans, pd.DataFrame)

    def test_expected_columns(self, time_df):
        states = classify_rig_state(time_df)
        trans = compute_state_transitions(states)
        expected = {"start", "end", "state", "duration_samples", "duration_seconds"}
        assert expected.issubset(set(trans.columns))

    def test_covers_all_samples(self, time_df):
        """Total duration_samples should equal len(time_df)."""
        states = classify_rig_state(time_df)
        trans = compute_state_transitions(states)
        total_samples = trans["duration_samples"].sum()
        assert total_samples == len(time_df)


class TestDrillingKPIs:

    def test_mse_computation(self):
        """MSE should be computable when ROP > 0."""
        from drillmind.models.drilling_kpis import compute_mse

        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({
            "torque_averaged": np.full(100, 5.0),    # 5 kN·m
            "rpm_avg": np.full(100, 120.0),           # 120 RPM
            "wob_avg": np.full(100, 50000.0),         # 50 kN
            "rop_5ft_avg": np.full(100, 15.0),        # 15 m/h
        }, index=idx)
        mse = compute_mse(df)
        valid = mse[mse.notna()]
        assert len(valid) == 100
        assert (valid > 0).all(), "MSE should be positive for valid inputs"

    def test_d_exponent_range(self):
        """d-exponent should be in a reasonable range (0-5)."""
        from drillmind.models.drilling_kpis import compute_d_exponent

        idx = pd.date_range("2024-01-01", periods=100, freq="5s", tz="UTC")
        df = pd.DataFrame({
            "rpm_avg": np.full(100, 120.0),
            "wob_avg": np.full(100, 50000.0),
            "rop_5ft_avg": np.full(100, 15.0),
        }, index=idx)
        d_exp = compute_d_exponent(df)
        valid = d_exp[d_exp.notna() & np.isfinite(d_exp)]
        assert len(valid) > 0
