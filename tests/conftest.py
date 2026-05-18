"""
Shared pytest fixtures for DrillMind test suite.

All fixtures use real Volve data (first 1000 rows for speed).
No synthetic data, no mocks for data content.
"""

import pytest
import pandas as pd

from drillmind.config import get_column_registry, get_settings


@pytest.fixture(scope="session")
def settings():
    """Application settings singleton."""
    return get_settings()


@pytest.fixture(scope="session")
def registry():
    """Column registry singleton."""
    return get_column_registry()


@pytest.fixture(scope="session")
def time_df():
    """First 1000 rows of the time log — enough for unit tests."""
    from drillmind.parsers.time_log_parser import load_time_log
    return load_time_log(nrows=1000)


@pytest.fixture(scope="session")
def time_df_large():
    """First 5000 rows — for feature engineering tests that need rolling windows."""
    from drillmind.parsers.time_log_parser import load_time_log
    return load_time_log(nrows=5000)


@pytest.fixture(scope="session")
def depth_df():
    """First 1000 rows of the depth log."""
    from drillmind.parsers.depth_log_parser import load_depth_log
    return load_depth_log(nrows=1000)


@pytest.fixture(scope="session")
def production_df():
    """Production data (full — only 15K rows)."""
    from drillmind.parsers.production_parser import load_production_data
    return load_production_data()


@pytest.fixture(scope="session")
def rop_df():
    """ROP + petrophysics data (full — only 151 rows)."""
    from drillmind.parsers.rop_parser import load_rop_data
    return load_rop_data()
