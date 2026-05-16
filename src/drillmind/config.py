"""
DrillMind Configuration Loader
==============================
Reads settings.yaml and column_registry.yaml from the config/ directory.
Provides typed, validated access to all configuration values.

Usage:
    from drillmind.config import get_settings, get_column_registry
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Resolve the project root — walk up from this file until we find pyproject.toml
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT: Path | None = None
for _parent in [_THIS_DIR, *_THIS_DIR.parents]:
    if (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break

if _PROJECT_ROOT is None:
    raise RuntimeError(
        "Cannot locate project root (no pyproject.toml found in parent directories)"
    )

CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Column Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnDef:
    """Single column definition from column_registry.yaml."""

    key: str           # internal registry key, e.g. "rop"
    raw: str           # exact header in the CSV, e.g. "Rate of Penetration m/h"
    name: str          # standardized name, e.g. "rop"
    unit: str          # physical unit, e.g. "m/h"
    category: str      # grouping, e.g. "rop"


class ColumnRegistry:
    """
    Provides lookups for column name mapping.

    Attributes
    ----------
    by_key : dict[str, ColumnDef]
        Keyed by the YAML key (e.g. "rop", "wob").
    by_raw : dict[str, ColumnDef]
        Keyed by the raw CSV header string.
    by_name : dict[str, ColumnDef]
        Keyed by the standardized internal name.
    """

    def __init__(self, registry_path: Path) -> None:
        if not registry_path.exists():
            raise FileNotFoundError(f"Column registry not found: {registry_path}")

        with open(registry_path, "r", encoding="utf-8") as fh:
            raw_data: dict[str, Any] = yaml.safe_load(fh)

        self.by_key: dict[str, ColumnDef] = {}
        self.by_raw: dict[str, ColumnDef] = {}
        self.by_name: dict[str, ColumnDef] = {}

        for key, entry in raw_data.items():
            col = ColumnDef(
                key=key,
                raw=entry["raw"],
                name=entry["name"],
                unit=entry["unit"],
                category=entry["category"],
            )
            self.by_key[key] = col
            self.by_raw[col.raw] = col
            self.by_name[col.name] = col

        logger.debug("Loaded {} column definitions from {}", len(self.by_key), registry_path.name)

    def raw_to_name_map(self, keys: list[str] | None = None) -> dict[str, str]:
        """Return {raw_header: standardized_name} for given keys, or all if None."""
        if keys is None:
            return {c.raw: c.name for c in self.by_key.values()}
        return {self.by_key[k].raw: self.by_key[k].name for k in keys if k in self.by_key}

    def get_raw_columns(self, category: str | None = None) -> list[str]:
        """Return list of raw CSV column names, optionally filtered by category."""
        return [
            c.raw for c in self.by_key.values()
            if category is None or c.category == category
        ]

    def get_names(self, category: str | None = None) -> list[str]:
        """Return list of standardized column names, optionally filtered by category."""
        return [
            c.name for c in self.by_key.values()
            if category is None or c.category == category
        ]

    def get_keys_for_category(self, category: str) -> list[str]:
        """Return registry keys belonging to a given category."""
        return [k for k, c in self.by_key.items() if c.category == category]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class ReplaySettings:
    speed_multiplier: int = 10
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    chunk_size: int = 100
    emit_interval_ms: int = 500


@dataclass
class QualitySettings:
    gap_threshold_seconds: float = 30.0
    spike_zscore_threshold: float = 4.0
    flatline_window: int = 20
    min_non_null_ratio: float = 0.3


@dataclass
class DataPaths:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    time_log: str = ""
    depth_log: str = ""
    production: str = ""
    rop_log: str = ""

    def resolve(self, project_root: Path) -> None:
        """Resolve relative paths against the project root."""
        self.raw_dir = str(project_root / self.raw_dir)
        self.processed_dir = str(project_root / self.processed_dir)
        self.time_log = str(project_root / self.time_log) if self.time_log else ""
        self.depth_log = str(project_root / self.depth_log) if self.depth_log else ""
        self.production = str(project_root / self.production) if self.production else ""
        self.rop_log = str(project_root / self.rop_log) if self.rop_log else ""


@dataclass
class Settings:
    project_name: str = "DrillMind"
    project_version: str = "0.1.0"
    well: str = ""
    field_name: str = ""
    operator: str = ""
    data: DataPaths = field(default_factory=DataPaths)
    replay: ReplaySettings = field(default_factory=ReplaySettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def _load_settings(settings_path: Path) -> Settings:
    """Parse settings.yaml into a Settings dataclass."""
    if not settings_path.exists():
        logger.warning("Settings file not found at {}, using defaults", settings_path)
        return Settings()

    with open(settings_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    project = raw.get("project", {})
    data_cfg = raw.get("data", {})
    replay_cfg = raw.get("replay", {})
    quality_cfg = raw.get("quality", {})
    api_cfg = raw.get("api", {})

    data_paths = DataPaths(
        raw_dir=data_cfg.get("raw_dir", "data/raw"),
        processed_dir=data_cfg.get("processed_dir", "data/processed"),
        time_log=data_cfg.get("time_log", ""),
        depth_log=data_cfg.get("depth_log", ""),
        production=data_cfg.get("production", ""),
        rop_log=data_cfg.get("rop_log", ""),
    )
    data_paths.resolve(_PROJECT_ROOT)

    settings = Settings(
        project_name=project.get("name", "DrillMind"),
        project_version=project.get("version", "0.1.0"),
        well=project.get("well", ""),
        field_name=project.get("field", ""),
        operator=project.get("operator", ""),
        data=data_paths,
        replay=ReplaySettings(**{k: v for k, v in replay_cfg.items()}),
        quality=QualitySettings(**{k: v for k, v in quality_cfg.items()}),
        api_host=api_cfg.get("host", "0.0.0.0"),
        api_port=api_cfg.get("port", 8000),
    )

    logger.debug(
        "Settings loaded — well={}, field={}",
        settings.well,
        settings.field_name,
    )
    return settings


# ---------------------------------------------------------------------------
# Module-level singletons (lazy)
# ---------------------------------------------------------------------------
_settings: Settings | None = None
_registry: ColumnRegistry | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = _load_settings(CONFIG_DIR / "settings.yaml")
    return _settings


def get_column_registry() -> ColumnRegistry:
    """Return the singleton ColumnRegistry instance."""
    global _registry
    if _registry is None:
        _registry = ColumnRegistry(CONFIG_DIR / "column_registry.yaml")
    return _registry


def get_project_root() -> Path:
    """Return the resolved project root directory."""
    return _PROJECT_ROOT
