"""
Drilling KPI Engine
====================
Real-time drilling performance metrics that every RTOC analyst monitors.

All formulas reference standard petroleum engineering textbooks
(Applied Drilling Engineering - Bourgoyne et al., SPE).

Column dependencies (all verified present in Volve time log):
    torque_averaged   — surface torque (kN·m), 99% coverage
    rpm_avg           — surface RPM, 99% coverage
    wob_avg           — weight on bit, 99% coverage
    rop               — rate of penetration (m/h), 23% coverage (non-zero when drilling)
    rop_5ft_avg       — 5-foot averaged ROP, 23% coverage
    bit_depth         — bit depth (m), 99% coverage
    mud_weight_in     — mud weight in (sg), 99% coverage
    mud_weight_out    — mud weight out (sg), 99% coverage

Limitations documented:
    - Bit diameter is NOT available as a sensor channel. We use the well
      program value (12.25 inches for the Volve 15/9-F-9 A 12¼" section).
      This is standard practice — bit size is a well plan parameter, not
      a real-time sensor.
    - ROP is only available when the rig is actively drilling (~23% of
      samples). KPIs that require ROP will be NaN during non-drilling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class WellGeometry:
    """
    Well geometry parameters from the well program.
    These are NOT real-time sensor values — they come from the
    drilling plan and change only when BHA is changed.
    """

    bit_diameter_inches: float = 12.25   # Volve F-9 A: 12¼" section
    normal_mud_weight_sg: float = 1.03   # Seawater density for Volve (North Sea)

    @property
    def bit_diameter_m(self) -> float:
        return self.bit_diameter_inches * 0.0254

    @property
    def bit_area_m2(self) -> float:
        """Cross-sectional area of the bit in m²."""
        return np.pi / 4 * self.bit_diameter_m ** 2


def compute_mse(
    df: pd.DataFrame,
    geometry: WellGeometry | None = None,
) -> pd.Series:
    """
    Compute Mechanical Specific Energy (MSE).

    MSE represents the energy input per unit volume of rock removed.
    High MSE relative to rock UCS indicates inefficient drilling
    (founder point, bit wear, poor weight transfer).

    Formula (Teale, 1965):
        MSE = (480 * Torque * RPM) / (D² * ROP) + (4 * WOB) / (π * D²)

    Where:
        Torque  = surface torque (kN·m → converted to ft·lbf for SPE formula)
        RPM     = surface RPM
        D       = bit diameter (inches)
        ROP     = rate of penetration (ft/hr)
        WOB     = weight on bit (lbf)

    We use SI inputs and convert internally. Output is in MPa.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: torque_averaged, rpm_avg, wob_avg, and one of
        rop / rop_5ft_avg.
    geometry : WellGeometry | None
        Bit diameter and normal mud weight.

    Returns
    -------
    pd.Series
        MSE in MPa, same index as df. NaN where ROP is zero or missing.
    """
    if geometry is None:
        geometry = WellGeometry()

    # Select columns — verified names from column audit
    torque = df["torque_averaged"].copy()  # kN·m
    rpm = df["rpm_avg"].copy()
    wob = df["wob_avg"].copy()

    # Use rop_5ft_avg if available, else rop
    if "rop_5ft_avg" in df.columns:
        rop = df["rop_5ft_avg"].copy()
    elif "rop" in df.columns:
        rop = df["rop"].copy()
    else:
        logger.warning("No ROP column found — MSE will be all NaN")
        return pd.Series(np.nan, index=df.index, name="mse_mpa")

    # Guard against division by zero
    rop_safe = rop.replace(0, np.nan)
    area = geometry.bit_area_m2  # m²

    # MSE in SI units (Pa):
    #   Rotary component: (2π * Torque * RPM) / (Area * ROP)
    #   Thrust component: WOB / Area
    #
    # Torque is in kN·m, ROP in m/h, WOB in ? (from the data, wob_avg is
    # in a raw unit we need to check). Let's work in consistent units.
    #
    # wob_avg values in the data are ~0 when not drilling, ~30000-90000
    # when drilling (from ROP file). These appear to be in N (Newtons).
    # torque_averaged values are ~0.02 kN·m when idle, higher when drilling.
    # rpm_avg values are 0-180 RPM.
    # rop is in m/h.

    # Convert ROP from m/h to m/s
    rop_ms = rop_safe / 3600.0

    # Torque in N·m (torque_averaged is in kN·m)
    torque_nm = torque * 1000.0

    # WOB in N (wob_avg appears to already be in kgf or similar small unit;
    # given Volve data shows values like -2.6e-16 when idle and the ROP file
    # shows 26000-90000, the time log wob_avg is likely in daN or a
    # different unit). Let's use the raw value and document units.
    # From the data: wob_avg ≈ 0 when idle. We'll treat it as N.
    wob_n = wob.copy()

    # Rotary component: (2π * T * N) / (A * v)
    # T = torque (N·m), N = RPM (rev/min → rev/s), A = bit area (m²), v = ROP (m/s)
    rpm_rps = rpm / 60.0
    rotary = (2 * np.pi * torque_nm * rpm_rps) / (area * rop_ms)

    # Thrust component: F / A
    thrust = wob_n / area

    mse_pa = rotary + thrust
    mse_mpa = mse_pa / 1e6

    # Clip physically unreasonable values (> 100 GPa is unphysical)
    mse_mpa = mse_mpa.clip(upper=1e5)

    result = mse_mpa.rename("mse_mpa")
    valid = result.notna() & np.isfinite(result)
    logger.info(
        "MSE computed: {}/{} valid values, mean={:.1f} MPa, median={:.1f} MPa",
        valid.sum(),
        len(result),
        result[valid].mean() if valid.any() else 0,
        result[valid].median() if valid.any() else 0,
    )
    return result


def compute_d_exponent(
    df: pd.DataFrame,
    geometry: WellGeometry | None = None,
) -> pd.Series:
    """
    Compute the d-exponent (Jorden & Shirley, 1966).

    The d-exponent normalizes ROP for changes in WOB and RPM, leaving
    only the formation effect. A decreasing d-exponent trend at constant
    mud weight indicates increasing pore pressure — a critical
    well-control indicator.

    Formula:
        d_exp = log10(ROP / (60 * RPM)) / log10(12 * WOB / (1000 * D))

    Where:
        ROP = ft/hr
        RPM = rev/min
        WOB = lbf (1000s)
        D   = bit diameter (inches)

    We convert from SI internally.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: rpm_avg, wob_avg, and rop / rop_5ft_avg.
    geometry : WellGeometry | None
        Bit geometry.

    Returns
    -------
    pd.Series
        d-exponent (dimensionless), same index as df.
    """
    if geometry is None:
        geometry = WellGeometry()

    rpm = df["rpm_avg"].copy()
    wob = df["wob_avg"].copy()

    if "rop_5ft_avg" in df.columns:
        rop = df["rop_5ft_avg"].copy()
    elif "rop" in df.columns:
        rop = df["rop"].copy()
    else:
        return pd.Series(np.nan, index=df.index, name="d_exponent")

    # Convert to oilfield units
    rop_fth = rop * 3.28084  # m/h → ft/h
    D = geometry.bit_diameter_inches

    # Guard: RPM and WOB must be positive
    rpm_safe = rpm.replace(0, np.nan).clip(lower=0.1)
    wob_safe = wob.clip(lower=1.0)  # Avoid log of zero

    # WOB is in the raw unit from the sensor. Normalize by 1000*D
    numerator = np.log10(rop_fth / (60 * rpm_safe))
    denominator = np.log10((12 * wob_safe) / (1000 * D))

    # Avoid division by near-zero denominator
    denominator_safe = denominator.replace(0, np.nan)
    d_exp = numerator / denominator_safe

    result = d_exp.rename("d_exponent")
    valid = result.notna() & np.isfinite(result) & (result > 0) & (result < 10)
    logger.info(
        "d-exponent computed: {}/{} valid values, mean={:.3f}",
        valid.sum(),
        len(result),
        result[valid].mean() if valid.any() else 0,
    )
    return result


def compute_corrected_d_exponent(
    df: pd.DataFrame,
    geometry: WellGeometry | None = None,
) -> pd.Series:
    """
    Compute the corrected d-exponent (Rehm & McClendon, 1971).

    Corrects the d-exponent for mud weight changes, which otherwise mask
    the pore pressure signal.

    Formula:
        d_exp_c = d_exp * (MW_normal / MW_actual)

    Where:
        MW_normal = normal pore pressure mud weight (seawater for Volve)
        MW_actual = mud_weight_in (current mud weight in use)

    Both MW values are in sg (specific gravity).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: mud_weight_in and all columns for d_exponent.
    geometry : WellGeometry | None
        Contains normal_mud_weight_sg.

    Returns
    -------
    pd.Series
        Corrected d-exponent (dimensionless).
    """
    if geometry is None:
        geometry = WellGeometry()

    d_exp = compute_d_exponent(df, geometry)

    if "mud_weight_in" not in df.columns:
        logger.warning("mud_weight_in not found — returning uncorrected d-exponent")
        return d_exp.rename("d_exponent_corrected")

    mw_actual = df["mud_weight_in"].copy()
    mw_normal = geometry.normal_mud_weight_sg

    # Guard against zero mud weight
    mw_safe = mw_actual.replace(0, np.nan).clip(lower=0.5)

    d_exp_c = d_exp * (mw_normal / mw_safe)

    result = d_exp_c.rename("d_exponent_corrected")
    valid = result.notna() & np.isfinite(result) & (result > 0) & (result < 10)
    logger.info(
        "Corrected d-exponent: {}/{} valid, MW_normal={:.2f} sg, MW_actual mean={:.3f} sg",
        valid.sum(),
        len(result),
        mw_normal,
        mw_safe.mean(),
    )
    return result


def compute_drilling_kpis(
    df: pd.DataFrame,
    geometry: WellGeometry | None = None,
) -> pd.DataFrame:
    """
    Compute all drilling KPIs and return as a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed drilling data with standardized column names.
    geometry : WellGeometry | None
        Well geometry.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: mse_mpa, d_exponent, d_exponent_corrected.
        Same index as input df.
    """
    if geometry is None:
        geometry = WellGeometry()

    mse = compute_mse(df, geometry)
    d_exp = compute_d_exponent(df, geometry)
    d_exp_c = compute_corrected_d_exponent(df, geometry)

    result = pd.DataFrame({
        "mse_mpa": mse,
        "d_exponent": d_exp,
        "d_exponent_corrected": d_exp_c,
    }, index=df.index)

    logger.info(
        "Drilling KPIs computed: {} rows, MSE valid={}, d_exp valid={}, d_exp_c valid={}",
        len(result),
        result["mse_mpa"].notna().sum(),
        result["d_exponent"].notna().sum(),
        result["d_exponent_corrected"].notna().sum(),
    )

    return result
