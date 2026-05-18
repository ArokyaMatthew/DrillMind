"""One-shot audit: exact column names, coverage, and sample values."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from drillmind.parsers.time_log_parser import load_time_log

df = load_time_log(nrows=5000)

print("=== COLUMNS WITH >50% COVERAGE ===")
coverage = df.select_dtypes("number").notna().mean().sort_values(ascending=False)
for col, ratio in coverage.items():
    if ratio > 0.5:
        print(f"  {ratio:.2f}  {col}")

print("\n=== COLUMNS WITH <50% BUT >0% COVERAGE ===")
for col, ratio in coverage.items():
    if 0 < ratio <= 0.5:
        print(f"  {ratio:.4f}  {col}")

print("\n=== KEY KPI COLUMNS — sample values ===")
kpi_cols = [
    "wob_avg", "rpm_avg", "torque_averaged", "spp",
    "flow_pumps", "bit_depth", "tvd", "mud_weight_in",
    "mud_weight_out", "rop_avg", "hookload_max",
    "weight_on_hook", "pit_volume_active", "gas_total",
]
for c in kpi_cols:
    if c in df.columns:
        vals = df[c].dropna()
        print(f"  {c:25s}  coverage={vals.shape[0]/len(df):.2f}  sample={vals.iloc[:3].tolist()}")
    else:
        print(f"  {c:25s}  *** NOT IN DATAFRAME ***")

print("\n=== ALL COLUMN NAMES ===")
for c in sorted(df.columns):
    print(f"  {c}")
