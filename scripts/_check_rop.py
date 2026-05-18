"""Check ROP availability across the full dataset."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from drillmind.parsers.time_log_parser import load_time_log

df = load_time_log(nrows=200000)

print("ROP coverage across 200K rows:")
for c in ["rop", "rop_2min_avg", "rop_5ft_avg"]:
    series = df[c].dropna()
    nz = (series.abs() > 0.001).sum()
    print(f"  {c:20s}: total={len(series):>6d}, nonzero={nz:>6d}")

print()
bd = df["bit_depth"].dropna()
print(f"bit_depth range: {bd.min():.1f} -> {bd.max():.1f} m")
# Compute ROP from bit_depth derivative
dt = df.index.to_series().diff().dt.total_seconds()
d_depth = bd.diff()
computed_rop = (d_depth / dt * 3600).dropna()  # m/h
pos_rop = computed_rop[computed_rop > 0.01]
print(f"Computed ROP (from d(depth)/dt): {len(pos_rop)} positive samples")
if len(pos_rop) > 0:
    print(f"  mean={pos_rop.mean():.3f} m/h, max={pos_rop.max():.3f} m/h")
