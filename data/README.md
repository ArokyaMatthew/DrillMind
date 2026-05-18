# Data Directory

This directory holds the Volve dataset files. They are not tracked in git due to size.

## Required Files

Download from [Equinor Volve Data Portal](https://www.equinor.com/energy/volve-data-sharing):

| File | Size | Description |
|------|------|-------------|
| `Norway-NA-15_47_9-F-9 A time.csv` | ~408 MB | **Required** — time-indexed drilling telemetry |
| `Norway-NA-15_47_9-F-9 A depth.csv` | ~5 MB | Optional — depth-indexed LWD/MWD data |
| `ROP data .csv` | ~10 KB | Optional — ROP + petrophysics |
| `Volve production data.xlsx` | ~2 MB | Optional — production history |

Place these files in `data/raw/` before running DrillMind.
