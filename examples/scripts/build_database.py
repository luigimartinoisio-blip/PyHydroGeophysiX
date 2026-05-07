"""
02_build_raw_database.py
========================
PyHydroGeophysiX · Macro-Area 1 · Layer 1 → Layer 2 hand-off
--------------------------------------------------------------
Reads every validated raw field file through the connector layer and
persists the standardised Pandas DataFrames as Parquet files inside
``data/processed/``.

Run from the project root
-------------------------
    python scripts/02_build_raw_database.py

Output layout
-------------
    data/processed/
    ├── cosmos_soil_moisture.parquet   – daily COSMOS VSM + precipitation
    ├── topography.parquet             – electrode topographic coordinates
    ├── electrode_gps.parquet          – GPS geodetic coordinates
    ├── sequence_electrodes.parquet    – electrode grid (from sequence file)
    └── sequence_measurements.parquet – ERT quadripole measurement list
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – works whether the script is launched from the project root
# or from the scripts/ directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_ingestion import ConnectorFactory  # noqa: E402 (must follow sys.path patch)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("build_raw_database")

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write *df* to *dest* as Parquet and log the result."""
    df.to_parquet(dest, index=True, engine="pyarrow")
    size_kb = dest.stat().st_size / 1024
    logger.info("  ✓  Saved  %-45s  (%d rows, %d cols, %.1f KB)",
                str(dest.relative_to(_PROJECT_ROOT)),
                len(df), len(df.columns), size_kb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 65)
    logger.info("PyHydroGeophysiX – Raw Database Builder")
    logger.info("Project root : %s", _PROJECT_ROOT)
    logger.info("=" * 65)

    # Ensure the output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory : %s", PROCESSED_DIR)

    errors: list[str] = []

    # ── 1. COSMOS soil moisture ─────────────────────────────────────────────
    logger.info("")
    logger.info("── [1/4] COSMOS Soil Moisture ─────────────────────────────")
    cosmos_src = RAW_DIR / "COSMOS_SOIL_MOISTURE" / "COSMOS-Carlone.xlsx.csv"
    try:
        df_cosmos = ConnectorFactory.get("cosmos").parse_data(cosmos_src)
        _save_parquet(df_cosmos, PROCESSED_DIR / "cosmos_soil_moisture.parquet")
    except Exception as exc:
        logger.error("  ✗  COSMOS failed: %s", exc)
        errors.append(f"cosmos: {exc}")

    # ── 2. Topography (electrode positions from field survey) ───────────────
    logger.info("")
    logger.info("── [2/4] Topography (#,X,Y,Z,Depth) ──────────────────────")
    topo_src = RAW_DIR / "Position" / "08topog_TL.txt"
    try:
        df_topo = ConnectorFactory.get("spatial").parse_data(topo_src)
        _save_parquet(df_topo, PROCESSED_DIR / "topography.parquet")
    except Exception as exc:
        logger.error("  ✗  Topography failed: %s", exc)
        errors.append(f"topography: {exc}")

    # ── 3. Electrode GPS coordinates ────────────────────────────────────────
    logger.info("")
    logger.info("── [3/4] Electrode GPS coordinates ────────────────────────")
    gps_src = RAW_DIR / "Position" / "electr_coord.csv"
    try:
        df_gps = ConnectorFactory.get("spatial").parse_data(gps_src)
        _save_parquet(df_gps, PROCESSED_DIR / "electrode_gps.parquet")
    except Exception as exc:
        logger.error("  ✗  GPS coordinates failed: %s", exc)
        errors.append(f"electrode_gps: {exc}")

    # ── 4. ERT sequence (two-block file → two Parquet files) ────────────────
    logger.info("")
    logger.info("── [4/4] ERT Sequence file ─────────────────────────────────")
    seq_src = RAW_DIR / "sequence" / "2DDsup+MG+DDsup_rec.txt"
    try:
        df_elec, df_seq = ConnectorFactory.get("sequence").parse_data(seq_src)
        _save_parquet(df_elec, PROCESSED_DIR / "sequence_electrodes.parquet")
        _save_parquet(df_seq,  PROCESSED_DIR / "sequence_measurements.parquet")
    except Exception as exc:
        logger.error("  ✗  Sequence failed: %s", exc)
        errors.append(f"sequence: {exc}")

    # ── Final report ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    if errors:
        logger.error("Build finished with %d error(s):", len(errors))
        for e in errors:
            logger.error("  • %s", e)
        sys.exit(1)
    else:
        parquet_files = sorted(PROCESSED_DIR.glob("*.parquet"))
        logger.info("Build complete – %d Parquet files written to data/processed/:",
                    len(parquet_files))
        for pf in parquet_files:
            logger.info("  → %s", pf.name)
        logger.info("=" * 65)


if __name__ == "__main__":
    main()
