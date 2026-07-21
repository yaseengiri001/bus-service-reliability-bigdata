"""
Shared helpers: project paths, wall-clock timing (algorithmic-efficiency
evidence), consistent figure saving, and Spark partition reporting.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

# --- canonical project paths (resolved from this file, space-safe) -----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"
DATABASE = PROJECT_ROOT / "database"
DOCS = PROJECT_ROOT / "docs"

for _p in (DATA_PROCESSED, FIGURES, MODELS, DATABASE):
    _p.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "bus_breakdown_delays.csv"
PROCESSED_PARQUET = DATA_PROCESSED / "incidents_clean.parquet"
SQLITE_DB = DATABASE / "bus_reliability.db"

# A single place to record every runtime measurement we report in the paper.
TIMINGS: dict[str, float] = {}


@contextmanager
def timed(label: str):
    """Context manager that records wall-clock seconds into TIMINGS[label]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        TIMINGS[label] = elapsed
        print(f"[timing] {label:<38} {elapsed:8.2f}s")


def partition_report(df, name: str) -> int:
    """Print and return the number of partitions backing a DataFrame."""
    n = df.rdd.getNumPartitions()
    print(f"[partitions] {name:<34} -> {n} partitions")
    return n


def save_fig(fig, filename: str, caption: str = "") -> Path:
    """Save a matplotlib figure to outputs/figures with tight bounding box."""
    path = FIGURES / filename
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"[figure] saved {path.name}" + (f"  ({caption})" if caption else ""))
    return path
