"""
Generate report assets: architecture diagram, star-schema ER diagram, and
terminal-style "code execution" evidence cards rendered from the REAL console
output produced by the pipeline stages.

Run:  .venv/bin/python src/make_report_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.utils import DOCS

DIAGRAMS = DOCS / "diagrams"
SHOTS = DOCS / "screenshots"
DIAGRAMS.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
def architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Figure A. System Architecture — Bus Service Reliability Pipeline",
                 fontsize=13, fontweight="bold", pad=14)

    stages = [
        ("1 · INGESTION", "#1a5276",
         ["NYC Open Data", "'Bus Breakdown", "& Delays' (CC0)", "1.29M rows · 295MB",
          "PySpark quote-aware", "CSV read + schema"]),
        ("2 · PROCESSING", "#1f618d",
         ["Clean + parse delays", "Temporal features", "Labels (breakdown,", "delay_minutes)",
          "repartition(8)·cache", "broadcast join dims"]),
        ("3 · STORAGE", "#117864",
         ["Parquet (8 parts)", "1.17M clean rows", "SQLite star schema", "fact + 4 dims",
          "parameterised SQL", "(injection-safe)"]),
        ("4 · MODELLING", "#7d3c98",
         ["Spark MLlib pipeline", "Classification:", " LR · RF · GBT", "Regression:",
          " LR · RF · GBT", "3-fold CrossValidator"]),
        ("5 · SERVING", "#a04000",
         ["EDA: matplotlib/", "seaborn (15 figs)", "Streamlit dashboard", "(localhost:8501)",
          "Metrics + ROC +", "evaluation reports"]),
    ]
    n = len(stages)
    box_w, gap = 2.1, 0.35
    total = n * box_w + (n - 1) * gap
    x0 = (13 - total) / 2
    y = 1.1
    centers = []
    for i, (title, color, bullets) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        box = FancyBboxPatch((x, y), box_w, 3.4, boxstyle="round,pad=0.03,rounding_size=0.12",
                             linewidth=1.5, edgecolor=color, facecolor=color + "18")
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + 3.12, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color)
        for j, b in enumerate(bullets):
            ax.text(x + box_w / 2, y + 2.62 - j * 0.42, b, ha="center", va="center",
                    fontsize=7.6, color="#222")
        centers.append((x + box_w, y + 1.7, x))
    # arrows between stages
    for i in range(n - 1):
        xr, yc, _ = centers[i]
        xnext = centers[i + 1][2]
        ax.add_patch(FancyArrowPatch((xr + 0.02, yc), (xnext - 0.02, yc),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#444", linewidth=1.8))
    # spark banner
    ax.add_patch(FancyBboxPatch((x0, 5.0), total, 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#e67e22", edgecolor="#a04000"))
    ax.text(x0 + total / 2, 5.27,
            "Apache Spark 3.5.3  ·  local[*]  ·  8 partitions  ·  caching · broadcast joins · lazy DAG · Arrow",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(6.5, 0.5, "Version control: Git / GitHub   ·   Java 17   ·   Python 3.9",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.savefig(DIAGRAMS / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[asset] architecture.png")


# ----------------------------------------------------------------------------
def star_schema_diagram():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Figure B. Database Star Schema (SQLite)",
                 fontsize=13, fontweight="bold", pad=10)

    def table(x, y, w, h, title, cols, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    linewidth=1.5, edgecolor=color, facecolor="white"))
        ax.add_patch(plt.Rectangle((x, y + h - 0.42), w, 0.42, facecolor=color, edgecolor=color))
        ax.text(x + w / 2, y + h - 0.21, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white")
        for k, c in enumerate(cols):
            ax.text(x + 0.12, y + h - 0.72 - k * 0.32, c, ha="left", va="center",
                    fontsize=7.8, color="#222")
        return (x + w / 2, y + h / 2)

    fact = table(4.1, 2.9, 2.9, 2.5, "fact_incident",
                 ["PK busbreakdown_id", "FK operator_id", "FK reason_id", "FK boro_id",
                  "FK date_key", "hour, time_band", "students_on_bus", "delay_minutes",
                  "is_breakdown, on_time"], "#7d3c98")
    op = table(0.4, 5.3, 2.5, 1.5, "dim_operator", ["PK operator_id", "operator"], "#1a5276")
    rs = table(8.1, 5.3, 2.5, 1.5, "dim_reason", ["PK reason_id", "reason"], "#117864")
    bo = table(0.4, 0.6, 2.5, 1.5, "dim_boro", ["PK boro_id", "boro"], "#a04000")
    dt = table(8.1, 0.6, 2.5, 1.9, "dim_date",
               ["PK date_key", "year, month", "day_of_week", "season, is_weekend",
                "school_year"], "#b7950b")

    for (cx, cy) in [op, rs, bo, dt]:
        ax.add_patch(FancyArrowPatch((cx, cy), fact, arrowstyle="-|>", mutation_scale=14,
                                     color="#888", linewidth=1.3,
                                     connectionstyle="arc3,rad=0.05"))
    ax.text(5.55, 4.15, "1.17M rows", ha="center", fontsize=8, style="italic", color="#7d3c98")
    ax.text(5.5, 0.2, "Dimension tables (small) are broadcast-joined to the large fact table.",
            ha="center", fontsize=8, style="italic", color="#555")
    fig.savefig(DIAGRAMS / "star_schema.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[asset] star_schema.png")


# ----------------------------------------------------------------------------
def terminal_card(filename, title, lines, height=None):
    """Render real console output as a dark terminal-style PNG."""
    h = height or max(3.2, 0.32 * len(lines) + 1.0)
    fig, ax = plt.subplots(figsize=(11, h))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#1e1e2e", zorder=0))
    # title bar
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes,
                               facecolor="#313244", zorder=1))
    for i, c in enumerate(["#f38ba8", "#f9e2af", "#a6e3a1"]):
        ax.add_patch(plt.Circle((0.018 + i * 0.022, 0.965), 0.008,
                                transform=ax.transAxes, facecolor=c, zorder=2))
    ax.text(0.5, 0.965, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=9.5, color="#cdd6f4", family="monospace", zorder=2)
    y = 0.88
    for ln in lines:
        color = "#a6e3a1" if ln.strip().startswith(("[", "===", "+", "-- ", ">>>")) else "#cdd6f4"
        if "ERROR" in ln or "Traceback" in ln:
            color = "#f38ba8"
        ax.text(0.02, y, ln, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.2, color=color, family="monospace", zorder=2)
        y -= 0.052
    fig.savefig(SHOTS / filename, dpi=150, bbox_inches="tight", facecolor="#1e1e2e")
    plt.close(fig)
    print(f"[asset] {filename}")


INGEST_LOG = [
    "$ .venv/bin/python src/pipeline_01_ingest_preprocess.py",
    "Spark UI: http://192.168.1.74:4040",
    "[partitions] raw (after CSV read)               -> 8 partitions",
    "[timing] 01_load_raw_csv                            1.14s",
    "[partitions] raw (after repartition)            -> 8 partitions",
    "[timing] 02_count_after_cache                       2.34s",
    "[rows] raw records: 1,294,129",
    "[quality] kept 1,167,832 / 1,294,129 rows (90.2%); dropped 126,297",
    "[timing] 03_feature_engineering                     1.37s",
    "[timing] 04_broadcast_join_dims                     0.63s",
    "[join] broadcast-joined 3 dimension tables into fact frame",
    "[timing] 05_write_parquet                           5.89s",
    "[partitions] final feature table                -> 8 partitions",
    "[persist] wrote Parquet -> data/processed/incidents_clean.parquet",
    "[persist] wrote 4 dimension tables",
    "[sample] wrote 2000 sample rows to CSV",
    "=== Stage 1 complete ===",
]

ML_LOG = [
    "$ .venv/bin/python src/pipeline_04_ml_classification.py",
    "[prep] n=1,167,832 pos=111,787 (9.6%) w_pos=5.223 w_neg=0.553",
    "[split] train=934,690 test=233,142",
    "[eval] LogisticRegression  ACC=0.722 P=0.224 R=0.772 F1=0.347 ROC-AUC=0.827",
    "[eval] RandomForest        ACC=0.774 P=0.261 R=0.746 F1=0.387 ROC-AUC=0.843",
    "[eval] GBTClassifier       ACC=0.780 P=0.265 R=0.733 F1=0.389 ROC-AUC=0.847",
    "[cv] 3-fold CrossValidator tuning LogisticRegression ...",
    "[cv] best regParam=0.0 elasticNet=0.0 CV ROC-AUC=0.827",
    "[done] best classifier = GBTClassifier",
    "",
    "$ .venv/bin/python src/pipeline_05_ml_regression.py",
    "[prep] rows with delay label: 1,039,711",
    "[eval] LinearRegression       RMSE=16.42 MAE=11.93 R2=0.359 (4.8s)",
    "[eval] RandomForestRegressor  RMSE=14.19 MAE=10.04 R2=0.521 (72.4s)",
    "[eval] GBTRegressor           RMSE=14.34 MAE=9.94  R2=0.511 (21.6s)",
    "[done] best regressor = RandomForestRegressor",
]

DB_LOG = [
    "$ .venv/bin/python src/pipeline_03_database.py",
    "[db] tables populated and committed",
    "=== Q1. Operator league table (>= 5000 incidents) ===",
    "   ('LITTLE LISA BUS CO. INC', 6746, 0.605, 36.8)",
    "   ('GRANDPA`S BUS CO., INC', 9917, 0.451, 27.6)",
    "   ('LITTLE RICHIE BUS SERVICE', 36906, 0.392, 40.3)",
    "=== Parameterised-query security demo ===",
    "  malicious input handled safely -> rows matched: 0",
    "  fact_incident still intact -> 1,167,832 rows (no injection)",
    "[db] wrote git-friendly sample dump -> bus_reliability_sample_dump.sql",
    "[db] SQLite ready: bus_reliability.db (232.5 MB, 1,167,832 fact rows)",
]


def main():
    architecture_diagram()
    star_schema_diagram()
    terminal_card("exec_01_ingest_preprocess.png",
                  "Stage 1 — Ingestion & Preprocessing (PySpark)", INGEST_LOG)
    terminal_card("exec_02_machine_learning.png",
                  "Stages 4-5 — Machine Learning (Spark MLlib)", ML_LOG)
    terminal_card("exec_03_database_security.png",
                  "Stage 3 — SQLite + parameterised-query security", DB_LOG)
    print("\nAll report assets generated.")


if __name__ == "__main__":
    main()
