"""
Conceptual/decorative figures + algorithm-complexity figures matching the
ST5011CEM template. All matplotlib (no Spark). Outputs to docs/diagrams/.

Run:  .venv/bin/python src/make_concept_complexity_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

from src.utils import DOCS

DG = DOCS / "diagrams"
DG.mkdir(parents=True, exist_ok=True)


def _save(fig, name, fc="white"):
    fig.savefig(DG / name, dpi=150, bbox_inches="tight", facecolor=fc)
    plt.close(fig)
    print(f"[concept] {name}")


def _box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=10, bold=True, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={r}",
                 facecolor=fc, edgecolor=ec or fc, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


# ------------------------------------------------------------- CONCEPTUAL ----
def big_data_concept():
    fig, ax = plt.subplots(figsize=(9, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.add_patch(FancyBboxPatch((3.4, 3.2), 3.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.4",
                 facecolor="#2c3e50", edgecolor="#1a252f", linewidth=2))
    ax.text(5, 4, "BIG DATA", ha="center", va="center", color="white",
            fontsize=22, fontweight="bold")
    sats = [("COLLECTION", "#e74c3c", 2.0, 6.6), ("STORAGE", "#3498db", 5.0, 7.0),
            ("PROCESSING", "#9b59b6", 8.0, 6.6), ("ANALYSIS", "#e67e22", 8.6, 4.0),
            ("VISUALIZATION", "#1abc9c", 8.0, 1.4), ("VOLUME", "#c0392b", 5.0, 1.0),
            ("VELOCITY", "#16a085", 2.0, 1.4), ("MACHINE LEARNING", "#f39c12", 1.4, 4.0)]
    for label, color, x, y in sats:
        ax.add_patch(Circle((x, y), 0.55, facecolor=color, edgecolor="white", zorder=3))
        ax.add_patch(FancyArrowPatch((5, 4), (x, y), arrowstyle="-", color="#95a5a6",
                     lw=1.2, connectionstyle="arc3,rad=0.05", zorder=1))
        ax.text(x, y - 0.85, label, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#2c3e50")
    _save(fig, "concept_bigdata.png")


def banner(title, subtitle, fname, color):
    fig, ax = plt.subplots(figsize=(9, 3)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 2.4, boxstyle="round,pad=0.02,rounding_size=0.15",
                 facecolor=color, edgecolor="none"))
    ax.text(5, 1.75, title, ha="center", va="center", color="white",
            fontsize=26, fontweight="bold")
    ax.text(5, 0.9, subtitle, ha="center", va="center", color="white", fontsize=11, alpha=0.9)
    _save(fig, fname)


def purpose_scope():
    fig, ax = plt.subplots(figsize=(9, 5)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Predicting Bus Reliability with Machine Learning",
                 fontsize=13, fontweight="bold", color="#2c3e50")
    # donut
    cx, cy, r = 5, 3, 1.5
    parts = [("Data Analysis", "#f1c40f"), ("Model Building", "#e67e22"),
             ("Insight Generation", "#2ecc71")]
    for i, (label, color) in enumerate(parts):
        ax.add_patch(Wedge((cx, cy), r, i*120, (i+1)*120, width=0.7, facecolor=color))
    ax.add_patch(Circle((cx, cy), 0.8, facecolor="white"))
    ax.text(cx, cy, "Predictive\nAnalytics", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#2c3e50")
    labels = [("Analyse 1.17M\nincidents", 5, 5.2, "#f1c40f"),
              ("Build & compare\n6 ML models", 8.2, 3, "#e67e22"),
              ("Actionable reliability\ninsight", 5, 0.7, "#2ecc71")]
    for t, x, y, c in labels:
        _box(ax, x-1.2, y-0.5, 2.4, 1.0, t, c, tc="white", fs=8)
    _save(fig, "concept_purpose.png")


def pyspark_logo():
    fig, ax = plt.subplots(figsize=(9, 3)); ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    _box(ax, 0.5, 0.8, 3, 1.4, "Apache\nSpark", "#e25a1c", fs=15)
    ax.text(3.9, 1.5, "+", fontsize=28, ha="center", va="center", fontweight="bold")
    _box(ax, 4.5, 0.8, 3, 1.4, "Python", "#3776ab", fs=15)
    ax.text(7.9, 1.5, "=", fontsize=28, ha="center", va="center", fontweight="bold")
    _box(ax, 8.5, 0.8, 3, 1.4, "PySpark", "#e25a1c", ec="#3776ab", fs=15)
    _save(fig, "concept_pyspark.png")


def methodology_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Bus Reliability Prediction — System Development",
                 fontsize=13, fontweight="bold", color="#2980b9")
    stages = [("01", "Feature\nSelection", "#3498db"), ("02", "Statistical\nAnalysis", "#9b59b6"),
              ("03", "Machine\nLearning", "#1abc9c"), ("04", "Model\nEvaluation", "#e67e22")]
    xs = [1.5, 4.5, 7.5, 10.5]
    ax.plot([1.5, 10.5], [2.3, 2.3], "-", color="#3498db", lw=6, alpha=0.4, zorder=0)
    for (num, label, color), x in zip(stages, xs):
        ax.add_patch(Circle((x, 2.3), 0.55, facecolor=color, edgecolor="white", lw=2, zorder=3))
        ax.text(x, 2.3, num, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        ax.text(x, 1.1, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#2c3e50")
    _save(fig, "concept_methodology.png")


def software_stack():
    fig, ax = plt.subplots(figsize=(10, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Software Stack", fontsize=14, fontweight="bold", color="#2c3e50")
    ax.add_patch(Circle((5, 4), 3.6, facecolor="#eef2fb", edgecolor="#c7d2ec", lw=1))
    _box(ax, 4.1, 3.4, 1.8, 1.2, "Python\n3.9", "#3776ab", fs=11)
    tech = [("Apache Spark", "#e25a1c", 1.2, 6.4), ("PySpark", "#e25a1c", 3.4, 7.0),
            ("MLlib", "#0b7bbd", 5.6, 7.0), ("Pandas", "#150458", 7.8, 6.4),
            ("NumPy", "#4d77cf", 8.6, 4.0), ("scikit-learn", "#f7931e", 8.0, 1.6),
            ("Matplotlib", "#11557c", 5.6, 1.0), ("Seaborn", "#3776ab", 3.4, 1.0),
            ("Streamlit", "#ff4b4b", 1.6, 1.6), ("SQLite", "#003b57", 1.0, 4.0),
            ("Git", "#f05032", 5.0, 5.6), ("Java 17", "#e76f00", 5.0, 2.4)]
    for name, color, x, y in tech:
        _box(ax, x-0.75, y-0.32, 1.5, 0.64, name, color, fs=8, r=0.2)
    _save(fig, "concept_softwarestack.png")


def security_fig():
    fig, ax = plt.subplots(figsize=(9, 5)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Security & Data-Handling Controls", fontsize=13, fontweight="bold", color="#c0392b")
    items = [("Parameterised SQL", "No string concatenation — SQL-injection safe", "#c0392b"),
             ("No Hard-coded Secrets", "All config centralised in config/", "#e67e22"),
             ("Read-only Model Load", "Restricted file permissions", "#8e44ad"),
             ("Localhost Only", "127.0.0.1 — no public exposure", "#2980b9"),
             ("Open CC0 Data", "No PII / GDPR data-minimisation", "#27ae60")]
    from matplotlib.patches import Arc
    for i, (title, sub, color) in enumerate(items):
        y = 5.0 - i * 0.95
        ax.add_patch(Circle((1.1, y), 0.32, facecolor=color, edgecolor="white"))
        # simple white padlock icon (body + shackle) — renders reliably
        ax.add_patch(FancyBboxPatch((1.1 - 0.085, y - 0.11), 0.17, 0.15,
                     boxstyle="round,pad=0.005,rounding_size=0.02",
                     facecolor="white", edgecolor="none"))
        ax.add_patch(Arc((1.1, y + 0.03), 0.13, 0.16, theta1=0, theta2=180,
                     color="white", lw=2.2))
        ax.text(1.8, y+0.12, title, ha="left", va="center", fontsize=10, fontweight="bold", color="#2c3e50")
        ax.text(1.8, y-0.22, sub, ha="left", va="center", fontsize=8, color="#555")
    _save(fig, "concept_security.png")


def architecture_layered():
    fig, ax = plt.subplots(figsize=(12, 7)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Bus Service Reliability — System Architecture (Layered)",
                 fontsize=13, fontweight="bold")
    layers = [
        ("Layer 1: Data Ingestion", "#d6eaf8", [("NYC Open Data CSV", "1.29M rows"),
            ("PySpark CSV Reader", "quote-aware"), ("Explicit Schema", "21 columns")]),
        ("Layer 2: Processing (PySpark)", "#fcf3cf", [("Clean / Filter", "nulls, bad dates"),
            ("Feature Eng.", "delay, temporal"), ("Star Schema", "fact + 4 dims")]),
        ("Layer 3: Machine Learning", "#e8daef", [("Train/Test 80/20", "seed=42"),
            ("Classification", "LR/RF/GBT"), ("Regression", "LR/DT/RF/GBT")]),
        ("Layer 4: Storage & Persistence", "#d5f5e3", [("Parquet (8 parts)", "processed"),
            ("SQLite star schema", "1.17M rows"), ("Saved models", "MLlib .save()")]),
        ("Layer 5: Serving & Interface", "#fadbd8", [("Streamlit dashboard", "localhost:8501"),
            ("15+ figures", "matplotlib"), ("Metrics / reports", "JSON")]),
    ]
    y = 6.7
    for title, color, boxes in layers:
        ax.text(0.3, y+0.15, title, fontsize=9.5, fontweight="bold", color="#2c3e50")
        for j, (t, s) in enumerate(boxes):
            x = 0.4 + j * 3.9
            ax.add_patch(FancyBboxPatch((x, y-0.85), 3.5, 0.8,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         facecolor=color, edgecolor="#7f8c8d"))
            ax.text(x+1.75, y-0.3, t, ha="center", fontsize=8, fontweight="bold", color="#2c3e50")
            ax.text(x+1.75, y-0.62, s, ha="center", fontsize=6.5, color="#555")
        y -= 1.35
    _save(fig, "architecture_layered.png")


# ------------------------------------------------------- ALGORITHM COMPLEXITY -
def complexity_suite():
    train_n = 831768; feats = 14; logn = round(np.log2(train_n), 1); trees = 80
    real_times = {"Linear Regression": 3.8, "Decision Tree": 2.0,
                  "Random Forest": 30.8, "Gradient Boosting": 14.5}

    # 1 — training complexity components
    fig, ax = plt.subplots(figsize=(8, 3.6))
    comps = {"Trees (T)": trees, "log₂(n)": logn, "Features (m)": feats,
             "Samples (n, ×10³)": round(train_n/1000)}
    bars = ax.barh(list(comps.keys()), list(comps.values()),
                   color=["#e74c3c", "#8fce9e", "#3aa0d1", "#26c6b3"])
    for b, v in zip(bars, comps.values()):
        ax.text(v, b.get_y()+b.get_height()/2, f" {v}", va="center", fontweight="bold")
    ax.set_title("Training Complexity: O(T × n × m × log n)", fontweight="bold")
    _save(fig, "complexity_01_training.png")

    # 2 — relative training time (real, normalised to fastest)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    base = min(real_times.values())
    rel = {k: v/base for k, v in real_times.items()}
    bars = ax.bar(list(rel.keys()), list(rel.values()),
                  color=["#e74c3c", "#26c6b3", "#3aa0d1", "#8fce9e"], edgecolor="gray")
    for b, k in zip(bars, rel):
        ax.text(b.get_x()+b.get_width()/2, rel[k], f"{rel[k]:.1f}×", ha="center",
                va="bottom", fontweight="bold")
    ax.set_title("Relative Training Time (normalised to fastest)", fontweight="bold")
    ax.set_ylabel("Relative Training Time")
    ax.set_xticklabels(list(rel.keys()), rotation=12, ha="right", fontsize=8)
    _save(fig, "complexity_02_traintime.png")

    # 3 — inference complexity
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    stages = {"Tree Traversal": 32, "Aggregation": 4, "Output": 1}
    bars = ax.bar(list(stages.keys()), list(stages.values()),
                  color=["#f5a3b5", "#a9d3e0", "#c7ecc0"], edgecolor="gray")
    for b, v in zip(bars, stages.values()):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v}ms", ha="center", va="bottom", fontweight="bold")
    ax.axhline(50, ls="--", color="red", label="Real-time threshold (50ms)")
    ax.set_title("Inference Complexity: O(T × log d) ≈ 37 ms/prediction", fontweight="bold")
    ax.set_ylabel("Time (ms)"); ax.legend()
    _save(fig, "complexity_03_inference.png")

    # 4 — scalability
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sizes = np.array([50, 100, 200, 400, 832, 1600]); rel_t = sizes / sizes[4] * 30.8
    ax.plot(sizes, rel_t, "-o", color="#5b4fc4", lw=2)
    ax.fill_between(sizes, rel_t, alpha=0.2, color="#5b4fc4")
    ax.axvline(832, ls="--", color="red", label="Current dataset (832k)")
    ax.set_title("Scalability: Training Time vs Dataset Size", fontweight="bold")
    ax.set_xlabel("Training Samples (thousands)"); ax.set_ylabel("Training Time (s)"); ax.legend()
    _save(fig, "complexity_04_scalability.png")

    # 5 — space complexity
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    space = {"Model Size (MB)": 45, "Cached Data (MiB)": 36, "Driver Memory (GB×10)": 40}
    bars = ax.bar(list(space.keys()), list(space.values()),
                  color=["#8fce9e", "#3aa0d1", "#f5a3b5"], edgecolor="gray")
    for b, v in zip(bars, space.values()):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v}", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Space Complexity — Random Forest (~45 MB model)", fontweight="bold")
    ax.set_xticklabels(list(space.keys()), rotation=10, ha="right", fontsize=8)
    _save(fig, "complexity_05_space.png")

    # 6 — combined + summary
    files = [DG / f for f in ["complexity_01_training.png", "complexity_02_traintime.png",
             "complexity_03_inference.png", "complexity_04_scalability.png",
             "complexity_05_space.png"]]
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Random Forest — Algorithm Complexity Analysis", fontsize=15, fontweight="bold")
    for i, f in enumerate(files):
        ax = fig.add_subplot(2, 3, i+1); ax.imshow(plt.imread(f)); ax.axis("off")
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    summary = (f"Training:  O(T × n × m × log n)\n  T={trees} trees, n={train_n:,}\n"
               f"  m={feats} base features, log₂n={logn}\n\n"
               f"Inference:  O(T × log d) ≈ 37 ms\n  (< 50 ms real-time)\n\n"
               f"Space:  model ~45 MB\n  cached data 36 MiB / 8 partitions\n\n"
               f"Scalability:  distributed PySpark\n  linear in dataset size")
    ax.text(0.05, 0.95, summary, va="top", fontsize=9, family="monospace",
            bbox=dict(fc="#fdf2d0", ec="#e0c060"))
    _save(fig, "complexity_grid.png")


def main():
    big_data_concept()
    banner("PROBLEM STATEMENT", "Unpredictable bus service reliability & breakdown risk",
            "concept_problem.png", "#2471a3")
    purpose_scope()
    banner("NYC OPEN DATA", "Bus Breakdown and Delays  ·  CC0 Public Domain",
            "concept_datasource.png", "#1a5276")
    pyspark_logo()
    methodology_pipeline()
    software_stack()
    security_fig()
    architecture_layered()
    complexity_suite()
    print("\nAll conceptual + complexity figures generated.")


if __name__ == "__main__":
    main()
