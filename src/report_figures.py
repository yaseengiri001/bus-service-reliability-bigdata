"""
Report figure generator — data-driven figures matching the ST5011CEM template.
Produces (into outputs/figures/):
  * EDA suite (10)             r_eda_*.png  + r_eda_grid.png
  * Model comparison suite (4) r_cmp_*.png  + r_cmp_grid.png
  * Model evaluation suite (6) r_eval_*.png + r_eval_grid.png

EDA uses PySpark aggregations (toPandas only at plot time). Evaluation uses the
saved regression_predictions.csv. Comparison uses regression_metrics.json.

Run:  .venv/bin/python src/report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from pyspark.sql import functions as F
from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, FIGURES, OUTPUTS

sns.set_theme(style="whitegrid")
PALETTE = ["#e74c6c", "#26c6b3", "#3aa0d1", "#8fce9e"]   # LR, DT, RF, GBT
DOW = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}


def _save(fig, name):
    fig.savefig(FIGURES / name, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}")


# ============================================================ EDA =============
def eda_suite(spark):
    df = spark.read.parquet(str(PROCESSED_PARQUET)).cache()

    # 1 — distribution of delay minutes
    pdf = df.filter(F.col("delay_minutes").isNotNull()).select("delay_minutes") \
            .sample(0.15, seed=1).toPandas()
    mean = pdf["delay_minutes"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(pdf["delay_minutes"], bins=40, color="#5b8fc9", edgecolor="white")
    ax.axvline(mean, color="red", ls="--", lw=2, label=f"Mean: {mean:.1f} min")
    ax.set_title("Distribution of Delay Minutes"); ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Frequency"); ax.legend()
    _save(fig, "r_eda_01_delay_dist.png")

    # 2 — average delay by hour (with peak bands)
    pdf = (df.filter(F.col("hour").isNotNull())
             .groupBy("hour").agg(F.avg("delay_minutes").alias("avg"))
             .orderBy("hour").toPandas())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(pdf["hour"], pdf["avg"], color="#f4c430", alpha=0.35)
    ax.plot(pdf["hour"], pdf["avg"], "-o", color="#e08e0b", ms=4)
    ax.axvspan(6, 9, color="red", alpha=0.12, label="Morning Rush")
    ax.axvspan(14, 17, color="orange", alpha=0.14, label="Evening Rush")
    ax.set_title("Average Delay by Hour"); ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Delay (minutes)"); ax.legend()
    _save(fig, "r_eda_02_delay_by_hour.png")

    # 3 — rush hour impact on delay
    pdf = (df.withColumn("rush", F.col("time_band").isin("AM_Peak", "PM_Peak"))
             .groupBy("rush").agg(F.avg("delay_minutes").alias("avg")).toPandas())
    pdf["label"] = pdf["rush"].map({True: "Rush Hour", False: "Non-Rush Hour"})
    pdf = pdf.sort_values("rush")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(pdf["label"], pdf["avg"], color=["#a9d3e0", "#f2764b"], edgecolor="gray")
    for b, v in zip(bars, pdf["avg"]):
        ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_title("Rush Hour Impact on Delay"); ax.set_ylabel("Average Delay (minutes)")
    _save(fig, "r_eda_03_rush_impact.png")

    # 4 — top 10 operators by incidents
    pdf = (df.groupBy("operator").agg(F.count("*").alias("n"))
             .orderBy(F.desc("n")).limit(10).toPandas())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(pdf["operator"][::-1], pdf["n"][::-1], color="#a45bb0")
    ax.set_title("Top 10 Operators by Incident Count"); ax.set_xlabel("Number of Incidents")
    _save(fig, "r_eda_04_top_operators.png")

    # 5 — average delay by run type
    pdf = (df.groupBy("run_type").agg(F.avg("delay_minutes").alias("avg"),
                                      F.count("*").alias("n"))
             .filter(F.col("n") > 1000).orderBy(F.desc("avg")).limit(8).toPandas())
    colors = ["#7fd0e0" if i % 2 else "#f2896b" for i in range(len(pdf))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(pdf["run_type"], pdf["avg"], color=colors, edgecolor="gray")
    ax.set_title("Average Delay by Run Type"); ax.set_ylabel("Average Delay (minutes)")
    ax.set_xticklabels(pdf["run_type"], rotation=25, ha="right", fontsize=8)
    _save(fig, "r_eda_05_delay_by_runtype.png")

    # 6 — distribution of students on bus
    pdf = (df.filter((F.col("students_on_bus") > 0) & (F.col("students_on_bus") < 60))
             .select("students_on_bus").sample(0.2, seed=3).toPandas())
    med = pdf["students_on_bus"].median()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(pdf["students_on_bus"], bins=30, color="#2aa198", edgecolor="white")
    ax.axvline(med, color="red", ls="--", lw=2, label=f"Median: {med:.0f}")
    ax.set_title("Distribution of Students On Board"); ax.set_xlabel("Students on Bus")
    ax.set_ylabel("Frequency"); ax.legend()
    _save(fig, "r_eda_06_students_dist.png")

    # 7 — average delay by day of week
    pdf = (df.filter(F.col("day_of_week").isNotNull())
             .groupBy("day_of_week").agg(F.avg("delay_minutes").alias("avg"))
             .orderBy("day_of_week").toPandas())
    pdf["dow"] = pdf["day_of_week"].map(DOW)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(pdf["dow"], pdf["avg"], "-o", color="#1e7d34", ms=7)
    ax.set_title("Average Delay by Day of Week"); ax.set_ylabel("Average Delay (minutes)")
    ax.set_xlabel("Day of Week")
    _save(fig, "r_eda_07_delay_by_dow.png")

    # 8 — average delay by time band
    order = ["AM_Peak", "Midday", "PM_Peak", "OffPeak"]
    pdf = (df.groupBy("time_band").agg(F.avg("delay_minutes").alias("avg")).toPandas())
    pdf = pdf.set_index("time_band").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(pdf["time_band"], pdf["avg"],
           color=["#ffd23f", "#ff9505", "#f26a4b", "#1d3f8a"], edgecolor="gray")
    ax.set_title("Average Delay by Time Band"); ax.set_ylabel("Average Delay (minutes)")
    ax.set_xlabel("Time Band")
    _save(fig, "r_eda_08_delay_by_timeband.png")

    # 9 — breakdown rate by borough
    pdf = (df.filter(F.col("boro").isNotNull())
             .groupBy("boro").agg(F.avg("is_breakdown").alias("rate"),
                                  F.count("*").alias("n"))
             .filter(F.col("n") > 2000).orderBy(F.desc("rate")).limit(10).toPandas())
    pdf = pdf.dropna(subset=["boro"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(pdf["boro"][::-1], pdf["rate"][::-1], color="#3182ce")
    ax.set_title("Breakdown Rate by Borough / Area"); ax.set_xlabel("Breakdown Rate")
    _save(fig, "r_eda_09_breakdown_by_boro.png")

    # combined 3x3 grid
    names = [f"r_eda_0{i}_" for i in range(1, 10)]
    files = sorted([p for p in FIGURES.glob("r_eda_0*.png")])
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(plt.imread(f)); ax.axis("off")
    fig.suptitle("Bus Incident Analysis — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold")
    _save(fig, "r_eda_grid.png")
    df.unpersist()


# ==================================================== MODEL COMPARISON ========
def comparison_suite():
    d = json.loads((OUTPUTS / "regression_metrics.json").read_text())
    r = pd.DataFrame(d["results"])
    order = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
    r = r.set_index("model").reindex(order).reset_index()

    def bar(col, title, ylabel, fname, better_high=False, thresh=None):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        bars = ax.bar(r["model"], r[col], color=PALETTE, edgecolor="gray")
        for b, v in zip(bars, r[col]):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        if thresh is not None:
            ax.axhline(thresh, ls="--", color="green", label=f"Good threshold ({thresh})")
            ax.legend()
        ax.set_title(title, fontweight="bold"); ax.set_ylabel(ylabel)
        ax.set_xticklabels(r["model"], rotation=15, ha="right", fontsize=8)
        _save(fig, fname)

    bar("r2", "R² Score Comparison (Higher is Better)", "R² Score",
        "r_cmp_01_r2.png", thresh=0.5)
    bar("mae", "MAE Comparison (Lower is Better)", "Mean Absolute Error (min)",
        "r_cmp_02_mae.png")
    bar("rmse", "RMSE Comparison (Lower is Better)", "Root Mean Squared Error",
        "r_cmp_03_rmse.png")
    bar("train_seconds", "Training Time Comparison", "Training Time (seconds)",
        "r_cmp_04_traintime.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [("r2", "R² Score (Higher Better)", 0.5), ("mae", "MAE (Lower Better)", None),
             ("rmse", "RMSE (Lower Better)", None), ("train_seconds", "Training Time (s)", None)]
    for ax, (col, title, th) in zip(axes.ravel(), specs):
        bars = ax.bar(r["model"], r[col], color=PALETTE, edgecolor="gray")
        for b, v in zip(bars, r[col]):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
        if th is not None:
            ax.axhline(th, ls="--", color="green")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticklabels(r["model"], rotation=18, ha="right", fontsize=7)
    fig.suptitle("Model Comparison — Bus Delay Prediction", fontsize=14, fontweight="bold")
    _save(fig, "r_cmp_grid.png")


# ==================================================== MODEL EVALUATION ========
def evaluation_suite():
    d = json.loads((OUTPUTS / "regression_metrics.json").read_text())
    best = [x for x in d["results"] if x["model"] == d["best_model"]][0]
    p = pd.read_csv(OUTPUTS / "regression_predictions.csv")
    imp = pd.DataFrame(d["feature_importances"])
    txt = f"R² = {best['r2']:.3f}\nMAE = {best['mae']:.2f} min\nRMSE = {best['rmse']:.2f} min"

    # actual vs predicted
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(p["actual"], p["prediction"], s=6, alpha=0.2, color="#3a6ea5")
    lim = [0, max(p["actual"].max(), p["prediction"].max()) + 5]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect Prediction")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
            bbox=dict(fc="#fff8b0", ec="none"), fontsize=9)
    ax.set_title("Actual vs Predicted Values", fontweight="bold")
    ax.set_xlabel("Actual Delay (min)"); ax.set_ylabel("Predicted Delay (min)")
    ax.legend(loc="lower right")
    _save(fig, "r_eval_01_actual_vs_pred.png")

    # residual plot
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(p["prediction"], p["residual"], s=6, alpha=0.2, color="#2e8b57")
    ax.axhline(0, color="red", ls="--", lw=2)
    ax.set_title(f"Residual Plot (Mean = {p['residual'].mean():.3f})", fontweight="bold")
    ax.set_xlabel("Predicted Delay (min)"); ax.set_ylabel("Residual (Actual − Predicted)")
    _save(fig, "r_eval_02_residual.png")

    # error distribution
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.hist(p["abs_error"], bins=40, color="#f2764b", edgecolor="white")
    ax.axvline(best["mae"], color="red", ls="--", lw=2, label=f"MAE = {best['mae']:.2f}")
    ax.set_title("Error Distribution", fontweight="bold")
    ax.set_xlabel("Absolute Error (minutes)"); ax.set_ylabel("Frequency"); ax.legend()
    _save(fig, "r_eval_03_error_dist.png")

    # feature importance
    fig, ax = plt.subplots(figsize=(7.5, 5))
    top = imp.head(10).iloc[::-1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top)))
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_title("Top 10 Feature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, "r_eval_04_feature_importance.png")

    # box plot of residuals
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    ax.boxplot(p["residual"], vert=True, patch_artist=True,
               boxprops=dict(facecolor="#a9d3e0"), medianprops=dict(color="red"))
    ax.axhline(0, color="blue", ls="--", alpha=0.5)
    ax.set_title("Error Distribution (Box Plot)", fontweight="bold")
    ax.set_ylabel("Residual (minutes)"); ax.set_xticklabels(["Residuals"])
    _save(fig, "r_eval_05_boxplot.png")

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    stats.probplot(p["residual"], dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", ms=4, alpha=0.5, color="#5b8fc9")
    ax.get_lines()[1].set(color="red", lw=2)
    ax.set_title("Q-Q Plot (Residuals Normality)", fontweight="bold")
    _save(fig, "r_eval_06_qq.png")

    files = [FIGURES / f for f in ["r_eval_01_actual_vs_pred.png", "r_eval_02_residual.png",
             "r_eval_03_error_dist.png", "r_eval_04_feature_importance.png",
             "r_eval_05_boxplot.png", "r_eval_06_qq.png"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, f in zip(axes.ravel(), files):
        ax.imshow(plt.imread(f)); ax.axis("off")
    fig.suptitle(f"{d['best_model']} Model — Comprehensive Evaluation",
                 fontsize=15, fontweight="bold")
    _save(fig, "r_eval_grid.png")


def main():
    spark = get_spark("ReportFigures")
    eda_suite(spark)
    spark.stop()
    comparison_suite()
    evaluation_suite()
    print("\nAll report figures generated.")


if __name__ == "__main__":
    main()
