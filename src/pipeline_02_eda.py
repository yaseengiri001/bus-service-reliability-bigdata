"""
Stage 2 — Exploratory Data Analysis & Visualisation  (PySpark + matplotlib)
===========================================================================
All statistics are computed with *PySpark* (describe, groupBy/agg, skewness,
kurtosis, approxQuantile, distributed Correlation). Data is converted to Pandas
ONLY at the final plotting step, exactly as the brief requires.

Outputs: numbered/captioned figures in outputs/figures + a JSON stats summary.

Run:  .venv/bin/python src/pipeline_02_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, FIGURES, OUTPUTS, timed, save_fig

sns.set_theme(style="whitegrid", palette="deep")
PLT_TITLE = {"fontsize": 12, "fontweight": "bold"}


def profile(df) -> dict:
    """Null counts, cardinality, basic stats, skewness & kurtosis (PySpark)."""
    n = df.count()
    prof: dict = {"n_rows": n, "null_counts": {}, "cardinality": {}}

    # null counts per column
    null_row = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).collect()[0].asDict()
    prof["null_counts"] = {k: int(v) for k, v in null_row.items()}

    # cardinality of key categoricals
    for c in ["operator", "boro", "reason", "run_type", "time_band",
              "season", "school_year", "incident_type"]:
        prof["cardinality"][c] = df.select(c).distinct().count()

    # distribution stats for delay_minutes & students_on_bus
    prof["stats"] = {}
    for c in ["delay_minutes", "students_on_bus", "hour"]:
        row = df.select(
            F.mean(c).alias("mean"), F.stddev(c).alias("std"),
            F.min(c).alias("min"), F.max(c).alias("max"),
            F.skewness(c).alias("skew"), F.kurtosis(c).alias("kurt"),
        ).collect()[0].asDict()
        med = df.approxQuantile(c, [0.25, 0.5, 0.75], 0.01)
        row.update({"q25": med[0], "median": med[1], "q75": med[2]})
        prof["stats"][c] = {k: (float(v) if v is not None else None)
                            for k, v in row.items()}

    # class balance for the ML targets
    prof["class_balance"] = {
        r["incident_type"]: int(r["cnt"])
        for r in df.groupBy("incident_type").agg(F.count("*").alias("cnt")).collect()
    }
    prof["breakdown_rate"] = float(df.agg(F.mean("is_breakdown")).collect()[0][0])
    prof["on_time_rate"] = float(df.agg(F.mean("on_time")).collect()[0][0])
    return prof


def outlier_report(df, col: str) -> dict:
    """IQR-based outlier detection using PySpark approxQuantile."""
    q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = df.filter((F.col(col) < lo) | (F.col(col) > hi)).count()
    return {"col": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower": lo, "upper": hi, "n_outliers": int(n_out)}


def make_figures(df):
    """One Spark aggregation per figure; toPandas only on the small result."""
    # Fig 1 — delay-minutes distribution
    pdf = (df.filter(F.col("delay_minutes").isNotNull())
             .select("delay_minutes").sample(0.15, seed=42).toPandas())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(pdf["delay_minutes"], bins=40, kde=True, color="#2b6cb0", ax=ax)
    ax.set_title("Figure 1. Distribution of Reported Delay (minutes)", **PLT_TITLE)
    ax.set_xlabel("Delay (minutes)"); ax.set_ylabel("Frequency (15% sample)")
    save_fig(fig, "fig01_delay_distribution.png"); plt.close(fig)

    # Fig 2 — incidents by reason
    pdf = (df.groupBy("reason").agg(F.count("*").alias("incidents"))
             .orderBy(F.desc("incidents")).limit(12).toPandas())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=pdf, y="reason", x="incidents", color="#dd6b20", ax=ax)
    ax.set_title("Figure 2. Incidents by Reported Reason", **PLT_TITLE)
    ax.set_xlabel("Number of incidents"); ax.set_ylabel("")
    save_fig(fig, "fig02_incidents_by_reason.png"); plt.close(fig)

    # Fig 3 — incidents by hour of day (peak pattern)
    pdf = (df.filter(F.col("hour").isNotNull())
             .groupBy("hour").agg(F.count("*").alias("incidents"))
             .orderBy("hour").toPandas())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=pdf, x="hour", y="incidents", marker="o", color="#2f855a", ax=ax)
    ax.axvspan(6, 9, alpha=0.12, color="red"); ax.axvspan(14, 17, alpha=0.12, color="red")
    ax.set_title("Figure 3. Incidents by Hour of Day (peak bands shaded)", **PLT_TITLE)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Number of incidents")
    save_fig(fig, "fig03_incidents_by_hour.png"); plt.close(fig)

    # Fig 4 — breakdown vs running-late by time band
    pdf = (df.groupBy("time_band", "incident_type").agg(F.count("*").alias("cnt"))
             .toPandas())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=pdf, x="time_band", y="cnt", hue="incident_type",
                order=["AM_Peak", "Midday", "PM_Peak", "OffPeak"], ax=ax)
    ax.set_title("Figure 4. Incident Type by Time Band", **PLT_TITLE)
    ax.set_xlabel("Time band"); ax.set_ylabel("Number of incidents")
    ax.legend(title="Incident type")
    save_fig(fig, "fig04_type_by_timeband.png"); plt.close(fig)

    # Fig 5 — top 15 operators by breakdown rate (min 500 incidents)
    pdf = (df.groupBy("operator")
             .agg(F.count("*").alias("incidents"),
                  F.mean("is_breakdown").alias("breakdown_rate"))
             .filter(F.col("incidents") >= 500)
             .orderBy(F.desc("breakdown_rate")).limit(15).toPandas())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=pdf, y="operator", x="breakdown_rate", color="#805ad5", ax=ax)
    ax.set_title("Figure 5. Top 15 Operators by Breakdown Rate (>=500 incidents)",
                 **PLT_TITLE)
    ax.set_xlabel("Breakdown rate"); ax.set_ylabel("")
    save_fig(fig, "fig05_operator_breakdown_rate.png"); plt.close(fig)

    # Fig 6 — incidents by borough
    pdf = (df.groupBy("boro").agg(F.count("*").alias("incidents"))
             .orderBy(F.desc("incidents")).limit(10).toPandas())
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=pdf, x="incidents", y="boro", color="#3182ce", ax=ax)
    ax.set_title("Figure 6. Incidents by Borough / Area", **PLT_TITLE)
    ax.set_xlabel("Number of incidents"); ax.set_ylabel("")
    save_fig(fig, "fig06_incidents_by_boro.png"); plt.close(fig)

    # Fig 7 — monthly trend across school years
    pdf = (df.filter(F.col("year").isNotNull())
             .groupBy("year", "month").agg(F.count("*").alias("incidents"))
             .toPandas())
    pdf["period"] = pdf["year"].astype(int).astype(str) + "-" + \
        pdf["month"].astype(int).astype(str).str.zfill(2)
    pdf = pdf.sort_values(["year", "month"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pdf["period"], pdf["incidents"], color="#c53030")
    ax.set_title("Figure 7. Monthly Incident Volume Over Time", **PLT_TITLE)
    ax.set_xlabel("Year-Month"); ax.set_ylabel("Incidents")
    step = max(1, len(pdf) // 12)
    ax.set_xticks(range(0, len(pdf), step))
    ax.set_xticklabels(pdf["period"].iloc[::step], rotation=45, ha="right", fontsize=7)
    save_fig(fig, "fig07_monthly_trend.png"); plt.close(fig)

    # Fig 8 — correlation heatmap (distributed Correlation, small matrix -> pandas)
    num_cols = ["delay_minutes", "students_on_bus", "hour", "month",
                "day_of_week", "is_weekend", "notified_schools",
                "notified_parents", "alerted_opt", "on_time", "is_breakdown"]
    cdf = df.select(*num_cols).na.fill(0)
    vec = VectorAssembler(inputCols=num_cols, outputCol="features").transform(cdf)
    corr = Correlation.corr(vec, "features").head()[0].toArray()
    import pandas as pd
    corr_pd = pd.DataFrame(corr, index=num_cols, columns=num_cols)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Figure 8. Correlation Matrix of Numeric Features (Pearson)",
                 **PLT_TITLE)
    save_fig(fig, "fig08_correlation_heatmap.png"); plt.close(fig)


def main():
    spark = get_spark("Stage2-EDA")
    df = spark.read.parquet(str(PROCESSED_PARQUET)).cache()
    print("[eda] loaded", df.count(), "rows")

    # PySpark SQL — a complex aggregation query (operator reliability league table)
    df.createOrReplaceTempView("incidents")
    print("\n=== PySpark SQL: operator reliability (top 10 by volume) ===")
    spark.sql("""
        SELECT operator,
               COUNT(*)                         AS incidents,
               ROUND(AVG(is_breakdown), 3)      AS breakdown_rate,
               ROUND(AVG(delay_minutes), 1)     AS avg_delay_min,
               ROUND(AVG(on_time), 3)           AS on_time_rate
        FROM incidents
        GROUP BY operator
        HAVING COUNT(*) >= 1000
        ORDER BY incidents DESC
        LIMIT 10
    """).show(truncate=False)

    with timed("eda_profiling"):
        prof = profile(df)
    prof["outliers"] = {c: outlier_report(df, c)
                        for c in ["delay_minutes", "students_on_bus"]}

    print("\n=== Data profile summary ===")
    print("rows:", prof["n_rows"])
    print("class balance:", prof["class_balance"])
    print("breakdown_rate: %.3f | on_time_rate: %.3f"
          % (prof["breakdown_rate"], prof["on_time_rate"]))
    print("delay skew/kurt: %.2f / %.2f"
          % (prof["stats"]["delay_minutes"]["skew"],
             prof["stats"]["delay_minutes"]["kurt"]))

    with timed("eda_figures"):
        make_figures(df)

    with open(OUTPUTS / "eda_summary.json", "w") as fh:
        json.dump(prof, fh, indent=2)
    print("\n[eda] wrote outputs/eda_summary.json and 8 figures")
    spark.stop()


if __name__ == "__main__":
    main()
