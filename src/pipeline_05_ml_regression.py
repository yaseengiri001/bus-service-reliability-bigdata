"""
Stage 5 — Regression: Delay-minutes prediction  (Spark MLlib)
=============================================================
Business question: how many minutes will a reported delay last?  Target maps to
the brief's *Travel Time Variability* reliability metric. Here ``reason`` IS a
valid predictor (it is known at the moment a delay is logged).

Three models compared: LinearRegression, RandomForestRegressor, GBTRegressor.
Metrics: RMSE, MAE, R2  (+ training time -> model-efficiency).

Run:  .venv/bin/python src/pipeline_05_ml_regression.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import (LinearRegression, RandomForestRegressor,
                                   GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator

from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, OUTPUTS, MODELS, save_fig

sns.set_theme(style="whitegrid")

CAT_COLS = ["operator_bucketed", "boro", "run_type", "reason", "time_band",
            "season", "school_age"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend", "students_on_bus",
            "year", "is_breakdown"]
LABEL = "delay_minutes"
TOP_N_OPERATORS = 25


def prepare(df):
    df = df.filter(F.col(LABEL).isNotNull())
    top_ops = [r["operator"] for r in
               (df.groupBy("operator").count()
                  .orderBy(F.desc("count")).limit(TOP_N_OPERATORS).collect())]
    df = df.withColumn(
        "operator_bucketed",
        F.when(F.col("operator").isin(top_ops), F.col("operator")).otherwise("OTHER"))
    return df


def feature_pipeline() -> Pipeline:
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
           for c in CAT_COLS]
    enc = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh") for c in CAT_COLS]
    asm = VectorAssembler(inputCols=[f"{c}_oh" for c in CAT_COLS] + NUM_COLS,
                          outputCol="features", handleInvalid="keep")
    return Pipeline(stages=idx + enc + [asm])


def feature_names(df, col="features"):
    attrs = df.schema[col].metadata.get("ml_attr", {}).get("attrs", {})
    names = {}
    for group in attrs.values():
        for a in group:
            names[a["idx"]] = a["name"]
    return [names.get(i, f"f{i}") for i in range(len(names))]


def evaluate(preds, name, train_s) -> dict:
    def ev(metric):
        return RegressionEvaluator(labelCol=LABEL, predictionCol="prediction",
                                   metricName=metric).evaluate(preds)
    rmse, mae, r2 = ev("rmse"), ev("mae"), ev("r2")
    m = {"model": name, "rmse": rmse, "mae": mae, "r2": r2,
         "train_seconds": round(train_s, 2),
         "model_efficiency_rmse_per_s": round(rmse / train_s, 3)}
    print(f"[eval] {name:<24} RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  ({train_s:.1f}s)")
    return m


def main():
    spark = get_spark("Stage5-Regression")
    print("Spark UI:", spark.sparkContext.uiWebUrl)
    df = prepare(spark.read.parquet(str(PROCESSED_PARQUET))).cache()
    print(f"[prep] rows with delay label: {df.count():,}")

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    fp = feature_pipeline().fit(train)
    tr = fp.transform(train).cache()
    te = fp.transform(test).cache()

    models = {
        "LinearRegression": LinearRegression(featuresCol="features", labelCol=LABEL,
                                             maxIter=50, regParam=0.1, elasticNetParam=0.1),
        "RandomForestRegressor": RandomForestRegressor(featuresCol="features", labelCol=LABEL,
                                                       numTrees=80, maxDepth=10, seed=42),
        "GBTRegressor": GBTRegressor(featuresCol="features", labelCol=LABEL,
                                     maxIter=40, maxDepth=5, seed=42),
    }

    results, fitted = [], {}
    for name, est in models.items():
        t0 = time.perf_counter()
        model = est.fit(tr)
        dt = time.perf_counter() - t0
        preds = model.transform(te)
        results.append(evaluate(preds, name, dt))
        fitted[name] = (model, preds)

    best_name = min(results, key=lambda r: r["rmse"])["model"]
    best_model, best_preds = fitted[best_name]

    # Fig 12 — predicted vs actual (best model, sample)
    pdf = best_preds.select(LABEL, "prediction").sample(0.03, seed=1).toPandas()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pdf[LABEL], pdf["prediction"], s=4, alpha=0.15, color="#2b6cb0")
    lim = [0, 185]
    ax.plot(lim, lim, "r--", alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Figure 12. Predicted vs Actual Delay — {best_name}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual delay (min)"); ax.set_ylabel("Predicted delay (min)")
    save_fig(fig, "fig12_pred_vs_actual.png"); plt.close(fig)

    # Fig 13 — residual distribution
    pdf["residual"] = pdf[LABEL] - pdf["prediction"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(pdf["residual"], bins=50, kde=True, color="#dd6b20", ax=ax)
    ax.axvline(0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Figure 13. Residual Distribution — {best_name}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual (actual - predicted) min")
    save_fig(fig, "fig13_residuals.png"); plt.close(fig)

    # Fig 14 — model comparison (RMSE / MAE)
    rdf = pd.DataFrame(results).set_index("model")[["rmse", "mae"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rdf.plot(kind="bar", ax=ax, color=["#3182ce", "#e53e3e"])
    ax.set_title("Figure 14. Regression Model Comparison (lower is better)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Minutes"); ax.set_xlabel("")
    ax.set_xticklabels(rdf.index, rotation=12, fontsize=8)
    save_fig(fig, "fig14_regression_comparison.png"); plt.close(fig)

    # Fig 15 — feature importance (best tree model)
    if hasattr(best_model, "featureImportances"):
        names = feature_names(te, "features")
        imp = best_model.featureImportances.toArray()
        top = pd.DataFrame({"feature": names, "importance": imp}) \
            .sort_values("importance", ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sns.barplot(data=top, y="feature", x="importance", color="#2f855a", ax=ax)
        ax.set_title(f"Figure 15. Top-15 Feature Importances — {best_name}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance"); ax.set_ylabel("")
        save_fig(fig, "fig15_feature_importance.png"); plt.close(fig)

    out = {"task": "regression_delay_minutes", "n_rows": df.count(),
           "results": results, "best_model": best_name}
    with open(OUTPUTS / "regression_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    best_model.write().overwrite().save(str(MODELS / f"reg_{best_name}"))
    print(f"\n[done] best regressor = {best_name}; metrics -> regression_metrics.json")
    spark.stop()


if __name__ == "__main__":
    main()
