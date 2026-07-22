"""
Stage 5 — Regression: Delay-minutes prediction  (Spark MLlib)
=============================================================
Compares FOUR regressors (Linear, Decision Tree, Random Forest, GBT) — matching
the reference report's model set — to predict delay duration in minutes (the
brief's Travel-Time Variability metric).

Persists: metrics JSON (all 4 models), the best model, a test-prediction sample
(actual/predicted) and feature importances — all consumed by src/report_figures.py.

Run:  .venv/bin/python src/pipeline_05_ml_regression.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import (LinearRegression, DecisionTreeRegressor,
                                   RandomForestRegressor, GBTRegressor)
from pyspark.ml.evaluation import RegressionEvaluator

from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, OUTPUTS, MODELS

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
    return df.withColumn(
        "operator_bucketed",
        F.when(F.col("operator").isin(top_ops), F.col("operator")).otherwise("OTHER"))


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
         "train_seconds": round(train_s, 2)}
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
        "Linear Regression": LinearRegression(featuresCol="features", labelCol=LABEL,
                                              maxIter=50, regParam=0.1, elasticNetParam=0.1),
        "Decision Tree": DecisionTreeRegressor(featuresCol="features", labelCol=LABEL,
                                               maxDepth=10, seed=42),
        "Random Forest": RandomForestRegressor(featuresCol="features", labelCol=LABEL,
                                               numTrees=80, maxDepth=10, seed=42),
        "Gradient Boosting": GBTRegressor(featuresCol="features", labelCol=LABEL,
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
    print(f"[best] {best_name}")

    # save a test-prediction sample for evaluation figures
    pred_pd = (best_preds.select(F.col(LABEL).alias("actual"), "prediction")
                         .sample(0.05, seed=1).limit(20000).toPandas())
    pred_pd["residual"] = pred_pd["actual"] - pred_pd["prediction"]
    pred_pd["abs_error"] = pred_pd["residual"].abs()
    pred_pd.to_csv(OUTPUTS / "regression_predictions.csv", index=False)
    print(f"[save] {len(pred_pd)} test predictions -> regression_predictions.csv")

    # feature importances (best tree model)
    importances = []
    if hasattr(best_model, "featureImportances"):
        names = feature_names(te, "features")
        imp = best_model.featureImportances.toArray()
        importances = sorted(
            [{"feature": n, "importance": float(v)} for n, v in zip(names, imp)],
            key=lambda x: -x["importance"])[:12]

    out = {"task": "regression_delay_minutes", "n_rows": df.count(),
           "results": results, "best_model": best_name,
           "feature_importances": importances}
    with open(OUTPUTS / "regression_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    best_model.write().overwrite().save(str(MODELS / f"reg_{best_name.replace(' ', '_')}"))
    print(f"[done] metrics + importances -> regression_metrics.json")
    spark.stop()


if __name__ == "__main__":
    main()
