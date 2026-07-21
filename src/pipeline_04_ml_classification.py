"""
Stage 4 — Classification: a-priori Breakdown-Risk model  (Spark MLlib)
=====================================================================
Business question: given only information known *before* an incident is
resolved (operator, borough, run type, time-of-day, load), how likely is a
reported incident to be a **Breakdown** (severe service failure) rather than
merely **Running Late**?  This is a non-compliance / severity classifier for a
Transport-Authority stakeholder.

Design decisions
----------------
* ``reason`` and ``delay_minutes`` are EXCLUDED from features to avoid target
  leakage (both are only fully known after the event / describe the outcome).
* Class imbalance (~9.6% breakdowns) is handled with an inverse-frequency
  ``classWeight`` column, and evaluation leads with F1 / Recall / ROC-AUC / PR-AUC
  rather than accuracy.
* Three models compared: LogisticRegression, RandomForest, GBTClassifier.
* A 3-fold CrossValidator tunes the LogisticRegression to document the tuning
  pipeline (VectorAssembler -> StringIndexer -> OneHot -> Scaler -> estimator).

Run:  .venv/bin/python src/pipeline_04_ml_classification.py
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
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier,
                                       GBTClassifier)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, FIGURES, OUTPUTS, MODELS, save_fig

sns.set_theme(style="whitegrid")

CAT_COLS = ["operator_bucketed", "boro", "run_type", "time_band", "season", "school_age"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend", "students_on_bus", "year"]
LABEL = "is_breakdown"
TOP_N_OPERATORS = 25


def prepare(df):
    """Bucket high-cardinality operator + add inverse-frequency class weights."""
    top_ops = [r["operator"] for r in
               (df.groupBy("operator").count()
                  .orderBy(F.desc("count")).limit(TOP_N_OPERATORS).collect())]
    df = df.withColumn(
        "operator_bucketed",
        F.when(F.col("operator").isin(top_ops), F.col("operator")).otherwise("OTHER"),
    )
    # inverse-frequency weights: balance the two classes
    n = df.count()
    n_pos = df.filter(F.col(LABEL) == 1).count()
    n_neg = n - n_pos
    w_pos, w_neg = n / (2.0 * n_pos), n / (2.0 * n_neg)
    df = df.withColumn(
        "classWeight",
        F.when(F.col(LABEL) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)),
    )
    print(f"[prep] n={n:,} pos={n_pos:,} ({100*n_pos/n:.1f}%) "
          f"w_pos={w_pos:.3f} w_neg={w_neg:.3f}")
    return df


def feature_pipeline() -> Pipeline:
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in CAT_COLS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh")
                for c in CAT_COLS]
    assembler = VectorAssembler(
        inputCols=[f"{c}_oh" for c in CAT_COLS] + NUM_COLS,
        outputCol="features_raw", handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=False, withStd=True)
    return Pipeline(stages=indexers + encoders + [assembler, scaler])


def evaluate(preds, name: str) -> dict:
    """Full metric suite for the positive (Breakdown) class."""
    b_roc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction",
                                          metricName="areaUnderROC")
    b_pr = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction",
                                         metricName="areaUnderPR")
    def mc(metric, label=None):
        e = MulticlassClassificationEvaluator(labelCol=LABEL, predictionCol="prediction",
                                              metricName=metric)
        if label is not None:
            e = e.setMetricLabel(label)
        return e.evaluate(preds)
    m = {
        "model": name,
        "accuracy": mc("accuracy"),
        "precision_breakdown": mc("precisionByLabel", 1.0),
        "recall_breakdown": mc("recallByLabel", 1.0),
        "f1_breakdown": mc("fMeasureByLabel", 1.0),
        "f1_weighted": mc("f1"),
        "roc_auc": b_roc.evaluate(preds),
        "pr_auc": b_pr.evaluate(preds),
    }
    print(f"[eval] {name:<20} ACC={m['accuracy']:.3f}  "
          f"P={m['precision_breakdown']:.3f}  R={m['recall_breakdown']:.3f}  "
          f"F1={m['f1_breakdown']:.3f}  ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}")
    return m


def roc_points(preds):
    """Collect (label, p1) sample and compute ROC via numpy (no sklearn dep needed)."""
    pdf = (preds.withColumn("p1", vector_to_array("probability")[1])
                .select(LABEL, "p1").sample(0.08, seed=7).toPandas())
    y = pdf[LABEL].values
    s = pdf["p1"].values
    order = np.argsort(-s)
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tps / max(P, 1)])
    fpr = np.concatenate([[0], fps / max(N, 1)])
    return fpr, tpr


def main():
    spark = get_spark("Stage4-Classification")
    print("Spark UI:", spark.sparkContext.uiWebUrl)
    df = prepare(spark.read.parquet(str(PROCESSED_PARQUET)).cache())

    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train = train.cache(); test = test.cache()
    print(f"[split] train={train.count():,} test={test.count():,}")

    # fit shared feature pipeline on train, reuse for all models
    fp = feature_pipeline().fit(train)
    tr = fp.transform(train).cache()
    te = fp.transform(test).cache()

    models = {
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol=LABEL, weightCol="classWeight", maxIter=50),
        "RandomForest": RandomForestClassifier(
            featuresCol="features", labelCol=LABEL, weightCol="classWeight",
            numTrees=80, maxDepth=10, seed=42),
        "GBTClassifier": GBTClassifier(
            featuresCol="features", labelCol=LABEL, weightCol="classWeight",
            maxIter=30, maxDepth=5, seed=42),
    }

    results, timings, fitted = [], {}, {}
    for name, est in models.items():
        t0 = time.perf_counter()
        model = est.fit(tr)
        train_s = time.perf_counter() - t0
        timings[name] = train_s
        preds = model.transform(te)
        m = evaluate(preds, name)
        m["train_seconds"] = round(train_s, 2)
        m["model_efficiency_f1_per_s"] = round(m["f1_breakdown"] / train_s, 4)
        results.append(m)
        fitted[name] = (model, preds)
        print(f"[time] {name} trained in {train_s:.1f}s")

    # ---- CrossValidator to document tuning (LogisticRegression) -------------
    print("\n[cv] 3-fold CrossValidator tuning LogisticRegression ...")
    lr = LogisticRegression(featuresCol="features", labelCol=LABEL,
                            weightCol="classWeight", maxIter=50)
    grid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.0, 0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5])
            .build())
    cv = CrossValidator(
        estimator=lr, estimatorParamMaps=grid,
        evaluator=BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC"),
        numFolds=3, parallelism=2, seed=42)
    cv_model = cv.fit(tr)
    best = cv_model.bestModel
    cv_summary = {
        "best_regParam": float(best.getRegParam()),
        "best_elasticNetParam": float(best.getElasticNetParam()),
        "best_cv_roc_auc": float(max(cv_model.avgMetrics)),
        "all_cv_roc_auc": [round(float(x), 4) for x in cv_model.avgMetrics],
    }
    print(f"[cv] best regParam={cv_summary['best_regParam']} "
          f"elasticNet={cv_summary['best_elasticNetParam']} "
          f"CV ROC-AUC={cv_summary['best_cv_roc_auc']:.3f}")

    # ---- figures ------------------------------------------------------------
    # Fig 9 — ROC curves for all three models
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (_, preds) in fitted.items():
        fpr, tpr = roc_points(preds)
        auc = next(r["roc_auc"] for r in results if r["model"] == name)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("Figure 9. ROC Curves — Breakdown Classification", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    save_fig(fig, "fig09_roc_curves.png"); plt.close(fig)

    # Fig 10 — confusion matrix for the best model (by F1)
    best_name = max(results, key=lambda r: r["f1_breakdown"])["model"]
    _, best_preds = fitted[best_name]
    cm = (best_preds.groupBy(LABEL, "prediction").count().toPandas()
          .pivot(index=LABEL, columns="prediction", values="count").fillna(0))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title(f"Figure 10. Confusion Matrix — {best_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    save_fig(fig, "fig10_confusion_matrix.png"); plt.close(fig)

    # Fig 11 — model comparison bar chart
    import pandas as pd
    rdf = pd.DataFrame(results).set_index("model")[
        ["f1_breakdown", "recall_breakdown", "roc_auc", "pr_auc"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rdf.plot(kind="bar", ax=ax)
    ax.set_title("Figure 11. Classification Model Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.set_xticklabels(rdf.index, rotation=12); ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, "fig11_model_comparison.png"); plt.close(fig)

    # ---- persist metrics + best model --------------------------------------
    out = {"task": "classification_is_breakdown", "label_balance_positive": 0.096,
           "results": results, "cross_validation": cv_summary, "best_model": best_name}
    with open(OUTPUTS / "classification_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    fitted[best_name][0].write().overwrite().save(str(MODELS / f"cls_{best_name}"))
    print(f"\n[done] best classifier = {best_name}; metrics -> classification_metrics.json")
    spark.stop()


if __name__ == "__main__":
    main()
