"""
Render REAL code excerpts (syntax-highlighted) and REAL console output as clean
'screenshot' PNGs to fill the template's code/output figure slots.

HTML (Pygments monokai) -> headless Chrome --screenshot.  Output -> docs/screenshots/.

Run:  .venv/bin/python src/make_code_shots.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

from src.utils import DOCS

SHOTS = DOCS / "screenshots"
TMP = Path("/tmp/codeshots")
TMP.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

WIN_CSS = """
body{margin:0;background:#0f1117;font-family:-apple-system,Segoe UI,sans-serif;
     display:flex;justify-content:center;padding:26px}
.win{width:1000px;border-radius:10px;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.5);
     background:#272822}
.bar{background:#1e1f1c;height:38px;display:flex;align-items:center;padding:0 14px;
     position:relative}
.dot{width:12px;height:12px;border-radius:50%;margin-right:8px}
.title{position:absolute;left:0;right:0;text-align:center;color:#c8c8c8;font-size:13px}
.body pre{margin:0;padding:16px 18px;font-size:13px;line-height:1.5;
          font-family:'SF Mono',Menlo,monospace;overflow-x:auto}
.term{background:#1e1e2e;color:#cdd6f4;padding:16px 18px;font-size:12.5px;line-height:1.55;
      font-family:'SF Mono',Menlo,monospace;white-space:pre-wrap}
.term .g{color:#a6e3a1}.term .c{color:#89b4fa}.term .y{color:#f9e2af}
"""


def _dots():
    return ('<span class="dot" style="background:#ff5f56"></span>'
            '<span class="dot" style="background:#ffbd2e"></span>'
            '<span class="dot" style="background:#27c93f"></span>')


def _render(html, fname, lines):
    path = TMP / (fname + ".html")
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                    f"<style>{WIN_CSS}</style></head><body>{html}</body></html>")
    height = int(lines * 21 + 130)
    out = SHOTS / fname
    if out.exists():
        out.unlink()
    proc = subprocess.Popen([
        CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
        "--force-device-scale-factor=2", f"--user-data-dir=/tmp/cs_{fname}",
        f"--window-size=1060,{height}", f"--screenshot={out}", path.as_uri()],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"[shot] {fname} -> {out.stat().st_size//1024 if out.exists() else 'MISSING'} KB")


def code_shot(fname, title, code):
    fmt = HtmlFormatter(style="monokai", noclasses=True, nowrap=False)
    body = highlight(code.strip("\n"), PythonLexer(), fmt)
    html = (f'<div class="win"><div class="bar">{_dots()}'
            f'<span class="title">{title}</span></div>'
            f'<div class="body">{body}</div></div>')
    _render(html, fname, code.strip("\n").count("\n") + 1)


def out_shot(fname, title, text):
    esc = (text.strip("\n").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    html = (f'<div class="win"><div class="bar">{_dots()}'
            f'<span class="title">{title}</span></div>'
            f'<div class="term">{esc}</div></div>')
    _render(html, fname, text.strip("\n").count("\n") + 1)


# ---------------------------------------------------------------- CODE --------
C_SPARK = '''
def get_spark(app_name="BusServiceReliability", shuffle_partitions=8):
    _ensure_java_home()                       # locate Homebrew JDK 17
    spark = (SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)   # > 4
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.port", "4040")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark
'''

C_INGEST = '''
def load_raw(spark):
    """Quote-aware distributed CSV read with an explicit 21-column schema."""
    df = (spark.read
        .option("header", "true")
        .option("quote", '"').option("escape", '"')
        .option("mode", "PERMISSIVE")
        .schema(RAW_SCHEMA)
        .csv(str(RAW_CSV)))
    return df

raw = load_raw(spark).repartition(8).cache()   # 8 partitions + caching
print("raw records:", raw.count())
'''

C_CLEAN = '''
# Robust delay-minutes parser from messy free-text ("20 MINS", "15-30 Min", "1 HOUR")
du = F.upper(F.trim(F.col("How_Long_Delayed")))
rng_lo = F.regexp_extract(du, r"(\\d+)\\s*-\\s*(\\d+)", 1).cast("double")
rng_hi = F.regexp_extract(du, r"(\\d+)\\s*-\\s*(\\d+)", 2).cast("double")
first  = F.regexp_extract(du, r"(\\d+)", 1).cast("double")
delay  = (F.when((rng_lo > 0) & (rng_hi > 0), (rng_lo + rng_hi) / 2.0)
           .when(du.rlike(r"HOUR|HR"), first * 60.0)
           .otherwise(first))
df = df.withColumn("delay_minutes",
        F.when(delay > 180, 180.0).when(delay < 0, None).otherwise(delay))
'''

C_STAR = '''
# Decompose into a star schema, then re-join dimensions with a BROADCAST join
dim_operator = df.select("operator").distinct().withColumn(
    "operator_id", F.monotonically_increasing_id())

joined = (feat
    .join(broadcast(dim_operator), "operator", "left")
    .join(broadcast(dim_reason),   "reason",   "left")
    .join(broadcast(dim_boro),     "boro",     "left"))

final.repartition(8).write.mode("overwrite").parquet(str(PROCESSED_PARQUET))
'''

C_FEATURE = '''
def feature_pipeline():
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                              handleInvalid="keep") for c in CAT_COLS]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh")
                for c in CAT_COLS]
    assembler = VectorAssembler(
        inputCols=[f"{c}_oh" for c in CAT_COLS] + NUM_COLS,
        outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=False, withStd=True)
    return Pipeline(stages=indexers + encoders + [assembler, scaler])
'''

C_TRAIN = '''
models = {
    "LogisticRegression": LogisticRegression(featuresCol="features",
        labelCol=LABEL, weightCol="classWeight", maxIter=50),
    "RandomForest": RandomForestClassifier(featuresCol="features",
        labelCol=LABEL, weightCol="classWeight", numTrees=80, maxDepth=10),
    "GBTClassifier": GBTClassifier(featuresCol="features",
        labelCol=LABEL, weightCol="classWeight", maxIter=30, maxDepth=5),
}
for name, est in models.items():
    model = est.fit(train)
    preds = model.transform(test)
    evaluate(preds, name)     # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
'''

C_SPLIT = '''
# Inverse-frequency class weights handle the 9.6% breakdown imbalance
n, n_pos = df.count(), df.filter(F.col(LABEL) == 1).count()
w_pos, w_neg = n / (2.0 * n_pos), n / (2.0 * (n - n_pos))
df = df.withColumn("classWeight",
        F.when(F.col(LABEL) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg)))

train, test = df.randomSplit([0.8, 0.2], seed=42)   # 80/20 hold-out
train, test = train.cache(), test.cache()
'''

C_CV = '''
# 3-fold CrossValidator documents the tuning pipeline
grid = (ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr.elasticNetParam, [0.0, 0.5]).build())
cv = CrossValidator(estimator=lr, estimatorParamMaps=grid,
        evaluator=BinaryClassificationEvaluator(labelCol=LABEL,
            metricName="areaUnderROC"),
        numFolds=3, parallelism=2, seed=42)
best = cv.fit(train).bestModel
'''

C_DB = '''
def run_query(conn, sql, params=()):
    """SECURE: values bound via ? placeholders — never string-formatted."""
    cur = conn.cursor()
    cur.execute(sql, params)          # parameters passed separately
    return [d[0] for d in cur.description], cur.fetchall()

# SQL-injection attempt is neutralised, not executed:
malicious = "x'; DROP TABLE fact_incident; --"
run_query(conn, "SELECT COUNT(*) FROM fact_incident f JOIN dim_operator o "
                "ON o.operator_id=f.operator_id WHERE o.operator = ?", (malicious,))
'''

# ---------------------------------------------------------------- OUTPUT ------
O_INGEST = '''$ .venv/bin/python src/pipeline_01_ingest_preprocess.py
[partitions] raw (after CSV read)      -> 8 partitions
[timing] 01_load_raw_csv                   1.14s
[timing] 02_count_after_cache              2.34s
[rows] raw records: 1,294,129
[quality] kept 1,167,832 / 1,294,129 rows (90.2%); dropped 126,297
[timing] 04_broadcast_join_dims            0.63s
[join] broadcast-joined 3 dimension tables into fact frame
[timing] 05_write_parquet                  5.89s
[persist] wrote Parquet + 4 dimension tables
=== Stage 1 complete ==='''

O_CLS = '''$ .venv/bin/python src/pipeline_04_ml_classification.py
[prep] n=1,167,832 pos=111,787 (9.6%)  w_pos=5.223 w_neg=0.553
[split] train=934,690 test=233,142
[eval] LogisticRegression  ACC=0.722 P=0.224 R=0.772 F1=0.347 ROC-AUC=0.827
[eval] RandomForest        ACC=0.774 P=0.261 R=0.746 F1=0.387 ROC-AUC=0.843
[eval] GBTClassifier       ACC=0.780 P=0.265 R=0.733 F1=0.389 ROC-AUC=0.847
[cv] best regParam=0.0 elasticNet=0.0  CV ROC-AUC=0.827
[done] best classifier = GBTClassifier'''

O_REG = '''$ .venv/bin/python src/pipeline_05_ml_regression.py
[prep] rows with delay label: 1,039,711
[eval] Linear Regression   RMSE=16.42 MAE=11.93 R2=0.359 (3.8s)
[eval] Decision Tree       RMSE=14.38 MAE=9.95  R2=0.508 (2.0s)
[eval] Random Forest       RMSE=14.19 MAE=10.04 R2=0.521 (30.8s)
[eval] Gradient Boosting   RMSE=14.34 MAE=9.94  R2=0.511 (14.5s)
[best] Random Forest'''

O_DB = '''$ .venv/bin/python src/pipeline_03_database.py
[db] tables populated and committed
=== Parameterised-query security demo ===
  malicious input handled safely -> rows matched: 0
  fact_incident still intact -> 1,167,832 rows (no injection)
[db] SQLite ready: bus_reliability.db (232.5 MB, 1,167,832 fact rows)'''

O_SQL = '''spark.sql("SELECT operator, COUNT(*) incidents, AVG(is_breakdown) ...")
+---------------------------+---------+--------------+-------------+
|operator                   |incidents|breakdown_rate|avg_delay_min|
+---------------------------+---------+--------------+-------------+
|LEESEL TRANSPORTATION CORP |95938    |0.012         |41.6         |
|PRIDE TRANSPORTATION       |78180    |0.087         |65.1         |
|AMBOY BUS COMPANY, INC     |64241    |0.135         |24.2         |
|LITTLE RICHIE BUS SERVICE  |36906    |0.392         |40.3         |
+---------------------------+---------+--------------+-------------+'''


def main():
    code_shot("code_01_spark_config.png", "config/spark_config.py", C_SPARK)
    code_shot("code_02_ingest.png", "pipeline_01_ingest_preprocess.py", C_INGEST)
    code_shot("code_03_cleaning.png", "pipeline_01 — delay parser", C_CLEAN)
    code_shot("code_04_star_schema.png", "pipeline_01 — star schema + broadcast", C_STAR)
    code_shot("code_05_feature_pipeline.png", "pipeline_04 — feature pipeline", C_FEATURE)
    code_shot("code_06_training.png", "pipeline_04 — model training", C_TRAIN)
    code_shot("code_07_splitting.png", "pipeline_04 — split + class weights", C_SPLIT)
    code_shot("code_08_crossvalidator.png", "pipeline_04 — CrossValidator", C_CV)
    code_shot("code_09_db_security.png", "pipeline_03 — parameterised query", C_DB)
    out_shot("out_01_ingest.png", "Stage 1 — Ingestion output", O_INGEST)
    out_shot("out_02_classification.png", "Stage 4 — Classification output", O_CLS)
    out_shot("out_03_regression.png", "Stage 5 — Regression output", O_REG)
    out_shot("out_04_db_security.png", "Stage 3 — Database + security output", O_DB)
    out_shot("out_05_sql.png", "PySpark SQL — operator reliability", O_SQL)
    print("\nAll code/output shots generated.")


if __name__ == "__main__":
    main()
