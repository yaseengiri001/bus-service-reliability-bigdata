# Bus Service Reliability — Big Data Predictive Analytics Platform

A distributed **PySpark** pipeline that ingests **1.29 million** real NYC bus
breakdown/delay records, engineers features at scale, persists them to a
relational **star schema**, and trains **six machine-learning models** (three
classification + three regression) to predict service disruptions — with an
interactive **Streamlit** dashboard for a Transport-Authority stakeholder.

> **Module:** Big Data Programming Project · **Stack:** Apache Spark 3.5.3 ·
> Spark MLlib · SQLite · Streamlit · Python 3.9 · Java 17

---

## 1. Business problem

Public-transport authorities need to know **which trips are at risk of a severe
breakdown** (versus a minor delay) and **how long delays will last**, so they
can pre-position response resources and hold operators accountable. This project
answers both questions from open incident data:

| Task | Type | Target | Reliability metric (brief §4) |
|------|------|--------|-------------------------------|
| Breakdown vs Running-Late | Classification | `is_breakdown` | Service Reliability / Non-compliance |
| Delay duration | Regression | `delay_minutes` | Travel-Time Variability |
| Operator benchmarking | Aggregation/SQL | breakdown-rate | Service Efficiency |

## 2. Dataset

* **Source (primary):** [NYC Open Data — *Bus Breakdown and Delays*](https://data.cityofnewyork.us/Transportation/Bus-Breakdown-and-Delays/ez4e-fazm) (dataset id `ez4e-fazm`), **CC0 Public Domain**.
* **Kaggle mirror:** [`mattop/new-york-city-bus-breakdown-and-delays`](https://www.kaggle.com/datasets/mattop/new-york-city-bus-breakdown-and-delays) (2015–2022 snapshot).
* **Scale:** 1,294,129 raw rows · 21 columns · ~295 MB CSV → **1,167,832** cleaned rows (well beyond the 100k big-data threshold).

Download the raw CSV (no login required):

```bash
mkdir -p data/raw
curl --compressed -L -o data/raw/bus_breakdown_delays.csv \
  "https://data.cityofnewyork.us/api/views/ez4e-fazm/rows.csv?accessType=DOWNLOAD"
```

## 3. Architecture

![Architecture](docs/diagrams/architecture.png)

`Ingestion (PySpark)` → `Processing (clean · feature-eng · star schema)` →
`Storage (Parquet + SQLite)` → `Modelling (MLlib: LR/RF/GBT)` →
`Serving (matplotlib EDA + Streamlit)`.

## 4. Repository structure

```
├── config/spark_config.py            # central, auditable SparkSession factory
├── src/
│   ├── pipeline_01_ingest_preprocess.py   # CSV → clean → features → star schema → Parquet
│   ├── pipeline_02_eda.py                 # PySpark stats/profiling + 8 figures
│   ├── pipeline_03_database.py            # SQLite star schema + parameterised queries
│   ├── pipeline_04_ml_classification.py   # 3 classifiers + CrossValidator
│   ├── pipeline_05_ml_regression.py       # 3 regressors
│   ├── spark_ui_demo.py                   # generates Spark-UI evidence
│   ├── make_report_assets.py              # diagrams + exec cards
│   └── utils.py
├── dashboard/app.py                  # Streamlit dashboard
├── database/                         # schema.sql · sample_queries.sql · sample dump
├── data/processed/                   # Parquet + dimension tables (+ sample CSV)
├── outputs/                          # figures/, models/, *_metrics.json
├── docs/                             # report, diagrams, screenshots
├── requirements.txt
└── README.md
```

## 5. Setup

**Prerequisites:** Python 3.9+, **Java 17** (`brew install openjdk@17` on macOS), Git.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

`config/spark_config.py` auto-locates `JAVA_HOME` and configures 8 shuffle
partitions, Arrow, AQE and the Java-17 module flags — no manual env setup needed.

## 6. Run the pipeline (in order)

```bash
.venv/bin/python src/pipeline_01_ingest_preprocess.py   # → data/processed/*.parquet
.venv/bin/python src/pipeline_02_eda.py                 # → outputs/figures/fig01-08
.venv/bin/python src/pipeline_03_database.py            # → database/bus_reliability.db
.venv/bin/python src/pipeline_04_ml_classification.py   # → classification_metrics.json
.venv/bin/python src/pipeline_05_ml_regression.py       # → regression_metrics.json
.venv/bin/streamlit run dashboard/app.py                # → http://localhost:8501
```

Spark UI is available at **http://localhost:4040** while any stage runs
(run `src/spark_ui_demo.py` to hold it open).

## 7. Results (held-out test set)

**Classification — predict Breakdown vs Running-Late** (class-weighted; 9.6 % positives)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:------:|
| LogisticRegression | 0.722 | 0.224 | 0.772 | 0.347 | 0.827 | 0.347 |
| RandomForest | 0.774 | 0.261 | 0.746 | 0.387 | 0.843 | 0.391 |
| **GBTClassifier** | **0.780** | **0.265** | 0.733 | **0.389** | **0.847** | **0.397** |

**Regression — predict delay minutes** (RMSE lower = better)

| Model | RMSE | MAE | R² |
|-------|:----:|:---:|:--:|
| LinearRegression | 16.42 | 11.93 | 0.359 |
| **RandomForestRegressor** | **14.19** | 10.04 | **0.521** |
| GBTRegressor | 14.34 | **9.94** | 0.511 |

## 8. Security & data handling

* All database access uses **parameterised queries** (`?` placeholders) — a
  built-in injection test (`x'; DROP TABLE fact_incident; --`) is neutralised.
* No credentials are hard-coded; all configuration lives in `config/`.
* Data is **CC0 Public Domain**; it contains **no personal data** (vehicle/route
  codes only), aligning with GDPR data-minimisation principles.

## 9. Reproducibility notes

* Random seeds fixed (`seed=42`) for splits and models.
* The 232 MB SQLite DB and large Parquet/model binaries are `.gitignore`d and
  regenerated by the scripts; a 5,000-row SQL sample dump and a 2,000-row CSV
  sample are committed.

---

*Author: [Student Name] · [Student ID] — Big Data Programming Project.*
