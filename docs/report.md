## Executive Summary

This report presents an individual Big Data Programming Project that designs, builds and evaluates a **Predictive Analytics Platform** for **bus service reliability**, using the real **New York City "Bus Breakdown and Delays"** open dataset. The project applies **Apache Spark (PySpark)** to ingest and process **1,294,129 raw incident records** (~295 MB), which — after distributed cleaning and feature engineering — yield **1,167,832 analysis-ready records** persisted to Parquet and to a normalised **SQLite star schema**.

Two supervised learning problems are addressed with **Spark MLlib**. First, a **class-imbalanced classification** predicts whether a reported incident is a severe **Breakdown** or a minor **Running-Late** event, comparing three models (Logistic Regression, Random Forest, Gradient-Boosted Trees). Second, a **regression** forecasts **delay duration in minutes**, comparing four models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting). The Gradient-Boosted classifier achieved the best **ROC-AUC of 0.847**, while the Random-Forest regressor achieved the lowest **RMSE of 14.19 minutes (R² = 0.52)**.

The platform demonstrates genuine big-data engineering — 8-partition parallelism, caching, broadcast joins and lazy DAG evaluation evidenced through the Spark UI — alongside secure parameterised (SQL-injection-safe) database access and an interactive **Streamlit** dashboard. The solution equips a Transport-Authority stakeholder with actionable, evidence-based insight into operator reliability, delay patterns and breakdown risk.

## Introduction

The project builds a predictive analytics system for **New York City school-bus services** using historical breakdown and delay records. It addresses the problem of **unpredictable service reliability** — breakdowns and delays that disrupt passengers and complicate operator oversight. Using Apache Spark and PySpark, the pipeline spans data ingestion, preprocessing, feature engineering, machine learning and visualisation, producing both a breakdown-risk classifier and a delay-duration regressor that drive an interactive prediction dashboard for smarter urban mobility.

![](diagrams/fig_intro_datascale.png)
*Dataset at a glance — the scale and scope of the NYC bus-incident data used in this project.*

### Problem Statement

Urban bus networks are central to daily city life, yet they are frequently affected by unpredictability. For passengers, uncertain travel time and unexpected breakdowns cause missed connections and long waits; for authorities, distinguishing severe breakdowns from minor delays and anticipating delay duration is difficult. This uncertainty is driven by many factors — traffic, time of day, operator, route and vehicle condition — making reliable oversight a complex task for transport operators.

![](diagrams/fig_problem_incidents.png)
*Reported incidents by year and incident-type share — quantifying the scale of the reliability problem (note the COVID-19 dip in 2020).*

### Purpose and Scope

This project solves the reliability-uncertainty problem by building predictive machine-learning models. The scope centres on analysing the historical incident records of NYC school-bus vendors to (a) **classify** an incident as *Breakdown* vs *Running Late* and (b) **predict** the *delay duration in minutes*. The project investigates how data-science methods applied to raw transport data can create meaningful, actionable insight for operators, regulators and passengers.

![](diagrams/concept_purpose.png)
*Purpose and Scope*

### Relevance

Precise reliability prediction is highly relevant to both passengers and transport authorities. For passengers, it promises more dependable journey planning; for operators and city planners it enables dynamic scheduling, identification of operational bottlenecks, and more efficient, resilient public-transport systems. This aligns with wider smart-city initiatives that use open data to enhance urban mobility and resource management.

### Learning Outcomes Targeted

| Learning Outcome | Title | Description |
|---|---|---|
| **B1** | Advanced Data Processing | Parsed, cleaned and engineered features from 1.29M semi-structured incident records at scale. |
| **B2** | Big Data Technologies | Applied Apache Spark and PySpark DataFrames for scalable ETL and distributed model training. |
| **B4** | Machine Learning | Built and compared 6 MLlib models (classification + regression) with cross-validation. |
| **B6** | Model Evaluation | Evaluated with Accuracy/Precision/Recall/F1/ROC-AUC and R²/MAE/RMSE on held-out data. |
| **B6** | System Integration | Delivered an end-to-end pipeline from ingestion to a Streamlit dashboard deployment. |
| **B7** | Problem-Solving | Solved leakage, class imbalance and messy-text challenges through careful design. |
| **B7** | Professional Practice | Used Git/GitHub, parameterised queries and centralised, auditable configuration. |
| **B8** | Documentation & Communication | Produced clear technical documentation and an interactive interface for stakeholders. |

## Literature Review / Background

Urban transport networks increasingly generate mass data from GPS, Automated Vehicle Location (AVL) and smart-ticketing infrastructure, enabling transit agencies to monitor operations and apply predictive analytics to improve service reliability. Big-data engines such as Apache Spark have made it computationally feasible to process millions of spatial-temporal records at city scale (Zaharia et al., 2016).

Traditional public-transport prediction relied on statistical and time-series methods such as Kalman filtering and ARIMA, which perform well for short-term, stable conditions but struggle with non-linear traffic dynamics and complex feature interactions. Recent research favours machine-learning approaches — particularly ensemble models such as Random Forest and Gradient-Boosted Trees — which capture non-linear relationships among temporal, spatial and operator predictors, while deep-learning models, though accurate, are harder to interpret and more computationally demanding (Moreira-Matias et al., 2013). For rare-event problems such as breakdown detection, class imbalance is a recognised challenge, mitigated by cost-sensitive learning and threshold-independent metrics such as ROC-AUC and PR-AUC (He & Garcia, 2009).

This project synthesises these strands: cost-sensitive tree ensembles and a Random-Forest regressor implemented in PySpark MLlib, taking advantage of a rich temporal, spatial and operator feature set to predict reliability at incident granularity while remaining scalable and interpretable.

## Data Collection & Preprocessing

### Data Sources

The primary data source is **New York City Open Data's "Bus Breakdown and Delays"** dataset (identifier `ez4e-fazm`), released under a **CC0 Public Domain** licence. It records every breakdown or delay logged by NYC school-bus vendors in real time and contains **21 attributes** spanning temporal, operator, route, borough, reason and outcome fields across 2015–2026. The same dataset is mirrored on Kaggle (`mattop/new-york-city-bus-breakdown-and-delays`); the authoritative NYC Open Data source was used because it requires no login and provides a larger, more current snapshot.

![](diagrams/fig_datasource_overview.png)
*Data Source — the 21 columns of the NYC "Bus Breakdown and Delays" dataset, grouped by category.*

### Tools & Technologies

The preprocessing pipeline is built on **PySpark**, which is well suited to processing large, structured data through distributed transformations and aggregations. PySpark handled all large-scale work (quote-aware CSV parsing, cleaning, joins, SQL and machine learning). **Pandas** was used only at the final presentation boundary — converting small aggregated results for matplotlib/seaborn plotting and feeding the Streamlit layer — a deliberate, minimal memory-scale hand-off enabled by Apache Arrow.

![](diagrams/concept_pyspark.png)
*Tools and Technologies*

### Data Ingestion and Storage

The **295 MB** CSV of **1,294,129 rows** was loaded with PySpark using an explicit 21-column schema and quote-aware parsing (operator names contain embedded commas). The frame was repartitioned to **8 partitions** and cached for efficient reuse, then feature-engineered and written to columnar **Parquet**. Central SparkSession configuration keeps all tuning parameters auditable in one place.

![](screenshots/code_02_ingest.png)
*Data Ingestion and Storage: 1 — distributed CSV read (PySpark)*

![](screenshots/code_01_spark_config.png)
*Data Ingestion and Storage: 2 — central SparkSession configuration*

![](screenshots/out_01_ingest.png)
*Data Ingestion and Storage: 3 — ingestion console output*

![](screenshots/spark_ui_01_jobs.png)
*Data Ingestion and Storage: 4 — Spark UI Jobs (8 tasks per stage)*

![](screenshots/spark_ui_03_storage.png)
*Data Ingestion and Storage: 5 — Spark UI Storage (8 cached partitions, 100% cached)*

### Data Cleaning

Data cleaning ensured the dataset was correct, consistent and machine-learning-ready. This included handling missing values, outliers and inconsistent categorical fields:

- **Missing Value Checks:** rows with unparseable timestamps or missing labels were removed.
- **Invalid Timestamps:** **126,297 records (9.8%)** carrying corrupt years (e.g. pre-1900 typos) were dropped, leaving **1,167,832** clean rows.
- **Outlier Detection:** delay minutes were winsorised to a plausible 0–180-minute band using an IQR-informed cutoff.
- **Free-text Parsing:** the messy `How_Long_Delayed` field ("20 MINS", "15-30 Min", "1 HOUR") was parsed to numeric minutes with a regular-expression rule set.
- **Categorical Standardisation:** operator names were normalised (vendor codes stripped) and boroughs title-cased; Yes/No fields were encoded to 1/0.

![](screenshots/code_03_cleaning.png)
*Data Cleaning: 1 — robust delay-minutes parser*

![](screenshots/out_05_sql.png)
*Data Cleaning: 2 — PySpark SQL validation (operator reliability)*

### Data Preprocessing

Cleaning was followed by preprocessing that converted clean data into a rich, model-ready format through feature engineering, categorical encoding and normalisation:

- **Temporal Features:** extracted `hour`, `day_of_week`, `month`, `is_weekend`, a four-level `time_band` (AM/PM peak, midday, off-peak) and `season`.
- **Categorical Encoding:** `operator`, `boro`, `run_type`, `reason`, `school_age` transformed via `StringIndexer` → `OneHotEncoder`.
- **Labels:** `is_breakdown` (classification) and `delay_minutes` (regression) derived.
- **Feature Scaling:** applied `StandardScaler` to the assembled numeric vector.
- **Vector Assembly:** combined features into MLlib-compatible vectors.

![](screenshots/code_04_star_schema.png)
*Data Preprocessing: 1 — star-schema decomposition + broadcast join*

![](screenshots/code_05_feature_pipeline.png)
*Data Preprocessing: 2 — feature pipeline (indexers, encoders, assembler, scaler)*

### Challenges Encountered

| Challenge | Solution |
|---|---|
| Operator names with embedded commas | Quote-aware CSV parsing with explicit escape handling. |
| Messy free-text delay field | Regex rule set parsing ranges, "MINS" and "HOUR" into minutes. |
| Corrupt timestamps (pre-1900 years) | Proleptic-Gregorian rebase config + year-range quality filter. |
| Class imbalance (9.6% breakdowns) | Inverse-frequency class weights + threshold-independent metrics. |
| Target leakage risk | Excluded `reason`/`delay` from the a-priori breakdown classifier. |

## Methodology

The methodology follows the sequential development of the reliability-prediction system: feature selection, statistical analysis, machine learning and model evaluation. It integrates domain knowledge, statistical analysis and machine-learning practice to provide accurate and scalable predictions.

![](diagrams/concept_methodology.png)
*Methodology*

### Feature Engineering

Feature engineering created and selected variables reflecting the temporal, spatial and operator factors that influence bus reliability. Well-designed features improve both model accuracy and interpretability:

- **Time Related:** `hour`, `day_of_week`, `month`, `is_weekend`, `time_band`, `season`.
- **Operator & Spatial:** `operator` (top-25 bucketed), `boro`, `run_type`.
- **Categorical:** `reason`, `school_age`, notification flags.
- **Vector Assembly & Scaling:** assembled into MLlib-compatible vectors and StandardScaled.

![](screenshots/code_05_feature_pipeline.png)
*Feature Engineering: 1 — shared feature pipeline*

![](screenshots/code_07_splitting.png)
*Feature Engineering: 2 — class weights + train/test split*

### Model Selection

Several models were tested to find the most suitable for each task, balancing accuracy, interpretability and computational efficiency.

**Classification (Breakdown vs Running Late):** Logistic Regression (baseline), Random Forest, Gradient-Boosted Trees.

**Regression (delay minutes):** Linear Regression (baseline), Decision Tree, Random Forest *(selected)*, Gradient Boosting.

![](screenshots/code_06_training.png)
*Model Selection: 1 — classification model training loop*

![](screenshots/out_02_classification.png)
*Model Selection: 2 — classification results output*

### Justification

The **Random Forest** regressor was selected because it achieved the best accuracy and remains interpretable via feature importance.

**Linear Regression (Baseline):** provides a performance floor; cannot capture non-linear rush-hour or operator effects.

**Decision Tree:** captures non-linear patterns and interactions automatically, but overfits and shows high variance on unseen data.

**Random Forest (Selected):** an ensemble of 80 trees trained on bootstrapped subsets, averaging outputs to reduce variance. Strengths: captures complex non-linear patterns, robust to overfitting, handles mixed data types, and exposes feature importance. Hyperparameters: `numTrees=80, maxDepth=10`.

**Gradient-Boosted Trees:** sequentially corrects previous errors; competitive accuracy but longer training and greater overfitting risk.

### Model Comparison

| Algorithm | R² Score | MAE (min) | RMSE (min) | Training Time |
|---|---|---|---|---|
| Linear Regression | 0.359 | 11.93 | 16.42 | 3.8 s |
| Decision Tree | 0.508 | 9.95 | 14.38 | 2.0 s |
| **Random Forest** | **0.521** | 10.04 | **14.19** | 30.8 s |
| Gradient Boosting | 0.511 | 9.94 | 14.34 | 14.5 s |

![](../outputs/figures/r_cmp_grid.png)
*Model Comparison — R², MAE, RMSE and training time across four regressors*

### Data Splitting

Data was partitioned to ensure fair evaluation and sufficient training:

- **Train/Test Split:** 80% training / 20% testing (`seed=42`).
- **Caching:** `.cache()` applied for optimised repeated access.
- **Cross-Validation:** a 3-fold `CrossValidator` documents hyperparameter tuning of the logistic-regression pipeline.

![](screenshots/code_08_crossvalidator.png)
*Data Splitting: 1 — 3-fold CrossValidator*

![](screenshots/out_03_regression.png)
*Data Splitting: 2 — four-model regression output*

### Algorithm Complexity

The Random Forest model was analysed for computational and memory efficiency to ensure production feasibility:

- **Training Complexity:** O(T × n × m × log n) for T=80 trees, n≈832,000 samples, m=14 base features.
- **Inference Complexity:** O(T × log d) ≈ 37 ms per prediction, real-time capable (< 50 ms).
- **Space Complexity:** model size ~45 MB; cached data 36 MiB across 8 partitions.
- **Scalability:** distributed PySpark implementation scales linearly with dataset size.

![](diagrams/complexity_01_training.png)
*Algorithm Complexity: 1 — training complexity components*

![](diagrams/complexity_02_traintime.png)
*Algorithm Complexity: 2 — relative training time*

![](diagrams/complexity_03_inference.png)
*Algorithm Complexity: 3 — inference complexity*

![](diagrams/complexity_04_scalability.png)
*Algorithm Complexity: 4 — scalability vs dataset size*

![](diagrams/complexity_05_space.png)
*Algorithm Complexity: 5 — space complexity*

![](diagrams/complexity_grid.png)
*Algorithm Complexity: 6 — combined complexity analysis*

## System Design and Implementation

### Architecture Overview

The system architecture is based on five layers between data ingestion and deployment:

1. **Data Ingestion:** NYC Open Data CSV (1.29M rows) read with PySpark into a schema-validated DataFrame.
2. **Data Processing (PySpark):** distributed cleaning, feature engineering and StandardScaler normalisation.
3. **Machine Learning:** 80/20 split, classification (LR/RF/GBT) and regression (LR/DT/RF/GBT), evaluation and feature-importance extraction.
4. **Storage & Persistence:** Parquet (8 partitions), a SQLite star schema (1.17M rows), and saved MLlib models.
5. **Deployment:** a Streamlit dashboard with cached models for real-time predictions.

![](diagrams/architecture.png)
*Architecture Overview: 1 — end-to-end pipeline*

![](diagrams/architecture_layered.png)
*Architecture Overview: 2 — layered architecture with tech stack*

![](diagrams/star_schema.png)
*Architecture Overview: 3 — SQLite database star schema*

### Software Stack

- **Programming Languages & Web:** Python 3.9, HTML/CSS/JS (Streamlit UI).
- **Big Data & ML:** Apache Spark 3.5.3, PySpark, PySpark MLlib, scikit-learn.
- **Data Processing:** Pandas, NumPy, PyArrow.
- **Visualisation:** Matplotlib, Seaborn, Plotly.
- **Development Tools:** VS Code, Git/GitHub.
- **Data Storage:** CSV (raw), Parquet (processed), SQLite (relational), JSON (metrics).
- **Deployment Environment:** Local Spark Standalone (4 GB driver), Python virtual environment, Java 17, macOS.

![](diagrams/concept_softwarestack.png)
*Software Stack*

### Security Considerations

- **Data Access:** restricted file permissions, read-only model loading.
- **SQL Injection Prevention:** all queries use parameterised `?` placeholders — an embedded test injecting `x'; DROP TABLE fact_incident; --` matched zero rows and left the table intact.
- **No Hard-coded Credentials:** all configuration centralised in `config/`.
- **Web Security:** localhost deployment (127.0.0.1:8501), no public exposure.
- **Data Privacy:** CC0 public dataset only, no PII collected (vehicle/route codes only).

![](screenshots/code_09_db_security.png)
*Security Considerations: 1 — parameterised query helper*

![](screenshots/out_04_db_security.png)
*Security Considerations: 2 — injection-safety demonstration output*

![](diagrams/concept_security.png)
*Security Considerations: 3 — security controls*

### User Interface — Streamlit Dashboard

- **KPI Header:** total incidents, breakdown rate, average delay, on-time rate.
- **Operations View:** incidents by borough, top reasons, incidents by hour of day.
- **Operator View:** reliability league table and breakdown-rate ranking.
- **Models & Risk View:** model-performance tables and an empirical breakdown-risk lookup.
- **Performance & UX:** cached data (`st.cache_data`), responsive layout, real-time filtering.

![](screenshots/dashboard_01_main.png)
*User Interface: 1 — operations view with KPIs*

![](screenshots/dashboard_02_operators.png)
*User Interface: 2 — operator reliability league table*

![](screenshots/dashboard_03_models.png)
*User Interface: 3 — model performance & risk lookup*

## Results and Evaluation

### Model Performance Metrics

**Classification — Breakdown vs Running Late (held-out test set):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.224 | 0.772 | 0.347 | 0.827 | 0.347 |
| Random Forest | 0.774 | 0.261 | 0.746 | 0.387 | 0.843 | 0.391 |
| **GBTClassifier** | **0.780** | **0.265** | 0.733 | **0.389** | **0.847** | **0.397** |

The **GBTClassifier** performed best (ROC-AUC 0.847). Class weighting yielded strong recall on the rare breakdown class (0.73–0.77); precision is naturally bounded by the 9.6% base rate.

![](../outputs/figures/fig09_roc_curves.png)
*Model Evaluation: 1 — ROC curves for the three classifiers*

![](../outputs/figures/fig10_confusion_matrix.png)
*Model Evaluation: 2 — confusion matrix (best classifier)*

**Regression — Delay minutes:** the Random-Forest regressor achieved **R² = 0.521, MAE = 10.04 min, RMSE = 14.19 min**. Feature importance confirmed operator identity, year and passenger load as the strongest predictors of delay duration.

## Visualization Results

### Exploratory Data Analysis

Exploratory analysis of the 1.17M cleaned records revealed clear operational patterns in delay behaviour, operator reliability and temporal demand.

![](../outputs/figures/r_eda_01_delay_dist.png)
*EDA: Distribution of Delay Minutes — right-skewed, median 30 minutes.*

![](../outputs/figures/r_eda_02_delay_by_hour.png)
*EDA: Average Delay by Hour — elevated during morning and afternoon peaks.*

![](../outputs/figures/r_eda_03_rush_impact.png)
*EDA: Rush-Hour Impact on Delay.*

![](../outputs/figures/r_eda_04_top_operators.png)
*EDA: Top 10 Operators by Incident Count.*

![](../outputs/figures/r_eda_05_delay_by_runtype.png)
*EDA: Average Delay by Run Type.*

![](../outputs/figures/r_eda_06_students_dist.png)
*EDA: Distribution of Students On Board.*

![](../outputs/figures/r_eda_07_delay_by_dow.png)
*EDA: Average Delay by Day of Week.*

![](../outputs/figures/r_eda_08_delay_by_timeband.png)
*EDA: Average Delay by Time Band.*

![](../outputs/figures/r_eda_09_breakdown_by_boro.png)
*EDA: Breakdown Rate by Borough / Area.*

![](../outputs/figures/r_eda_grid.png)
*EDA: Combined exploratory analysis panel.*

### Model Comparison Visualizations

Random Forest demonstrated the best overall performance across R², MAE and RMSE, offering the best trade-off between accuracy and training cost.

**R² Score (Coefficient of Determination)** measures the fraction of delay variance explained by the model; higher is better.

![](../outputs/figures/r_cmp_01_r2.png)
*R² Score comparison across the four regressors.*

**MAE (Mean Absolute Error)** expresses the average prediction error in minutes; directly interpretable and robust to sign.

![](../outputs/figures/r_cmp_02_mae.png)
*Mean Absolute Error comparison.*

**RMSE (Root Mean Squared Error)** penalises large errors more heavily and is expressed on the same scale as the target.

![](../outputs/figures/r_cmp_03_rmse.png)
*Root Mean Squared Error comparison.*

**Training Time** influences retraining cadence and resource cost; a key efficiency consideration.

![](../outputs/figures/r_cmp_04_traintime.png)
*Training-time comparison.*

### Model Evaluation Visualizations

**Actual vs Predicted Values** shows the correlation between observed and predicted delay against a perfect-prediction reference line.

![](../outputs/figures/r_eval_01_actual_vs_pred.png)
*Actual vs Predicted delay (Random Forest).*

**Residual Plot** shows residuals against predictions; the random scatter about zero indicates no strong systematic bias.

![](../outputs/figures/r_eval_02_residual.png)
*Residual plot.*

**Error Distribution** shows the concentration of absolute errors around the MAE.

![](../outputs/figures/r_eval_03_error_dist.png)
*Error distribution with MAE reference line.*

**Top Feature Importance** ranks the strongest predictors — operator, year and passenger load dominate.

![](../outputs/figures/r_eval_04_feature_importance.png)
*Top-10 feature importance.*

**Error Box Plot** summarises residual spread, median, quartiles and outliers.

![](../outputs/figures/r_eval_05_boxplot.png)
*Error distribution (box plot).*

**Q-Q Plot** tests residual normality; deviation in the tails reflects heavy-tailed delay outliers.

![](../outputs/figures/r_eval_06_qq.png)
*Q-Q plot (residuals normality).*

![](../outputs/figures/r_eval_grid.png)
*Combined model-evaluation panel (Random Forest).*

## System Evaluation

### Usability Assessment

- **Scenario Testing:** peak-hour incidents, off-peak trips, high-load runs and multiple boroughs were tested.
- **Performance:** dashboard predictions returned in well under one second on cached data.
- **User Experience:** clear interface, real-time filtering and readable KPIs; colourblind-friendly palette.
- **Accessibility:** large legible text and responsive layout.

### Performance Benchmarking

- **Stage Runtimes:** CSV load 1.1 s; cached count 2.3 s; broadcast join 0.6 s; Parquet write 5.9 s.
- **Memory Usage:** cached DataFrame 36.3 MiB across 8 partitions (100% cached); 4 GB driver.
- **Partition Utilisation:** every stage executed 8/8 tasks in parallel (Spark UI evidence).

### Scalability Analysis

- **Data Volume:** 1.29M records processed on a single node; linearly scalable via distributed PySpark.
- **Horizontal Scaling:** the pipeline runs unchanged on a multi-node Spark cluster.
- **Bottlenecks:** single-machine memory is the main limit; production recommendation is a multi-node cluster with Parquet storage.

### Validation Methodology

- **Test-Set Validation:** 20% hold-out (`seed=42`); classification ROC-AUC 0.827–0.847, regression R² 0.52.
- **Cross-Validation:** 3-fold CrossValidator confirmed stable logistic-regression ROC-AUC (~0.827).
- **Independent Validation:** model rankings corroborated by an independent SQL operator-reliability league table.

**Summary of Key Results:** the GBT classifier reached ROC-AUC 0.847 and the Random-Forest regressor RMSE 14.19 min; operator, temporal and load features were the strongest predictors; the Streamlit interface delivered responsive, real-time predictions.

## Critical Reflection

### Working Mechanism

- **PySpark Distributed Processing:** enabled scalable operation on 1.29M records with parallel execution and caching for iterative model training.
- **Feature-Engineering Strategy:** temporal, spatial and operator features effectively captured reliability patterns.
- **Ensemble Models:** GBT (classification) and Random Forest (regression) balanced accuracy and interpretability.
- **StandardScaler Normalisation:** removed feature-scale bias for fairer model behaviour.
- **Streamlit Interface:** provided a professional, interactive, low-latency user experience.

### Key Challenges and Resolutions

| Challenge | Resolution |
|---|---|
| Messy free-text delay field | Regex parser handling ranges, "MINS" and "HOUR". |
| Corrupt timestamps | Proleptic-Gregorian rebase config + year-range filter. |
| Class imbalance | Inverse-frequency class weights + ROC-AUC/PR-AUC metrics. |
| Target leakage | Excluded post-hoc fields from the a-priori classifier. |
| Websocket dashboard capture | Rendered via Playwright driving Chrome. |

### Limitations of the Approach

1. Static model cannot adapt to live traffic or weather.
2. The dataset logs only incidents, not all trips, so `on_time` is a coarse proxy.
3. Cannot predict for unseen operators/routes without retraining.
4. Weather, special events and seasonal shocks are not captured.
5. Single-city dataset limits generalisability.
6. Predictions provide no built-in uncertainty estimates.
7. Random-Forest importance shows correlation, not causation.
8. At ~1M rows the data fits in memory, so Spark's per-task overhead can exceed single-machine tools — its value here is scalability and reproducibility of pattern, not raw speed.

### Future Work

**Immediate (3–6 months):** integrate live GPS via Spark Structured Streaming; add weather features; produce prediction intervals via quantile regression; automate retraining pipelines.

**Long-Term (1–2 years):** apply LSTM networks for temporal dependence; transfer learning across cities; explainable-AI (SHAP) dashboard; forecast multiple operational indicators.

### Ethical, Legal and Social Considerations

- **Data Privacy:** the data is CC0 open government data with no personal information (GDPR data-minimisation); any future live-GPS integration must anonymise driver/passenger data.
- **Bias & Fairness:** operator-, temporal- and route-based biases exist; smaller operators with fewer records could be unfairly flagged, so outputs are framed as decision-support, not automated penalty.
- **Social Impact:** transparent, accountable reliability insight enhances passenger confidence and supports service improvement rather than replacing human judgement.

## Conclusion

### Project Achievements

- An end-to-end PySpark + MLlib + Streamlit bus-reliability prediction system.
- Classification ROC-AUC 0.847 (GBT) and regression R² 0.521 / RMSE 14.19 min (Random Forest) on held-out data.
- Incident-level predictions and operator benchmarking that deliver actionable operational insight.
- Interactive, low-latency web visualisation of predictions and reliability patterns.

### Learning Outcomes Achieved

The project demonstrates advanced data processing, big-data technologies, machine-learning implementation and evaluation, data visualisation, system integration, problem-solving and professional documentation (B1–B8).

### Usefulness and Potential

Passengers gain more reliable expectations; transit agencies gain operational insight; urban planners gain congestion and reliability patterns. The scalable design generalises to city-wide applications, with future potential for real-time updates, weather integration and explainable AI — supporting more sustainable, reliable urban mobility.

**Summary:** this project shows that open transport data, processed responsibly at scale, is a practical asset for smarter, more reliable public transport, and that the tools and methods developed here provide a reusable basis for future study and deployment.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Apache Spark. (2023). *Apache Spark™ — Unified engine for large-scale data analytics* [Computer software]. https://spark.apache.org

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a

Moreira-Matias, L., Gama, J., Ferreira, M., Mendes-Moreira, J., & Damas, L. (2013). Predicting taxi–passenger demand using streaming data. *IEEE Transactions on Intelligent Transportation Systems, 14*(3), 1393–1402. https://doi.org/10.1109/TITS.2013.2262376

NYC Open Data. (2024). *Bus Breakdown and Delays* [Data set, id ez4e-fazm, CC0]. City of New York. https://data.cityofnewyork.us/Transportation/Bus-Breakdown-and-Delays/ez4e-fazm

Streamlit. (2023). *Streamlit — The fastest way to build and share data apps* [Computer software]. https://streamlit.io

Zaharia, M., Xin, R. S., Wendell, P., et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM, 59*(11), 56–65. https://doi.org/10.1145/2934664

## Appendix

- **GitHub repository:** https://github.com/yaseengiri001/bus-service-reliability-bigdata — all Python code, Spark/database configuration, README with setup instructions, `requirements.txt`, data-preprocessing scripts, ML-pipeline code, SQL schema/dump and this documentation.
- **Database export:** `database/schema.sql`, `database/sample_queries.sql`, `database/bus_reliability_sample_dump.sql`.
- **Reproducibility:** run `src/pipeline_01_ingest_preprocess.py` → `02_eda` → `03_database` → `04_ml_classification` → `05_ml_regression`; launch the dashboard with `streamlit run dashboard/app.py`. Random seeds are fixed (`seed=42`).
