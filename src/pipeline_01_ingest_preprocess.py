"""
Stage 1 — Ingestion & Preprocessing  (PySpark)
==============================================
Loads the raw NYC "Bus Breakdown and Delays" CSV (~530k rows, 21 cols),
cleans it, engineers features/labels, decomposes it into a normalised
star schema (1 fact + 4 dimension tables) and rejoins the dimensions with a
*broadcast* join to build the analysis-ready feature table.

Big-data techniques demonstrated (mapped to the brief)
------------------------------------------------------
* explicit schema + quote-aware distributed CSV read
* repartition to 8 partitions  (> 4 required) and report counts
* .cache() the reused raw frame                     (caching)
* broadcast() the small dimension tables            (broadcast join)
* write partitioned Parquet                          (persistence / checkpoint)
* wall-clock timing of each stage                    (algorithmic efficiency)

Run:  .venv/bin/python src/pipeline_01_ingest_preprocess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import broadcast

from config.spark_config import get_spark
from src.utils import (RAW_CSV, PROCESSED_PARQUET, DATA_PROCESSED, DATA_RAW,
                       timed, partition_report)

# ---- explicit schema: read everything as string, parse deliberately ---------
RAW_COLUMNS = [
    "School_Year", "Busbreakdown_ID", "Run_Type", "Bus_No", "Route_Number",
    "Reason", "Schools_Serviced", "Occurred_On", "Created_On", "Boro",
    "Bus_Company_Name", "How_Long_Delayed", "Number_Of_Students_On_The_Bus",
    "Has_Contractor_Notified_Schools", "Has_Contractor_Notified_Parents",
    "Have_You_Alerted_OPT", "Informed_On", "Incident_Number",
    "Last_Updated_On", "Breakdown_or_Running_Late", "School_Age_or_PreK",
]
RAW_SCHEMA = T.StructType([T.StructField(c, T.StringType(), True) for c in RAW_COLUMNS])


def load_raw(spark):
    """Quote-aware distributed CSV read with an explicit schema."""
    csv_path = RAW_CSV if RAW_CSV.exists() else (DATA_RAW / "bus_breakdown_delays_full.csv")
    df = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("mode", "PERMISSIVE")            # tolerate the odd malformed row
        .schema(RAW_SCHEMA)
        .csv(str(csv_path))
    )
    return df


def engineer_features(df):
    """Parse timestamps, delay minutes, categorical cleaning and labels."""
    # --- timestamps (format: MM/dd/yyyy hh:mm:ss a, 12-hour) -----------------
    occurred = F.to_timestamp("Occurred_On", "MM/dd/yyyy hh:mm:ss a")
    df = (
        df.withColumn("occurred_ts", occurred)
          .withColumn("hour", F.hour("occurred_ts"))
          .withColumn("day_of_week", F.dayofweek("occurred_ts"))     # 1=Sun..7=Sat
          .withColumn("month", F.month("occurred_ts"))
          .withColumn("year", F.year("occurred_ts"))
    )
    df = df.withColumn(
        "is_weekend", F.col("day_of_week").isin(1, 7).cast("int")
    ).withColumn(
        "time_band",
        F.when((F.col("hour") >= 6) & (F.col("hour") <= 9), "AM_Peak")
         .when((F.col("hour") >= 14) & (F.col("hour") <= 17), "PM_Peak")
         .when((F.col("hour") >= 10) & (F.col("hour") <= 13), "Midday")
         .otherwise("OffPeak"),
    ).withColumn(
        "season",
        F.when(F.col("month").isin(12, 1, 2), "Winter")
         .when(F.col("month").isin(3, 4, 5), "Spring")
         .when(F.col("month").isin(6, 7, 8), "Summer")
         .otherwise("Autumn"),
    )

    # --- robust delay-minutes parser from messy free text --------------------
    du = F.upper(F.trim(F.col("How_Long_Delayed")))
    rng_lo = F.regexp_extract(du, r"(\d+)\s*-\s*(\d+)", 1).cast("double")
    rng_hi = F.regexp_extract(du, r"(\d+)\s*-\s*(\d+)", 2).cast("double")
    first_num = F.regexp_extract(du, r"(\d+)", 1).cast("double")
    has_hour = du.rlike(r"HOUR|HR")
    delay = (
        F.when((rng_lo > 0) & (rng_hi > 0), (rng_lo + rng_hi) / 2.0)
         .when(has_hour, first_num * 60.0)
         .otherwise(first_num)
    )
    df = df.withColumn("delay_minutes_raw", delay)
    # winsorise to a sensible school-bus range; keep >180 as capped outliers
    df = df.withColumn(
        "delay_minutes",
        F.when(F.col("delay_minutes_raw") > 180, F.lit(180.0))
         .when(F.col("delay_minutes_raw") < 0, F.lit(None))
         .otherwise(F.col("delay_minutes_raw")),
    )

    # --- categorical cleaning -------------------------------------------------
    op = F.upper(F.trim(F.col("Bus_Company_Name")))
    op = F.regexp_replace(op, r"\s*\(.*$", "")             # drop "(B2321" vendor code
    op = F.regexp_replace(op, r"[.,]+$", "")
    df = df.withColumn("operator", F.when(op == "", "UNKNOWN").otherwise(op))
    df = df.withColumn(
        "boro",
        F.when(F.trim(F.col("Boro")) == "", "Unknown")
         .otherwise(F.initcap(F.trim(F.col("Boro")))),
    )
    df = df.withColumn("reason", F.trim(F.col("Reason")))
    df = df.withColumn("run_type", F.trim(F.col("Run_Type")))
    df = df.withColumn(
        "students_on_bus",
        F.coalesce(F.col("Number_Of_Students_On_The_Bus").cast("int"), F.lit(0)),
    )
    df = df.withColumn(
        "students_on_bus",
        F.when(F.col("students_on_bus") < 0, 0).otherwise(F.col("students_on_bus")),
    )
    df = df.withColumn(
        "school_age",
        F.when(F.upper(F.trim(F.col("School_Age_or_PreK"))).contains("PRE"), "PreK")
         .otherwise("School-Age"),
    )
    # Yes/No -> 1/0
    for raw, new in [
        ("Has_Contractor_Notified_Schools", "notified_schools"),
        ("Has_Contractor_Notified_Parents", "notified_parents"),
        ("Have_You_Alerted_OPT", "alerted_opt"),
    ]:
        df = df.withColumn(new, (F.upper(F.trim(F.col(raw))) == "YES").cast("int"))

    # --- labels ---------------------------------------------------------------
    df = df.withColumn(
        "incident_type", F.trim(F.col("Breakdown_or_Running_Late"))
    ).withColumn(
        "is_breakdown", (F.col("incident_type") == "Breakdown").cast("int")
    )
    # on-time / reliability flag (delay within urban tolerance proxy)
    df = df.withColumn(
        "on_time", F.when(F.col("delay_minutes") <= 15, 1).otherwise(0)
    )
    df = df.withColumn("busbreakdown_id", F.col("Busbreakdown_ID").cast("long"))
    df = df.withColumn("school_year", F.col("School_Year"))
    return df


def quality_filter(df):
    """Keep rows usable for supervised learning; report how many are dropped."""
    before = df.count()
    df = df.filter(F.col("incident_type").isin("Breakdown", "Running Late"))
    df = df.filter(F.col("occurred_ts").isNotNull())
    df = df.filter(F.col("busbreakdown_id").isNotNull())
    # drop obviously-bogus timestamps (data-entry typos, e.g. year 1900)
    df = df.filter((F.col("year") >= 2010) & (F.col("year") <= 2027))
    after = df.count()
    print(f"[quality] kept {after:,} / {before:,} rows "
          f"({100*after/before:.1f}%); dropped {before-after:,}")
    return df


def build_star_schema(df):
    """Decompose the wide frame into dimension tables + a fact table."""
    dim_operator = (
        df.select("operator").distinct()
          .withColumn("operator_id", F.monotonically_increasing_id())
    )
    dim_reason = (
        df.select("reason").distinct()
          .withColumn("reason_id", F.monotonically_increasing_id())
    )
    dim_boro = (
        df.select("boro").distinct()
          .withColumn("boro_id", F.monotonically_increasing_id())
    )
    dim_calendar = (
        df.select("year", "month", "day_of_week", "season",
                  "is_weekend", "school_year").distinct()
          .withColumn("date_id", F.monotonically_increasing_id())
    )
    return dim_operator, dim_reason, dim_boro, dim_calendar


def main():
    spark = get_spark("Stage1-IngestPreprocess")
    print("Spark UI:", spark.sparkContext.uiWebUrl)

    with timed("01_load_raw_csv"):
        raw = load_raw(spark)
        partition_report(raw, "raw (after CSV read)")

    # repartition to 8 (>4) and cache the reused frame
    raw = raw.repartition(8)
    partition_report(raw, "raw (after repartition)")
    raw = raw.cache()
    with timed("02_count_after_cache"):
        n_raw = raw.count()
    print(f"[rows] raw records: {n_raw:,}")

    with timed("03_feature_engineering"):
        feat = engineer_features(raw)
        feat = quality_filter(feat)

    # build & broadcast-join dimension tables -> demonstrates broadcast join
    dim_operator, dim_reason, dim_boro, dim_calendar = build_star_schema(feat)
    with timed("04_broadcast_join_dims"):
        joined = (
            feat
            .join(broadcast(dim_operator), "operator", "left")
            .join(broadcast(dim_reason), "reason", "left")
            .join(broadcast(dim_boro), "boro", "left")
        )
        _ = joined.count()
    print("[join] broadcast-joined 3 dimension tables into fact frame")

    # final analysis-ready feature table
    feature_cols = [
        "busbreakdown_id", "school_year", "year", "month", "day_of_week",
        "hour", "time_band", "season", "is_weekend",
        "operator", "operator_id", "boro", "boro_id", "run_type", "reason",
        "reason_id", "school_age", "students_on_bus",
        "notified_schools", "notified_parents", "alerted_opt",
        "delay_minutes", "on_time", "incident_type", "is_breakdown",
        "occurred_ts",
    ]
    final = joined.select(*feature_cols)

    with timed("05_write_parquet"):
        (final.repartition(8)
              .write.mode("overwrite")
              .parquet(str(PROCESSED_PARQUET)))
    partition_report(final, "final feature table")
    print(f"[persist] wrote Parquet -> {PROCESSED_PARQUET}")

    # persist dimension tables as parquet for the DB stage
    for name, d in [("dim_operator", dim_operator), ("dim_reason", dim_reason),
                    ("dim_boro", dim_boro), ("dim_calendar", dim_calendar)]:
        d.write.mode("overwrite").parquet(str(DATA_PROCESSED / f"{name}.parquet"))
    print("[persist] wrote 4 dimension tables")

    # small CSV sample for the git repo (brief: 'sample datasets')
    sample_pd = final.limit(2000).toPandas()
    sample_pd.to_csv(DATA_PROCESSED / "sample_feature_table.csv", index=False)
    print(f"[sample] wrote {len(sample_pd)} sample rows to CSV")

    print("\n=== Stage 1 complete ===")
    final.printSchema()
    spark.stop()


if __name__ == "__main__":
    main()
