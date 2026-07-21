"""
Spark UI evidence generator
===========================
Exercises the distributed-computing techniques the brief asks us to evidence
(partitioning, caching, shuffle, broadcast join, lazy evaluation / DAG) and
then holds the SparkSession open so the Spark UI at http://localhost:4040 can
be screenshotted (Jobs, Stages, Storage, Executors, SQL tabs).

Run:  .venv/bin/python src/spark_ui_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.storagelevel import StorageLevel

from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, DATA_PROCESSED, partition_report

HOLD_SECONDS = 480


def main():
    spark = get_spark("SparkUI-Evidence-BusReliability")
    sc = spark.sparkContext
    print("=" * 60)
    print("Spark UI available at:", sc.uiWebUrl)
    print("=" * 60)

    # --- read + repartition to 8 (lazy until an action) ----------------------
    df = spark.read.parquet(str(PROCESSED_PARQUET))
    partition_report(df, "parquet read")
    df = df.repartition(8, "operator")           # hash-repartition on a key
    partition_report(df, "after repartition(8)")

    # --- CACHE and materialise (populates the Storage tab: 8 partitions) -----
    df.persist(StorageLevel.MEMORY_AND_DISK)
    df.createOrReplaceTempView("incidents")
    n = df.count()
    print(f"[cache] materialised {n:,} rows across 8 in-memory partitions")

    # --- shuffle-heavy aggregation (Stages tab: shuffle read/write) ----------
    agg = (df.groupBy("operator")
             .agg(F.count("*").alias("incidents"),
                  F.avg("is_breakdown").alias("breakdown_rate"),
                  F.avg("delay_minutes").alias("avg_delay"))
             .orderBy(F.desc("incidents")))
    print("[shuffle] top operators by volume:")
    agg.show(5, truncate=False)

    # --- BROADCAST join (small dim broadcast to large fact) ------------------
    dim_boro = spark.read.parquet(str(DATA_PROCESSED / "dim_boro.parquet"))
    joined = df.join(broadcast(dim_boro), "boro", "left")
    print("[broadcast] joined rows:", joined.count())

    # --- Spark SQL query (SQL tab shows the query plan) ----------------------
    spark.sql("""
        SELECT time_band, COUNT(*) AS incidents,
               ROUND(AVG(delay_minutes),1) AS avg_delay
        FROM incidents GROUP BY time_band ORDER BY incidents DESC
    """).show(truncate=False)

    # --- report the DAG / lazy-eval + storage info ---------------------------
    print("\n[DAG] number of RDD partitions in cached frame:",
          df.rdd.getNumPartitions())
    print("[lazy] transformations build a DAG; only .count()/.show() trigger jobs")

    print(f"\n>>> Holding SparkSession open for {HOLD_SECONDS}s so the Spark UI "
          f"can be captured (Jobs / Stages / Storage / Executors / SQL).")
    print(">>> Visit http://localhost:4040")
    time.sleep(HOLD_SECONDS)
    spark.stop()


if __name__ == "__main__":
    main()
