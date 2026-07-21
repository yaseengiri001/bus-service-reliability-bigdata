"""
Stage 3 — Relational persistence: SQLite star schema  (secure DB layer)
=======================================================================
Loads the Spark-processed Parquet into a normalised SQLite star schema
(1 fact + 4 dimension tables with primary/foreign keys and indexes), then
demonstrates SECURE, *parameterised* querying (no string concatenation -> no
SQL-injection surface).

Deliverables written to database/:
  * schema.sql               — DDL
  * bus_reliability.db       — populated SQLite database (gitignored: large)
  * sample_queries.sql       — documented example analytical queries
  * bus_reliability_sample_dump.sql — schema + dims + 5k fact-row sample (git)

Run:  .venv/bin/python src/pipeline_03_database.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pyspark.sql import functions as F
from config.spark_config import get_spark
from src.utils import PROCESSED_PARQUET, DATA_PROCESSED, SQLITE_DB, DATABASE

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE dim_operator (
    operator_id INTEGER PRIMARY KEY,
    operator    TEXT NOT NULL
);
CREATE TABLE dim_reason (
    reason_id INTEGER PRIMARY KEY,
    reason    TEXT
);
CREATE TABLE dim_boro (
    boro_id INTEGER PRIMARY KEY,
    boro    TEXT NOT NULL
);
CREATE TABLE dim_date (
    date_key    TEXT PRIMARY KEY,      -- yyyy-mm-dd
    year        INTEGER,
    month       INTEGER,
    day_of_week INTEGER,
    season      TEXT,
    is_weekend  INTEGER,
    school_year TEXT
);
CREATE TABLE fact_incident (
    busbreakdown_id  INTEGER PRIMARY KEY,
    operator_id      INTEGER,
    reason_id        INTEGER,
    boro_id          INTEGER,
    date_key         TEXT,
    hour             INTEGER,
    time_band        TEXT,
    run_type         TEXT,
    school_age       TEXT,
    students_on_bus  INTEGER,
    notified_schools INTEGER,
    notified_parents INTEGER,
    alerted_opt      INTEGER,
    delay_minutes    REAL,
    on_time          INTEGER,
    incident_type    TEXT,
    is_breakdown     INTEGER,
    occurred_ts      TEXT,
    FOREIGN KEY (operator_id) REFERENCES dim_operator (operator_id),
    FOREIGN KEY (reason_id)   REFERENCES dim_reason (reason_id),
    FOREIGN KEY (boro_id)     REFERENCES dim_boro (boro_id),
    FOREIGN KEY (date_key)    REFERENCES dim_date (date_key)
);
CREATE INDEX idx_fact_operator ON fact_incident (operator_id);
CREATE INDEX idx_fact_boro     ON fact_incident (boro_id);
CREATE INDEX idx_fact_date     ON fact_incident (date_key);
CREATE INDEX idx_fact_bkd      ON fact_incident (is_breakdown);
"""

SAMPLE_QUERIES = """-- =============================================================
-- Sample analytical queries (star-schema joins). All application
-- code executes these via parameterised statements (see run_query()).
-- =============================================================

-- Q1. Operator reliability league table (>= :min_incidents incidents)
SELECT o.operator, COUNT(*) AS incidents,
       ROUND(AVG(f.is_breakdown), 3) AS breakdown_rate,
       ROUND(AVG(f.delay_minutes), 1) AS avg_delay_min
FROM fact_incident f
JOIN dim_operator o ON o.operator_id = f.operator_id
GROUP BY o.operator
HAVING COUNT(*) >= :min_incidents
ORDER BY breakdown_rate DESC;

-- Q2. Breakdown counts by borough for a given year
SELECT b.boro, COUNT(*) AS breakdowns
FROM fact_incident f
JOIN dim_boro b ON b.boro_id = f.boro_id
JOIN dim_date d ON d.date_key = f.date_key
WHERE f.is_breakdown = 1 AND d.year = :year
GROUP BY b.boro ORDER BY breakdowns DESC;

-- Q3. Top delay reasons for a specific operator
SELECT r.reason, COUNT(*) AS n, ROUND(AVG(f.delay_minutes), 1) AS avg_delay
FROM fact_incident f
JOIN dim_reason r ON r.reason_id = f.reason_id
JOIN dim_operator o ON o.operator_id = f.operator_id
WHERE o.operator = :operator
GROUP BY r.reason ORDER BY n DESC;

-- Q4. Peak-hour delay profile
SELECT hour, COUNT(*) AS incidents, ROUND(AVG(delay_minutes),1) AS avg_delay
FROM fact_incident GROUP BY hour ORDER BY hour;
"""


def build_database(spark):
    df = spark.read.parquet(str(PROCESSED_PARQUET))
    df = df.withColumn("date_key", F.date_format("occurred_ts", "yyyy-MM-dd")) \
           .withColumn("occurred_ts", F.col("occurred_ts").cast("string"))

    # dimension frames (small) -> pandas
    dim_operator = spark.read.parquet(str(DATA_PROCESSED / "dim_operator.parquet")).toPandas()
    dim_reason = spark.read.parquet(str(DATA_PROCESSED / "dim_reason.parquet")).toPandas()
    dim_boro = spark.read.parquet(str(DATA_PROCESSED / "dim_boro.parquet")).toPandas()
    dim_date = (df.select("date_key", "year", "month", "day_of_week",
                          "season", "is_weekend", "school_year")
                  .dropDuplicates(["date_key"]).toPandas())

    fact_cols = ["busbreakdown_id", "operator_id", "reason_id", "boro_id", "date_key",
                 "hour", "time_band", "run_type", "school_age", "students_on_bus",
                 "notified_schools", "notified_parents", "alerted_opt",
                 "delay_minutes", "on_time", "incident_type", "is_breakdown", "occurred_ts"]
    fact = df.select(*fact_cols).toPandas()
    print(f"[db] pulled fact={len(fact):,} rows, dims: "
          f"op={len(dim_operator)} reason={len(dim_reason)} boro={len(dim_boro)} date={len(dim_date)}")

    # (re)create the SQLite database
    if SQLITE_DB.exists():
        SQLITE_DB.unlink()
    (DATABASE / "schema.sql").write_text(SCHEMA_SQL)
    (DATABASE / "sample_queries.sql").write_text(SAMPLE_QUERIES)

    conn = sqlite3.connect(str(SQLITE_DB))
    conn.executescript(SCHEMA_SQL)

    # bulk insert with executemany inside one transaction (fast + parameterised)
    def bulk(table, frame, cols):
        ph = ",".join(["?"] * len(cols))
        rows = list(frame[cols].itertuples(index=False, name=None))
        conn.executemany(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})", rows)

    bulk("dim_operator", dim_operator, ["operator_id", "operator"])
    bulk("dim_reason", dim_reason, ["reason_id", "reason"])
    bulk("dim_boro", dim_boro, ["boro_id", "boro"])
    bulk("dim_date", dim_date, ["date_key", "year", "month", "day_of_week",
                                "season", "is_weekend", "school_year"])
    bulk("fact_incident", fact, fact_cols)
    conn.commit()
    print("[db] tables populated and committed")
    return conn


def run_query(conn, sql, params=()):
    """SECURE query helper: values bound via placeholders, never string-formatted."""
    cur = conn.cursor()
    cur.execute(sql, params)          # <-- parameters passed separately (no injection)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def security_demo(conn):
    """Show that a malicious 'operator' value cannot break out of the query."""
    print("\n=== Parameterised-query security demo ===")
    malicious = "x'; DROP TABLE fact_incident; --"
    sql = "SELECT COUNT(*) FROM fact_incident f JOIN dim_operator o " \
          "ON o.operator_id=f.operator_id WHERE o.operator = ?"
    _, rows = run_query(conn, sql, (malicious,))
    print(f"  malicious input handled safely -> rows matched: {rows[0][0]}")
    # prove the table still exists
    _, rows = run_query(conn, "SELECT COUNT(*) FROM fact_incident")
    print(f"  fact_incident still intact -> {rows[0][0]:,} rows (no injection)")


def demo_queries(conn):
    print("\n=== Q1. Operator league table (>= 5000 incidents) ===")
    cols, rows = run_query(conn, """
        SELECT o.operator, COUNT(*) incidents,
               ROUND(AVG(f.is_breakdown),3) breakdown_rate,
               ROUND(AVG(f.delay_minutes),1) avg_delay
        FROM fact_incident f JOIN dim_operator o ON o.operator_id=f.operator_id
        GROUP BY o.operator HAVING COUNT(*) >= ?
        ORDER BY breakdown_rate DESC LIMIT 8""", (5000,))
    print(" | ".join(cols))
    for r in rows:
        print("  ", r)

    print("\n=== Q2. Breakdowns by borough in 2019 ===")
    cols, rows = run_query(conn, """
        SELECT b.boro, COUNT(*) breakdowns
        FROM fact_incident f JOIN dim_boro b ON b.boro_id=f.boro_id
        JOIN dim_date d ON d.date_key=f.date_key
        WHERE f.is_breakdown=1 AND d.year=?
        GROUP BY b.boro ORDER BY breakdowns DESC LIMIT 6""", (2019,))
    for r in rows:
        print("  ", r)


def export_sample_dump(conn):
    """Write a git-friendly dump: schema + dims + 5000 fact rows."""
    out = DATABASE / "bus_reliability_sample_dump.sql"
    with open(out, "w") as fh:
        for line in conn.iterdump():
            if line.startswith("INSERT INTO \"fact_incident\"") or \
               line.startswith("INSERT INTO fact_incident"):
                continue  # skip full fact insert (too large)
            fh.write(line + "\n")
        # append only a 5000-row fact sample
        cur = conn.cursor()
        cur.execute("SELECT * FROM fact_incident LIMIT 5000")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            vals = ",".join("NULL" if v is None else
                            (f"'{str(v).replace(chr(39), chr(39)*2)}'" if isinstance(v, str)
                             else str(v)) for v in row)
            fh.write(f"INSERT INTO fact_incident ({','.join(cols)}) VALUES ({vals});\n")
    print(f"[db] wrote git-friendly sample dump -> {out.name} "
          f"({out.stat().st_size/1024:.0f} KB)")


def main():
    spark = get_spark("Stage3-Database")
    conn = build_database(spark)
    demo_queries(conn)
    security_demo(conn)
    export_sample_dump(conn)

    # DB size + row count summary
    cur = conn.cursor()
    n = cur.execute("SELECT COUNT(*) FROM fact_incident").fetchone()[0]
    print(f"\n[db] SQLite ready: {SQLITE_DB.name} "
          f"({SQLITE_DB.stat().st_size/1e6:.1f} MB, {n:,} fact rows)")
    conn.close()
    spark.stop()


if __name__ == "__main__":
    main()
