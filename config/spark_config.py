"""
Central SparkSession factory for the Bus Service Reliability pipeline.

Design notes (mapped to the brief's "Big Data Evidence" requirement)
--------------------------------------------------------------------
* ``spark.sql.shuffle.partitions = 8``  -> exceeds the mandated >= 4 partitions
  and makes shuffle-stage parallelism visible in the Spark UI.
* Adaptive Query Execution (AQE) is enabled so the DAG is optimised at runtime;
  we deliberately disable *coalescePartitions* so the configured partition count
  stays observable in the UI instead of being auto-collapsed to 1.
* Arrow is enabled so the *single* Spark -> Pandas hand-off (final plotting /
  dashboard step) is columnar and fast, per the tool-choice justification.
* Java 17 module flags (``--add-opens``) are injected defensively so Spark 3.5
  runs cleanly on the Homebrew JDK 17 present on the build machine.

All configuration is centralised here (no hard-coded values scattered across
scripts) so it can be reported in the README and audited for security.
"""
from __future__ import annotations

import os
import subprocess
from pyspark.sql import SparkSession

# --- number of shuffle/output partitions used throughout the pipeline --------
SHUFFLE_PARTITIONS = 8            # > 4  (brief requirement)

# --- Java 17 compatibility: Spark 3.5 needs these module opens ---------------
_JAVA17_MODULE_OPTS = " ".join(
    f"--add-opens=java.base/{pkg}=ALL-UNNAMED"
    for pkg in (
        "java.lang", "java.lang.invoke", "java.lang.reflect", "java.io",
        "java.net", "java.nio", "java.util", "java.util.concurrent",
        "java.util.concurrent.atomic", "sun.nio.ch", "sun.nio.cs",
        "sun.security.action", "sun.util.calendar",
    )
)


def _ensure_java_home() -> None:
    """Locate a JDK if JAVA_HOME is unset (Homebrew JDK 17 on this machine)."""
    if os.environ.get("JAVA_HOME"):
        return
    # Check the common Homebrew JDK locations first (no error output).
    candidates = [
        "/opt/homebrew/opt/openjdk@17",
        "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        "/opt/homebrew/opt/openjdk",
        "/usr/local/opt/openjdk@17",
    ]
    for c in candidates:
        if os.path.isdir(c):
            os.environ["JAVA_HOME"] = c
            return
    # Fall back to the macOS JDK locator only if nothing above exists.
    try:
        jh = subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if jh and os.path.isdir(jh):
            os.environ["JAVA_HOME"] = jh
    except Exception:
        pass


def get_spark(app_name: str = "BusServiceReliability",
              shuffle_partitions: int = SHUFFLE_PARTITIONS) -> SparkSession:
    """Build (or fetch) the project SparkSession with tuned, auditable config."""
    _ensure_java_home()

    spark = (
        SparkSession.builder
        .appName(app_name)
        # local[*] uses every core on the driver machine (single-node cluster).
        .master("local[*]")
        # ---- partitioning / parallelism (Big Data Evidence) -----------------
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.default.parallelism", shuffle_partitions)
        # ---- Adaptive Query Execution (keep partition count observable) -----
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        # ---- Arrow for the single Spark -> Pandas plotting hand-off ---------
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        # ---- Proleptic-Gregorian rebase: tolerate rare pre-1900 bad dates ---
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        # ---- resources ------------------------------------------------------
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        # ---- keep the Spark UI on a predictable port for screenshots --------
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        # ---- Java 17 module access ------------------------------------------
        .config("spark.driver.extraJavaOptions", _JAVA17_MODULE_OPTS)
        .config("spark.executor.extraJavaOptions", _JAVA17_MODULE_OPTS)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


if __name__ == "__main__":
    s = get_spark("ConfigSmokeTest")
    print("Spark version :", s.version)
    print("Master        :", s.sparkContext.master)
    print("Shuffle parts :", s.conf.get("spark.sql.shuffle.partitions"))
    print("Default parall:", s.conf.get("spark.default.parallelism"))
    print("JAVA_HOME     :", os.environ.get("JAVA_HOME"))
    print("Spark UI      :", s.sparkContext.uiWebUrl)
    s.stop()
