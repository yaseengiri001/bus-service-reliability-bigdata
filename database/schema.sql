
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
