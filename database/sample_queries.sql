-- =============================================================
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
