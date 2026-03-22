CREATE OR REPLACE TABLE mart.feature_diagnostics AS
SELECT 'ctr_30d' AS feature_name,
       AVG(CASE WHEN ctr_30d IS NULL THEN 1 ELSE 0 END) AS null_rate,
       APPROX_QUANTILE(ctr_30d, 0.5) AS median,
       APPROX_QUANTILE(ctr_30d, 0.95) AS p95
FROM mart.training_examples
UNION ALL
SELECT 'hours_since_last_event' AS feature_name,
       AVG(CASE WHEN hours_since_last_event IS NULL THEN 1 ELSE 0 END) AS null_rate,
       APPROX_QUANTILE(hours_since_last_event, 0.5) AS median,
       APPROX_QUANTILE(hours_since_last_event, 0.95) AS p95
FROM mart.training_examples
UNION ALL
SELECT 'avg_dwell_seconds' AS feature_name,
       AVG(CASE WHEN avg_dwell_seconds IS NULL THEN 1 ELSE 0 END) AS null_rate,
       APPROX_QUANTILE(avg_dwell_seconds, 0.5) AS median,
       APPROX_QUANTILE(avg_dwell_seconds, 0.95) AS p95
FROM mart.training_examples;
