-- Feature coverage, sparsity, and distribution checks for model-readiness.
CREATE OR REPLACE TABLE mart.feature_diagnostics AS
WITH base AS (
  SELECT * FROM mart.training_examples
)
SELECT 'ctr_30d' AS feature_name,
       AVG(CASE WHEN ctr_30d IS NULL THEN 1 ELSE 0 END) AS null_rate,
       MIN(ctr_30d) AS min_value,
       APPROX_QUANTILE(ctr_30d, 0.5) AS median,
       APPROX_QUANTILE(ctr_30d, 0.95) AS p95,
       MAX(ctr_30d) AS max_value
FROM base
UNION ALL
SELECT 'hours_since_last_event',
       AVG(CASE WHEN hours_since_last_event IS NULL THEN 1 ELSE 0 END),
       MIN(hours_since_last_event),
       APPROX_QUANTILE(hours_since_last_event, 0.5),
       APPROX_QUANTILE(hours_since_last_event, 0.95),
       MAX(hours_since_last_event)
FROM base
UNION ALL
SELECT 'avg_dwell_seconds',
       AVG(CASE WHEN avg_dwell_seconds IS NULL THEN 1 ELSE 0 END),
       MIN(avg_dwell_seconds),
       APPROX_QUANTILE(avg_dwell_seconds, 0.5),
       APPROX_QUANTILE(avg_dwell_seconds, 0.95),
       MAX(avg_dwell_seconds)
FROM base;

CREATE OR REPLACE TABLE mart.category_coverage AS
SELECT
  category,
  COUNT(*) AS rows,
  AVG(label) AS positive_rate
FROM mart.training_examples
GROUP BY 1
ORDER BY rows DESC;
