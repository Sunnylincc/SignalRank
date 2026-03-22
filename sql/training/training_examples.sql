-- Pointwise ranking labels plus retrieval training positives.
CREATE OR REPLACE TABLE mart.training_examples AS
WITH max_ts AS (
  SELECT MAX(event_ts) AS as_of_ts FROM interactions
), base AS (
  SELECT
    i.user_id,
    i.item_id,
    i.event_ts,
    i.surface,
    i.country,
    CASE WHEN i.event_type IN ('click', 'conversion') THEN 1 ELSE 0 END AS label,
    CASE WHEN i.event_type = 'conversion' THEN 1 ELSE 0 END AS conversion_label
  FROM interactions i
  CROSS JOIN max_ts
  WHERE i.event_ts >= as_of_ts - INTERVAL '30 days'
)
SELECT
  b.user_id,
  b.item_id,
  b.event_ts,
  b.surface,
  b.country,
  b.label,
  b.conversion_label,
  uf.ctr_30d,
  uf.cvr_30d,
  uf.avg_session_depth,
  uf.hours_since_last_event,
  it.ctr_14d,
  it.cvr_14d,
  it.avg_dwell_seconds,
  it.hours_since_last_seen,
  it.category,
  it.price_bucket,
  it.language
FROM base b
LEFT JOIN mart.user_features uf USING (user_id)
LEFT JOIN mart.item_features it USING (item_id);

CREATE OR REPLACE TABLE mart.retrieval_pairs AS
SELECT DISTINCT
  user_id,
  item_id,
  1 AS positive_label
FROM mart.training_examples
WHERE label = 1;
