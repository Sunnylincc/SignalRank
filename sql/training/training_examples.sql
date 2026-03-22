CREATE OR REPLACE TABLE mart.training_examples AS
WITH labeled AS (
  SELECT
    i.user_id,
    i.item_id,
    i.event_ts,
    i.surface,
    i.country,
    CASE WHEN i.event_type IN ('click', 'conversion') THEN 1 ELSE 0 END AS label,
    CASE WHEN i.event_type = 'conversion' THEN 1 ELSE 0 END AS conversion_label
  FROM interactions i
  WHERE i.event_ts >= NOW() - INTERVAL '30 days'
), joined AS (
  SELECT
    l.user_id,
    l.item_id,
    l.event_ts,
    l.surface,
    l.country,
    l.label,
    l.conversion_label,
    uf.ctr_30d,
    uf.cvr_30d,
    uf.hours_since_last_event,
    it.ctr_14d,
    it.cvr_14d,
    it.avg_dwell_seconds,
    it.category,
    it.price_bucket,
    it.language
  FROM labeled l
  LEFT JOIN mart.user_features uf USING (user_id)
  LEFT JOIN mart.item_features it USING (item_id)
)
SELECT * FROM joined;
