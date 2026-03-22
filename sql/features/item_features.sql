CREATE OR REPLACE TABLE mart.item_features AS
WITH engagement AS (
  SELECT
    item_id,
    COUNT(*) AS impressions_14d,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks_14d,
    SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END) AS conversions_14d,
    AVG(dwell_seconds) AS avg_dwell_seconds,
    MAX(event_ts) AS last_seen_ts
  FROM interactions
  WHERE event_ts >= NOW() - INTERVAL '14 days'
  GROUP BY 1
)
SELECT
  i.item_id,
  i.category,
  i.price_bucket,
  i.language,
  e.impressions_14d,
  e.clicks_14d,
  e.conversions_14d,
  COALESCE(e.clicks_14d * 1.0 / NULLIF(e.impressions_14d, 0), 0) AS ctr_14d,
  COALESCE(e.conversions_14d * 1.0 / NULLIF(e.clicks_14d, 0), 0) AS cvr_14d,
  e.avg_dwell_seconds,
  DATE_DIFF('hour', e.last_seen_ts, NOW()) AS hours_since_last_seen
FROM items i
LEFT JOIN engagement e ON i.item_id = e.item_id;
