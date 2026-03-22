-- Item-level performance + recency features.
CREATE OR REPLACE TABLE mart.item_features AS
WITH max_ts AS (
  SELECT MAX(event_ts) AS as_of_ts FROM interactions
), recent AS (
  SELECT i.*
  FROM interactions i
  CROSS JOIN max_ts
  WHERE i.event_ts >= as_of_ts - INTERVAL '14 days'
), performance AS (
  SELECT
    item_id,
    COUNT(*) AS impressions_14d,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks_14d,
    SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END) AS conversions_14d,
    AVG(dwell_seconds) AS avg_dwell_seconds,
    MAX(event_ts) AS last_event_ts
  FROM recent
  GROUP BY 1
)
SELECT
  it.item_id,
  it.category,
  it.price_bucket,
  it.language,
  COALESCE(p.impressions_14d, 0) AS impressions_14d,
  COALESCE(p.clicks_14d, 0) AS clicks_14d,
  COALESCE(p.conversions_14d, 0) AS conversions_14d,
  COALESCE(p.clicks_14d * 1.0 / NULLIF(p.impressions_14d, 0), 0.0) AS ctr_14d,
  COALESCE(p.conversions_14d * 1.0 / NULLIF(p.clicks_14d, 0), 0.0) AS cvr_14d,
  COALESCE(p.avg_dwell_seconds, 0.0) AS avg_dwell_seconds,
  COALESCE(DATE_DIFF('hour', p.last_event_ts, m.as_of_ts), 9999) AS hours_since_last_seen
FROM items it
LEFT JOIN performance p USING (item_id)
CROSS JOIN max_ts m;
