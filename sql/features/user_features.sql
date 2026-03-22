-- User-level behavioral features used by retrieval and ranking.
CREATE OR REPLACE TABLE mart.user_features AS
WITH max_ts AS (
  SELECT MAX(event_ts) AS as_of_ts FROM interactions
), recent AS (
  SELECT i.*
  FROM interactions i
  CROSS JOIN max_ts
  WHERE i.event_ts >= as_of_ts - INTERVAL '30 days'
), by_user AS (
  SELECT
    user_id,
    COUNT(*) AS impressions_30d,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks_30d,
    SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END) AS conversions_30d,
    AVG(session_depth) AS avg_session_depth,
    MAX(event_ts) AS last_event_ts
  FROM recent
  GROUP BY 1
)
SELECT
  u.user_id,
  u.impressions_30d,
  u.clicks_30d,
  u.conversions_30d,
  COALESCE(u.clicks_30d * 1.0 / NULLIF(u.impressions_30d, 0), 0.0) AS ctr_30d,
  COALESCE(u.conversions_30d * 1.0 / NULLIF(u.clicks_30d, 0), 0.0) AS cvr_30d,
  u.avg_session_depth,
  DATE_DIFF('hour', u.last_event_ts, m.as_of_ts) AS hours_since_last_event
FROM by_user u
CROSS JOIN max_ts m;
