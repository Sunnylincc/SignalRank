CREATE OR REPLACE TABLE mart.user_features AS
WITH base AS (
  SELECT
    user_id,
    COUNT(*) AS impressions_30d,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS clicks_30d,
    SUM(CASE WHEN event_type = 'conversion' THEN 1 ELSE 0 END) AS conversions_30d,
    MAX(event_ts) AS last_event_ts,
    AVG(session_depth) AS avg_session_depth
  FROM interactions
  WHERE event_ts >= NOW() - INTERVAL '30 days'
  GROUP BY 1
)
SELECT
  user_id,
  impressions_30d,
  clicks_30d,
  conversions_30d,
  CASE WHEN impressions_30d > 0 THEN clicks_30d * 1.0 / impressions_30d ELSE 0 END AS ctr_30d,
  CASE WHEN clicks_30d > 0 THEN conversions_30d * 1.0 / clicks_30d ELSE 0 END AS cvr_30d,
  DATE_DIFF('hour', last_event_ts, NOW()) AS hours_since_last_event,
  avg_session_depth
FROM base;
