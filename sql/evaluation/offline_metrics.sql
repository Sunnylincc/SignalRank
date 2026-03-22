-- Deterministic baseline score built from feature priors (for SQL-side evaluation demos).
CREATE OR REPLACE TABLE mart.scored_eval_examples AS
SELECT
  user_id,
  item_id,
  surface,
  country,
  label,
  LEAST(
    0.995,
    GREATEST(
      0.005,
      0.05
      + 0.45 * COALESCE(ctr_30d, 0.0)
      + 0.30 * COALESCE(ctr_14d, 0.0)
      + 0.15 * COALESCE(cvr_14d, 0.0)
      + 0.05 * CASE WHEN hours_since_last_seen <= 24 THEN 1 ELSE 0 END
    )
  ) AS pred_score
FROM mart.training_examples;

CREATE OR REPLACE TABLE mart.offline_metric_slices AS
SELECT
  surface,
  country,
  COUNT(*) AS rows,
  AVG(label) AS empirical_ctr,
  AVG(pred_score) AS avg_prediction,
  AVG(CASE WHEN label = 1 THEN -LOG(GREATEST(pred_score, 1e-12)) ELSE -LOG(GREATEST(1 - pred_score, 1e-12)) END) AS logloss,
  AVG(CASE WHEN (pred_score >= 0.5) = (label = 1) THEN 1 ELSE 0 END) AS accuracy_at_0_5
FROM mart.scored_eval_examples
GROUP BY 1, 2;
