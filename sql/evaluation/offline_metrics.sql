CREATE OR REPLACE TABLE scored_eval_examples AS
SELECT
  surface,
  country,
  label,
  LEAST(0.99, GREATEST(0.01, 0.1 + 0.8 * RANDOM())) AS pred_score
FROM mart.training_examples;

CREATE OR REPLACE TABLE mart.offline_metric_slices AS
SELECT
  surface,
  country,
  COUNT(*) AS n,
  AVG(label) AS empirical_ctr,
  AVG(pred_score) AS avg_pred,
  AVG(CASE WHEN label = 1 THEN -LOG(GREATEST(pred_score, 1e-8)) ELSE -LOG(GREATEST(1 - pred_score, 1e-8)) END) AS logloss
FROM scored_eval_examples
GROUP BY 1, 2;
