# Databricks notebook source
# MAGIC %pip install --quiet databricks-sdk mlflow-skinny --upgrade

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorMetric,
    MonitorMetricType,
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
    MonitorCronSchedule,
    MonitorInfoStatus,
    MonitorRefreshInfoState,
)

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StructField


# COMMAND ----------

catalog = "bank"
schema  = "gold"

source_predictions_table = "bank.gold.subscription_predictions"
monitoring_table         = "bank.gold.subscription_inference_monitoring"
baseline_table           = "bank.gold.subscription_monitoring_baseline"

print("Source predictions table:", source_predictions_table)
print("Monitoring table        :", monitoring_table)
print("Baseline table          :", baseline_table)


# COMMAND ----------

pred_df = spark.table(source_predictions_table)

print("Prediction table rows   :", pred_df.count())
print("Prediction table columns:", pred_df.columns)

display(pred_df.limit(10))


# COMMAND ----------

# FIX: removed id_col="customer_id" — that column does not exist in
#      the source table and is not needed by MonitorInferenceLog.
prediction_col    = "prediction"
label_col         = "y_flag"
timestamp_col     = "inference_timestamp"
model_version_col = "model_version"
source_timestamp_col = "scored_at"


# COMMAND ----------

# FIX: spread inference_timestamp across the last 28 days.
# Databricks Lakehouse Monitoring only looks back 30 days for inference
# logs. The original scored_at values in a static training dataset are
# months old, so the monitor refresh found zero rows and produced empty
# metric tables. We replace them with synthetic recent timestamps.
monitoring_df = pred_df.withColumn(
    "inference_timestamp",
    (
        F.unix_timestamp(F.current_timestamp())
        - (F.rand(seed=42) * 28 * 86400)   # random offset 0–28 days
    ).cast("timestamp")
)

# Cast columns to the types Lakehouse Monitoring expects.
monitoring_df = monitoring_df.withColumn("prediction",    F.col("prediction").cast("int"))
monitoring_df = monitoring_df.withColumn("y_flag",        F.col("y_flag").cast("int"))
monitoring_df = monitoring_df.withColumn("model_version", F.col("model_version").cast("string"))

# Write ONCE — the original notebook wrote this table twice (cells 6+7).
(
    monitoring_df.write
    .mode("overwrite")
    .format("delta")
    .option("overwriteSchema", "true")
    .saveAsTable(monitoring_table)
)

print("Monitoring table written:", monitoring_table)
print("Rows   :", monitoring_df.count())
print("Columns:", len(monitoring_df.columns))

# Verify timestamps are recent — all rows must fall within last 30 days.
print("\nTimestamp range (must be within last 30 days):")
spark.sql(
    f"SELECT MIN(inference_timestamp) AS oldest,"
    f"       MAX(inference_timestamp) AS newest"
    f" FROM {monitoring_table}"
).show(truncate=False)

display(monitoring_df.limit(10))


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Change Data Feed is required for Lakehouse Monitoring to track
# MAGIC -- row-level changes between refreshes.
# MAGIC ALTER TABLE bank.gold.subscription_inference_monitoring
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Baseline: 20 % sample, without the raw scored_at column.
# MAGIC -- inference_timestamp is kept because it was cast from scored_at.
# MAGIC CREATE OR REPLACE TABLE bank.gold.subscription_monitoring_baseline AS
# MAGIC SELECT * EXCEPT (scored_at)
# MAGIC FROM bank.gold.subscription_inference_monitoring
# MAGIC WHERE rand(42) < 0.2;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS baseline_rows
# MAGIC FROM bank.gold.subscription_monitoring_baseline;
# MAGIC

# COMMAND ----------

baseline_df = spark.table(baseline_table)
print("Baseline rows:", baseline_df.count())
display(baseline_df.limit(10))


# COMMAND ----------

# Custom business metric: average missed revenue on false negatives.
# False negative = model predicted 0 (no subscription) but actual = 1.
# FIX: name uses underscore ("missed_column_metric"), NOT a space.
#      Databricks uses this name as a SQL column name in the output
#      metrics table — spaces in SQL identifiers break the refresh job.
missed_subscription_loss_metrics = [
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="missed_column_metric",          # ← underscore, not space
        input_columns=[":table"],
        definition="""
        avg(
            CASE
                WHEN CAST({{prediction_col}} AS INT) != CAST({{label_col}} AS INT)
                     AND CAST({{label_col}} AS INT) = 1
                THEN -GREATEST(CAST(balance AS DOUBLE), 0.0)
                ELSE 0.0
            END
        )
        """,
        output_data_type=StructField("output", DoubleType()).json()
    )
]

print("Custom metrics defined:", [m.name for m in missed_subscription_loss_metrics])


# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bank.monitoring")
print("Output schema ready: bank.monitoring")


# COMMAND ----------

w = WorkspaceClient()
print(f"WorkspaceClient ready. Target table: {monitoring_table}")

# FIX: always delete any existing monitor first so the fresh settings
# (corrected metric name, updated slicing expressions, etc.) are applied.
# Skipping this step causes create() to silently reuse the old broken
# configuration, which is why refreshes kept failing.
try:
    w.quality_monitors.delete(table_name=monitoring_table)
    print("Deleted existing monitor — will recreate with correct settings.")
    time.sleep(5)   # brief pause so deletion propagates
except Exception:
    print("No existing monitor found — proceeding to create.")

monitor_info = w.quality_monitors.create(
    table_name=monitoring_table,
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        timestamp_col="inference_timestamp",
        prediction_col="prediction",
        granularities=["1 day"],
        model_id_col="model_version",
        label_col="y_flag",
    ),
    schedule=MonitorCronSchedule(
        quartz_cron_expression="0 0 12 * * ?",
        timezone_id="UTC",
    ),
    assets_dir=f"{os.getcwd()}/monitoring",
    output_schema_name="bank.monitoring",
    baseline_table_name=baseline_table,
    slicing_exprs=[
        "job",
        "balance_tier",
        "duration_bucket",
        "financial_stress = 1",
        "was_previously_contacted = 1",
    ],
    custom_metrics=missed_subscription_loss_metrics,
)

print("Monitor created successfully.")
print("  Profile table:", monitor_info.profile_metrics_table_name)
print("  Drift table  :", monitor_info.drift_metrics_table_name)
monitor_info


# COMMAND ----------

while monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
    print("Monitor pending — waiting 10 s...")
    time.sleep(10)
    monitor_info = w.quality_monitors.get(table_name=monitoring_table)

print("Monitor status:", monitor_info.status)
assert monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, \
    f"Unexpected status: {monitor_info.status}"


# COMMAND ----------

# MAGIC %sql
# MAGIC -- Expected: 0 rows — tables are created only after a successful refresh.
# MAGIC SHOW TABLES IN bank.monitoring;
# MAGIC

# COMMAND ----------

# FIX: always force a NEW refresh instead of checking if one already
# exists. The original code used `if len(refreshes)==0` which meant
# a new refresh was never triggered when old (failed) ones were present.
# The same FAILED refresh ID appeared every run.
print("Triggering a fresh refresh...")
refresh_resp = w.quality_monitors.run_refresh(table_name=monitoring_table)
refresh_id   = refresh_resp.refresh_id
print(f"New Refresh ID: {refresh_id}")

terminal_states = {
    MonitorRefreshInfoState.SUCCESS,
    MonitorRefreshInfoState.FAILED,
    MonitorRefreshInfoState.CANCELED,
}

# Poll every 30 s until the refresh reaches a terminal state.
# A 45 k-row table typically takes 2–5 minutes on a small cluster.
while True:
    info = w.quality_monitors.get_refresh(
        table_name=monitoring_table,
        refresh_id=refresh_id,
    )
    print(f"  Refresh state: {info.state}")
    if info.state in terminal_states:
        break
    time.sleep(30)

print(f"\nFinal state: {info.state}")

if info.state == MonitorRefreshInfoState.FAILED:
    print("Refresh FAILED. Exact error message:")
    print(info.message)          # prints the SQL error from the job
    raise RuntimeError("Monitor refresh failed — see message above.")
elif info.state == MonitorRefreshInfoState.SUCCESS:
    print("Refresh succeeded! Metric tables are now populated.")


# COMMAND ----------

monitor_details = w.quality_monitors.get(table_name=monitoring_table)
print("monitor_version:", monitor_details.monitor_version)
# monitor_version should now be 1 (increments on every successful refresh)
monitor_details


# COMMAND ----------

# MAGIC %sql
# MAGIC -- After a successful refresh this should return 2 rows:
# MAGIC --   subscription_inference_monitoring_drift_metrics
# MAGIC --   subscription_inference_monitoring_profile_metrics
# MAGIC SHOW TABLES IN bank.monitoring;
# MAGIC

# COMMAND ----------

profile_table = monitor_details.profile_metrics_table_name
print("Sampling profile metrics from:", profile_table)
display(spark.table(profile_table).limit(20))


# COMMAND ----------

drift_table = monitor_details.drift_metrics_table_name
print("Sampling drift metrics from:", drift_table)
display(spark.table(drift_table).limit(20))
