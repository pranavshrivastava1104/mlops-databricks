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
schema = "gold"

source_predictions_table = "bank.gold.subscription_predictions"
monitoring_table = "bank.gold.subscription_inference_monitoring"
baseline_table = "bank.gold.subscription_monitoring_baseline"

print("Source predictions table:", source_predictions_table)
print("Monitoring table:", monitoring_table)
print("Baseline table:", baseline_table)

# COMMAND ----------

pred_df = spark.table(source_predictions_table)

print("Prediction table rows:", pred_df.count())
print("Prediction table columns:")
print(pred_df.columns)

display(pred_df.limit(10))

# COMMAND ----------

# id_col = "customer_id"
prediction_col = "prediction"
label_col = "y_flag"
timestamp_col = "inference_timestamp"
model_version_col = "model_version"


source_timestamp_col = "scored_at"

# COMMAND ----------

monitoring_df = pred_df.withColumn(
    "inference_timestamp",
    F.col("scored_at").cast("timestamp")
)

# Make sure prediction is integer for classification monitoring.
monitoring_df = monitoring_df.withColumn(
    "prediction",
    F.col("prediction").cast("int")
)

# Make sure actual label is integer.
monitoring_df = monitoring_df.withColumn(
    "y_flag",
    F.col("y_flag").cast("int")
)

# Make sure model version is string because monitoring groups by model_id column.
monitoring_df = monitoring_df.withColumn(
    "model_version",
    F.col("model_version").cast("string")
)

# Write monitoring-ready table.
(
    monitoring_df.write
    .mode("overwrite")
    .format("delta")
    .option("overwriteSchema", "true")
    .saveAsTable(monitoring_table)
)

print("Created monitoring table:", monitoring_table)
print("Rows:", monitoring_df.count())
print("Columns:", len(monitoring_df.columns))

display(monitoring_df.limit(10))

# COMMAND ----------

# monitoring_df.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(
#     monitoring_table
# )

# print("Created monitoring table:", monitoring_table)
# print("Rows:", monitoring_df.count())

# display(monitoring_df.limit(10))

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE bank.gold.subscription_inference_monitoring
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# %sql
# MERGE INTO bank.gold.subscription_inference_monitoring AS target
# USING (
#     SELECT customer_id,y_flag
#     FROM bank.gold.subscription_inference_monitoring
#     WHERE y_flag IS NOT NULL
# ) AS source
# ON target.customer_id = source.customer_id
# WHEN MATCHED THEN UPDATE SET target.y_flag=source.y_flag

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bank.gold.subscription_monitoring_baseline AS
# MAGIC SELECT * EXCEPT (scored_at)
# MAGIC FROM bank.gold.subscription_inference_monitoring
# MAGIC WHERE rand(42) < 0.2;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) AS baseline_rows
# MAGIC FROM bank.gold.subscription_monitoring_baseline;

# COMMAND ----------

baseline_df = spark.table(baseline_table)

print("Baseline rows:", baseline_df.count())

display(baseline_df.limit(10))

# COMMAND ----------

#defining custome business metrics like 
# like false negatives : model predictred = 0 but actual = 1 , 
#business risk : person actally defaulted but model gives wrong. 

missed_subscription_loss_metrics=[
    MonitorMetric(
        type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
        name="missed_column_metric",
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

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bank.monitoring")

# COMMAND ----------

# Add this before create()
try:
    w.quality_monitors.delete(table_name=monitoring_table)
    print("Deleted old monitor.")
except:
    pass
# Then call w.quality_monitors.create(...)

# COMMAND ----------

w = WorkspaceClient()
print("creating workspace client for inference table {monitoring table}")
try:
    monitor_info=w.quality_monitors.create(
        table_name=monitoring_table,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            timestamp_col="inference_timestamp",
            prediction_col="prediction",
            granularities=["1 day"],
            model_id_col="model_version",
            label_col="y_flag"
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
        custom_metrics=missed_subscription_loss_metrics
    )
    print("Monitor creation submitted.")
except Exception as monitor_exception:
    if "already exist" in str(monitor_exception).lower():
        print("Monitor already exists. Retrieving existing monitor info.")

        monitor_info = w.quality_monitors.get(
            table_name=monitoring_table
        )

    else:
        raise monitor_exception
monitor_info 

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN bank.monitoring;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN bank.monitoring LIKE '*drift*';

# COMMAND ----------

while monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
    print("Monitor is pending. Waiting...")
    time.sleep(10)

    monitor_info = w.quality_monitors.get(
        table_name=monitoring_table
    )

print("Monitor status:", monitor_info.status)

assert monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Error creating monitor"

# COMMAND ----------

def get_refreshes():
    return w.quality_monitors.list_refreshes(
        table_name=monitoring_table,
    ).refreshes

# COMMAND ----------

refreshes=get_refreshes()
if len(refreshes)==0:
    print("not running")
    w.quality_monitors.run_refresh(
        table_name=monitoring_table
    )
    time.sleep(5)
    refreshes=get_refreshes()
    print("again runnned")
run_info=refreshes[0]
print("Refresh ID:", run_info.refresh_id)
print("Initial refresh state:", run_info.state)


# COMMAND ----------

refresh_detail = w.quality_monitors.get_refresh(
    table_name=monitoring_table,
    refresh_id=run_info.refresh_id
)
print("Failure message:", refresh_detail.message)

# COMMAND ----------

monitor_details = w.quality_monitors.get(
    table_name=monitoring_table
)

monitor_details

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN bank.monitoring;