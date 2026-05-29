# Databricks notebook source
# MAGIC %pip install --quiet databricks-sdk mlflow-skinny --upgrade dbldatagen
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.dropdown(
    "perf_metric",
    "f1_score.macro",
    [
        "accuracy_score",
        "precision.weighted",
        "recall.weighted",
        "f1_score.macro"
    ]
)

dbutils.widgets.dropdown(
    "drift_metric",
    "js_distance",
    [
        "chi_squared_test.statistic",
        "chi_squared_test.pvalue",
        "tv_distance",
        "l_infinity_distance",
        "js_distance"
    ]
)

dbutils.widgets.text("model_id", "*", "Model Id")

# COMMAND ----------

import time
from datetime import datetime, timedelta

import dbldatagen as dg

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorRefreshInfoState

from pyspark.sql import functions as F
from pyspark.sql.functions import col, abs, first
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, StringType

# COMMAND ----------

monitoring_table = "bank.gold.subscription_inference_monitoring"
baseline_table = "bank.gold.subscription_monitoring_baseline"

print("Monitoring table:", monitoring_table)
print("Baseline table:", baseline_table)

# COMMAND ----------

monitoring_df = spark.table(monitoring_table)

print("Monitoring table rows:", monitoring_df.count())
print("Monitoring table columns:")
print(monitoring_df.columns)

display(monitoring_df.limit(10))

# COMMAND ----------


    print("Simulating drift data...")

    base_df = spark.table(monitoring_table)

    sample_size = 5000

    sample_df = (
        base_df
        .limit(sample_size)
        .withColumn("row_num", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
    )

    drift_generator = (
        dg.DataGenerator(
            sparkSession=spark,
            name="bank_drift_generator",
            rows=sample_size,
            partitions=4
        )
        .withColumn(
            "row_num",
            IntegerType(),
            minValue=1,
            maxValue=sample_size,
            uniqueValues=sample_size
        )
        .withColumn(
            "forced_y_flag",
            IntegerType(),
            values=[1, 0],
            weights=[8, 2]
        )
        .withColumn(
            "forced_prediction",
            IntegerType(),
            values=[1, 0],
            weights=[7, 3]
        )
    )

    drift_control_df = drift_generator.build()

    max_ts = base_df.select(F.max("inference_timestamp").alias("max_ts")).collect()[0]["max_ts"]

    if max_ts is None:
        drift_ts_expr = F.current_timestamp()
    else:
        drift_ts_expr = F.expr("current_timestamp() + INTERVAL 1 DAY")

    drift_df = (
        sample_df
        .join(drift_control_df, on="row_num", how="inner")
        .drop("row_num")
        .withColumn("y_flag", F.col("forced_y_flag").cast("int"))
        .withColumn("prediction", F.col("forced_prediction").cast("int"))
        .drop("forced_y_flag", "forced_prediction")
        .withColumn("inference_timestamp", drift_ts_expr)
        .withColumn("model_version", F.col("model_version").cast("string"))
    )

    (
        drift_df.write
        .mode("append")
        .format("delta")
        .saveAsTable(monitoring_table)
    )

    print("Appended drift rows:", drift_df.count())


# COMMAND ----------

w = WorkspaceClient()

print("Triggering monitor refresh for:", monitoring_table)

refresh_info = w.quality_monitors.run_refresh(
    table_name=monitoring_table
)

print("Refresh triggered:")
print(refresh_info)

# COMMAND ----------

# Sometimes refresh_info may not immediately contain state in all runtimes.
# So we fetch latest refresh from list_refreshes.

def get_latest_refresh(table_name: str):
    refreshes = w.quality_monitors.list_refreshes(
        table_name=table_name
    ).refreshes or []

    if len(refreshes) == 0:
        return None

    return refreshes[0]


latest_refresh = get_latest_refresh(monitoring_table)

if latest_refresh is None:
    print("No refresh found yet. Waiting 10 seconds...")
    time.sleep(10)
    latest_refresh = get_latest_refresh(monitoring_table)

if latest_refresh is None:
    raise RuntimeError("No monitor refresh found after triggering refresh.")

refresh_id = latest_refresh.refresh_id

print("Tracking refresh ID:", refresh_id)
print("Initial state:", latest_refresh.state)

# COMMAND ----------

while True:
    run_info = w.quality_monitors.get_refresh(
        table_name=monitoring_table,
        refresh_id=refresh_id
    )

    print("Refresh state:", run_info.state)

    if run_info.state not in (
        MonitorRefreshInfoState.PENDING,
        MonitorRefreshInfoState.RUNNING,
    ):
        break

    time.sleep(60)

print("Final refresh info:")
print(run_info)

if run_info.state != MonitorRefreshInfoState.SUCCESS:
    raise RuntimeError(f"Monitor refresh failed: {run_info}")

# COMMAND ----------

monitor_info = w.quality_monitors.get(
    table_name=monitoring_table
)

profile_table_name = monitor_info.profile_metrics_table_name
drift_table_name = monitor_info.drift_metrics_table_name

print("Profile metrics table:", profile_table_name)
print("Drift metrics table:", drift_table_name)