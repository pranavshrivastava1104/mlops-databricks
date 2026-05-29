# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

# Databricks notebook source

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("model_name", "bank.models.subscription_classifier")
dbutils.widgets.text("model_version", "5")
dbutils.widgets.text("source_table", "bank.silver.data_cleaned")
dbutils.widgets.text("output_table", "bank.gold.subscription_predictions")
dbutils.widgets.dropdown("smoke_test", "false", ["true", "false"])

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
source_table = dbutils.widgets.get("source_table")
output_table = dbutils.widgets.get("output_table")
smoke_test = dbutils.widgets.get("smoke_test").lower() == "true"

print("Model name:", model_name)
print("Model version to deploy:", model_version)
print("Source table:", source_table)
print("Output table:", output_table)
print("Smoke test:", smoke_test)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

if not model_version:
    raise ValueError("model_version parameter is required for deployment.")

# COMMAND ----------

# In deployment, we move the champion alias to the approved model version.
if smoke_test:
    print("Smoke test enabled. Not moving champion alias.")
else:
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="phase",
        value="champion"
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="deployment_status",
        value="deployed"
    )

    print(f"Moved champion alias to version {model_version}")

# COMMAND ----------

# Always load by alias after deployment.
# This is the production-safe pattern.
model_uri = f"models:/{model_name}@champion"

print("Loading model from:", model_uri)

# COMMAND ----------

# Try to get model input columns from model signature.
# This avoids manually hardcoding feature columns if the signature exists.
try:
    model_info = mlflow.models.get_model_info(model_uri)

    feature_cols = [
        input_spec.name
        for input_spec in model_info.signature.inputs.inputs
        if input_spec.name is not None
    ]

    if len(feature_cols) == 0:
        raise ValueError("Model signature exists but has no named input columns.")

    print("Feature columns loaded from model signature:")
    print(feature_cols)

except Exception as e:
    print("Could not infer feature columns from model signature.")
    print("Falling back to known Bank Marketing engineered features.")
    print(str(e)[:500])

    feature_cols = [
        "age_group",
        "balance_tier",
        "contact_intensity",
        "was_previously_contacted",
        "recency_score",
        "duration_bucket",
        "financial_stress"
    ]

# COMMAND ----------

source_df = spark.table(source_table)

print("Source row count:", source_df.count())
print("Source columns:", source_df.columns)

missing_features = [
    c for c in feature_cols
    if c not in source_df.columns
]

if missing_features:
    raise ValueError(
        f"Source table {source_table} is missing model input columns: {missing_features}"
    )

display(source_df.limit(5))

# COMMAND ----------

champion_udf = mlflow.pyfunc.spark_udf(
    spark=spark,
    model_uri=model_uri,
    result_type="double"
)

# COMMAND ----------

preds_df = source_df.withColumn(
    "prediction",
    champion_udf(*[F.col(c) for c in feature_cols])
)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bank.gold")

prediction_output_df = (
    preds_df
    .withColumn("model_name", F.lit(model_name))
    .withColumn("model_alias", F.lit("champion"))
    .withColumn("model_version", F.lit(str(model_version)))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

# Keep all source columns dynamically.
# Do not manually hardcode 40-50 business/feature columns.
metadata_cols = [
    "prediction",
    "model_name",
    "model_alias",
    "model_version",
    "scored_at"
]

final_cols = [
    c for c in prediction_output_df.columns
    if c not in metadata_cols
] + metadata_cols

prediction_output_df = prediction_output_df.select(*final_cols)

display(prediction_output_df.limit(10))

# COMMAND ----------

if smoke_test:
    print("Smoke test enabled. Not writing output table.")
    print("Smoke test row count:", prediction_output_df.count())
else:
    (
        prediction_output_df.write
        .mode("overwrite")
        .format("delta")
        .option("overwriteSchema", "true")
        .saveAsTable(output_table)
    )

    print("Saved predictions to:", output_table)
    print("Rows:", prediction_output_df.count())

# COMMAND ----------

dbutils.jobs.taskValues.set(
    key="deployed_model_version",
    value=str(model_version)
)

dbutils.jobs.taskValues.set(
    key="prediction_output_table",
    value=output_table
)

print("Deployment completed.")