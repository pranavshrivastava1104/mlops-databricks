# Databricks notebook source
import mlflow
from mlflow import MlflowClient

from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,?_
mlflow.set_registry_uri("databricks-uc")
model_name="bank.models.subscription_classifier"
model_alias="champion"
model_uri=f"models:/{model_name}@{model_alias}"
print("model_uri:",model_uri)

# COMMAND ----------

client=MlflowClient()
champion_mode_details=client.get_model_version_by_alias(
    name=model_name,
    alias=model_alias
)

model_version=champion_mode_details.version
print(model_version)

# COMMAND ----------

silver_df = spark.table("bank.silver.data_cleaned")

print("Silver row count:", silver_df.count())

display(silver_df.limit(5))

# COMMAND ----------

feature_cols=[
    "age_group",
    "balance_tier",
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "duration_bucket",
    "financial_stress"
]

# COMMAND ----------

inference_df=silver_df.select(
    *feature_cols
)
display(inference_df.limit(10))
    

# COMMAND ----------

champion_udf=mlflow.pyfunc.spark_udf(
    spark=spark,
    model_uri=model_uri,
    result_type="double"
)

# COMMAND ----------

preds_df = silver_df.withColumn(
    "prediction",
    champion_udf(*[F.col(c) for c in feature_cols])
)

display(
    preds_df.select(
        "age",
        "job",
        "balance",
        "y_flag",
        *feature_cols,
        "prediction"
    ).limit(20)
)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bank.gold")

# COMMAND ----------

prediction_output_df = (
    preds_df
    .withColumn("model_name", F.lit(model_name))
    .withColumn("model_alias", F.lit(model_alias))
    .withColumn("model_version", F.lit(str(model_version)))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

# Select useful business columns plus prediction metadata.
# This is the table a business/analytics team can query.
prediction_output_df = prediction_output_df.select(
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "contact",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y_flag",
    *feature_cols,
    "prediction",
    "model_name",
    "model_alias",
    "model_version",
    "scored_at"
)

# COMMAND ----------

prediction_output_df.write.mode("overwrite").format("delta").saveAsTable(
    "bank.gold.subscription_predictions"
)

print("Saved predictions to bank.gold.subscription_predictions")

# COMMAND ----------

gold_df = spark.table("bank.gold.subscription_predictions")

print("Gold prediction row count:", gold_df.count())

display(gold_df.limit(10))

# COMMAND ----------

display(
    gold_df
    .groupBy("prediction")
    .count()
    .orderBy("prediction")
)

# COMMAND ----------

display(
    gold_df
    .groupBy("y_flag")
    .count()
    .orderBy("y_flag")
)