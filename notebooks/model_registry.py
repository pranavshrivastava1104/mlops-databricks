# Databricks notebook source
import mlflow
from mlflow import MlflowClient

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_name="bank.models.subscription_classifier"
spark.sql("CREATE SCHEMA IF NOT EXISTS bank.models")
client=MlflowClient()

# COMMAND ----------

print(runs_df.columns.tolist())

# COMMAND ----------

# Get current Databricks user so we can locate the Phase 4 experiment.
current_user = spark.sql("SELECT current_user()").collect()[0][0]

# This should match the experiment path used in Phase 4.
experiment_name = f"/Users/pranavshrivastava1104@gmail.com/bank_subscription_experiment"

# Search all finished runs and sort by weighted F1.
# This gives the best model from Phase 4 without manually checking the UI.
runs_df = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.f1_weighted DESC"]
)

display(runs_df[[
    "run_id",
    "tags.mlflow.runName",
    "tags.model_type",
    "metrics.f1_score_wighted",
    "metrics.f1_score",
    "metrics.roc_auc"
]].head(10))

# COMMAND ----------

best_run_id = runs_df.iloc[0]["run_id"]
best_f1 = float(runs_df.iloc[0]["metrics.f1_score_wighted"])
best_run_name = runs_df.iloc[0]["tags.mlflow.runName"]

print("Best run ID:", best_run_id)
print("Best run name:", best_run_name)
print("Best weighted F1:", best_f1)

# COMMAND ----------

model_uri = f"runs:/{best_run_id}/model"
model_version_details=mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)
print("registered model version name:",model_version_details.name)
print("registered model version:",  model_version_details.version)

# COMMAND ----------

champion_version = model_version_details.version
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=champion_version
)

print(f"Champion: {model_name}@champion, version {champion_version}")

# COMMAND ----------

client.update_registered_model(
    name=model_name,
    description=(
        "Binary classifier predicting bank term deposit subscription. "
        "Trained on the UCI Bank Marketing dataset. "
        "Features include age group, balance tier, contact intensity, financial stress, "
        "recency score, duration bucket, and previous contact flag."
    )
)

# COMMAND ----------

client.update_model_version(
    name=model_name,
    version=champion_version,
    description=(
        f"Champion model version registered from MLflow run {best_run_id}. "
        f"Run name: {best_run_name}. Weighted F1: {best_f1:.4f}."
    )
)

# COMMAND ----------

champion_model_uri=f"models:/{model_name}@champion"
champion_model=mlflow.pyfunc.load_model(model_uri=model_uri)
print("Loaded champion model from:", champion_model_uri)

# COMMAND ----------

silver_df = spark.table("bank.silver.data_cleaned")

feature_cols = [
    "age_group",
    "balance_tier",
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "duration_bucket",
    "financial_stress"
]

# Convert a few rows to Pandas because pyfunc models usually accept Pandas input.
sample_input = silver_df.select(*feature_cols).limit(5).toPandas()

display(sample_input)

# COMMAND ----------

sample_predictions = champion_model.predict(sample_input)

print(sample_predictions)

# COMMAND ----------

challenger_run_id=runs_df.iloc[1]["run_id"]
challenger_f1=float(runs_df.iloc[1]["metrics.f1_score_wighted"])


print("Challenger run:", challenger_run_id)
print("Challenger F1:", challenger_f1)

# COMMAND ----------

challenger_uri=f"runs:/{challenger_run_id}/model"
challnger_details=mlflow.register_model(
    model_uri=challenger_uri,
    name=model_name
)
challenger_version=challnger_details.version
client.set_registered_model_alias(
    name=model_name,
    alias="challenger",
    version=challenger_version
)

client.update_model_version(
    name=model_name,
    version=challenger_version,
    description=f"Challenger version registered from run {challenger_run_id}. Weighted F1={challenger_f1:.4f}."
)

client.set_model_version_tag(model_name, challenger_version, "dataset", "bank-marketing-uci")
client.set_model_version_tag(model_name, challenger_version, "phase", "challenger")
client.set_model_version_tag(model_name, challenger_version, "training_f1_weighted", str(challenger_f1))

print(f"Challenger: {model_name}@challenger, version {challenger_version}")  


# COMMAND ----------

def evaluate_and_promote(
    model_name: str,
    champion_alias: str = "champion",
    challenger_alias: str = "challenger",
    metric_name: str = "f1_weighted",
    threshold: float = 0.01
):
    """
    Compare champion and challenger using MLflow run metrics.
    Promote challenger only if it beats champion by more than threshold.

    threshold=0.01 means challenger must be 1% better.
    """

    client = MlflowClient()

    champion_version = client.get_model_version_by_alias(model_name, champion_alias)
    challenger_version = client.get_model_version_by_alias(model_name, challenger_alias)

    champion_run = mlflow.get_run(champion_version.run_id)
    challenger_run = mlflow.get_run(challenger_version.run_id)

    champion_metric = champion_run.data.metrics.get(metric_name)
    challenger_metric = challenger_run.data.metrics.get(metric_name)

    if champion_metric is None:
        raise ValueError(f"Champion run does not have metric: {metric_name}")

    if challenger_metric is None:
        raise ValueError(f"Challenger run does not have metric: {metric_name}")

    required_score = champion_metric * (1 + threshold)

    print(f"Champion version: {champion_version.version}, {metric_name}: {champion_metric:.4f}")
    print(f"Challenger version: {challenger_version.version}, {metric_name}: {challenger_metric:.4f}")
    print(f"Promotion threshold: challenger must exceed {required_score:.4f}")

    if challenger_metric > required_score:
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=challenger_version.version
        )

        client.set_model_version_tag(
            name=model_name,
            version=challenger_version.version,
            key="phase",
            value="champion"
        )

        client.set_model_version_tag(
            name=model_name,
            version=champion_version.version,
            key="phase",
            value="previous_champion"
        )

        print(
            f"Challenger promoted to champion: "
            f"{metric_name} improved from {champion_metric:.4f} to {challenger_metric:.4f}"
        )

        return {
            "promoted": True,
            "old_champion_version": champion_version.version,
            "new_champion_version": challenger_version.version,
            "champion_metric": champion_metric,
            "challenger_metric": challenger_metric
        }

    else:
        print(
            f"Champion retained: challenger {metric_name} ({challenger_metric:.4f}) "
            f"did not exceed threshold ({champion_metric:.4f} * 1.01 = {required_score:.4f})"
        )

        return {
            "promoted": False,
            "champion_version": champion_version.version,
            "challenger_version": challenger_version.version,
            "champion_metric": champion_metric,
            "challenger_metric": challenger_metric
        }

# COMMAND ----------

promotion_result = evaluate_and_promote(
    model_name=model_name,
    champion_alias="champion",
    challenger_alias="challenger",
    metric_name="f1_score_wighted",
    threshold=0.01
)

promotion_result

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

model_uri = "models:/bank.models.subscription_classifier@champion"



# COMMAND ----------

key_df=spark.table("bank.silver.data_cleaned").select("customer_id")
scored_df=fe.score_batch(
    df=key_df,
    model_uri=model_uri,
    result_type="double"
)
display(scored_df)

# COMMAND ----------

