# Databricks notebook source
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

model_name = "bank.models.subscription_classifier"

versions = client.search_model_versions(f"name = '{model_name}'")

for v in versions:
    print(
        "Version:", v.version,
        "| Run ID:", v.run_id,
        "| Status:", v.status,
        "| Aliases:", v.aliases
    )

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("model_name", "bank.models.subscription_classifier")
dbutils.widgets.text("model_version", "5")
dbutils.widgets.text("min_f1", "0.72")
dbutils.widgets.text("promotion_threshold", "0.01")
dbutils.widgets.text("metric_name", "f1_weighted")

# COMMAND ----------

# Databricks notebook source

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

dbutils.widgets.text("model_name", "bank.models.subscription_classifier")
dbutils.widgets.text("model_version", "5")
dbutils.widgets.text("min_f1", "0.72")
dbutils.widgets.text("promotion_threshold", "0.01")
dbutils.widgets.text("metric_name", "f1_weighted")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version").strip()
min_f1 = float(dbutils.widgets.get("min_f1"))
promotion_threshold = float(dbutils.widgets.get("promotion_threshold"))
metric_name = dbutils.widgets.get("metric_name")

print("Model name:", model_name)
print("Candidate model version:", model_version)
print("Minimum F1:", min_f1)
print("Promotion threshold:", promotion_threshold)
print("Metric name:", metric_name)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

if not model_version:
    raise ValueError("model_version parameter is required for deployment evaluation.")

candidate_version_info = client.get_model_version(
    name=model_name,
    version=model_version
)

candidate_run_id = candidate_version_info.run_id

if not candidate_run_id:
    raise ValueError(
        f"Model version {model_version} does not have a source MLflow run_id."
    )

candidate_run = mlflow.get_run(candidate_run_id)

print("Candidate run ID:", candidate_run_id)
print("Candidate tags:", candidate_run.data.tags)
print("Candidate metrics:", candidate_run.data.metrics)

# COMMAND ----------

def get_metric_safely(run, preferred_metric: str) -> float:
    """
    Read metric from MLflow run.

    Your older notebooks have used slightly different names such as:
    f1_weighted, f1_score_weighted, f1_score_wighted.

    This helper avoids deployment failure due to metric-name mismatch.
    """

    fallback_metrics = [
        preferred_metric,
        "f1_weighted",
        "f1_score_weighted",
        "f1_score_wighted",
        "test_f1_score",
        "val_f1",
        "f1_binary",
    ]

    for metric in fallback_metrics:
        if metric in run.data.metrics:
            return float(run.data.metrics[metric])

    raise ValueError(
        f"No valid F1 metric found in run {run.info.run_id}. "
        f"Available metrics: {list(run.data.metrics.keys())}"
    )

candidate_f1 = get_metric_safely(candidate_run, metric_name)

print("Candidate F1:", candidate_f1)

# COMMAND ----------

# Check minimum quality gate.
minimum_quality_passed = candidate_f1 >= min_f1

print("Minimum quality passed:", minimum_quality_passed)

# COMMAND ----------

# Compare candidate with current champion if champion exists.
try:
    champion_info = client.get_model_version_by_alias(
        name=model_name,
        alias="champion"
    )

    champion_run = mlflow.get_run(champion_info.run_id)
    champion_f1 = get_metric_safely(champion_run, metric_name)

    required_f1 = champion_f1 * (promotion_threshold)

    beats_champion = candidate_f1 > required_f1

    print("Current champion version:", champion_info.version)
    print("Champion F1:", champion_f1)
    print("Required F1 for promotion:", required_f1)
    print("Candidate beats champion:", beats_champion)

except Exception as e:
    print("No current champion found or champion metric unavailable.")
    print("Candidate can become first champion if it passes min_f1.")
    champion_info = None
    champion_f1 = None
    required_f1 = None
    beats_champion = True

# COMMAND ----------

evaluation_passed = minimum_quality_passed and beats_champion

client.set_model_version_tag(
    name=model_name,
    version=model_version,
    key="deployment_evaluation_status",
    value="passed" if evaluation_passed else "failed"
)

client.set_model_version_tag(
    name=model_name,
    version=model_version,
    key="candidate_f1",
    value=str(candidate_f1)
)

if champion_f1 is not None:
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="champion_f1_at_evaluation",
        value=str(champion_f1)
    )

# COMMAND ----------

dbutils.jobs.taskValues.set(
    key="evaluation_passed",
    value=str(evaluation_passed).lower()
)

dbutils.jobs.taskValues.set(
    key="candidate_f1",
    value=str(candidate_f1)
)

dbutils.jobs.taskValues.set(
    key="candidate_run_id",
    value=candidate_run_id
)

print("Evaluation passed:", evaluation_passed)

if not evaluation_passed:
    raise ValueError(
        f"Candidate model version {model_version} failed deployment evaluation. "
        f"candidate_f1={candidate_f1}, min_f1={min_f1}, "
        f"champion_f1={champion_f1}, required_f1={required_f1}"
    )

print("Candidate model passed deployment evaluation.")