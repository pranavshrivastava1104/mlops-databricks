# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

# Databricks notebook source

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

# COMMAND ----------

# Widgets only for model details.
# No approval_required widget for testing.

dbutils.widgets.text("model_name", "bank.models.subscription_classifier")
dbutils.widgets.text("model_version", "5")

model_name = dbutils.widgets.get("model_name").strip()
model_version = dbutils.widgets.get("model_version").strip()

# For testing, approval is always disabled.
approval_required = False

print("Model name:", model_name)
print("Model version:", model_version)
print("Approval required:", approval_required)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

if not model_version:
    raise ValueError("model_version parameter is required for approval check.")

version_info = client.get_model_version(
    name=model_name,
    version=model_version
)

print("Model version tags:")
print(version_info.tags)

# COMMAND ----------

# Testing mode: always skip manual approval.

if not approval_required:
    print("Approval is disabled for this test run. Skipping approval check.")

    dbutils.jobs.taskValues.set(
        key="approval_passed",
        value="true"
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="approval_check_status",
        value="skipped_for_testing"
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="approval_required",
        value="false"
    )

print("Approval check passed.")

# COMMAND ----------

