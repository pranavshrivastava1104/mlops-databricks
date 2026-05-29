# Databricks notebook source
# Databricks notebook source

# COMMAND ----------

# Optional. Use only if your cluster/runtime does not already have these.
# On Databricks Runtime ML, you may skip this to avoid dependency conflicts.
# MAGIC %pip install --quiet mlflow databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

import os
import mlflow

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from mlflow.tracking.client import MlflowClient

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

current_user = spark.sql("SELECT current_user()").collect()[0][0]

print("Current user:", current_user)

# COMMAND ----------

current_directory = os.getcwd()

print("Current notebook directory:", current_directory)

# COMMAND ----------

# REQUIRED: Unity Catalog model name.
model_name = "bank.models.subscription_classifier"

# REQUIRED: Resolve candidate model version.
# Prefer challenger if available. Otherwise use latest-model. Otherwise fail.
client = MlflowClient(registry_uri="databricks-uc")

try:
    candidate_version_info = client.get_model_version_by_alias(
        name=model_name,
        alias="challenger"
    )
    model_version = str(candidate_version_info.version)
    print("Using challenger version:", model_version)

except Exception:
    try:
        candidate_version_info = client.get_model_version_by_alias(
            name=model_name,
            alias="latest-model"
        )
        model_version = str(candidate_version_info.version)
        print("Using latest-model version:", model_version)

    except Exception:
        raise ValueError(
            f"No challenger or latest-model alias found for {model_name}. "
            f"Set an alias before creating the deployment job."
        )

# REQUIRED: Job name.
job_name = "Bank-Subscription-Model-Deployment-Job"

# REQUIRED: Notebook task paths.
evaluation_notebook_path = f"{current_directory}/10a_deployment_evaluation"
approval_notebook_path = f"{current_directory}/10b_deployment_approval_check"
deployment_notebook_path = f"{current_directory}/10c_deploy_champion_and_score"

print("Model name:", model_name)
print("Model version:", model_version)
print("Job name:", job_name)
print("Evaluation notebook:", evaluation_notebook_path)
print("Approval notebook:", approval_notebook_path)
print("Deployment notebook:", deployment_notebook_path)

# COMMAND ----------

# This follows the exact reference pattern:
# Evaluation -> Approval_Check -> Deployment
job_settings = jobs.JobSettings(
    name=job_name,

    tasks=[
        jobs.Task(
            task_key="Evaluation",
            notebook_task=jobs.NotebookTask(
                notebook_path=evaluation_notebook_path,
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                    "min_f1": "{{job.parameters.min_f1}}",
                    "promotion_threshold": "{{job.parameters.promotion_threshold}}",
                    "metric_name": "{{job.parameters.metric_name}}",
                }
            ),
            max_retries=0,
        ),

        jobs.Task(
            task_key="Approval_Check",
            notebook_task=jobs.NotebookTask(
                notebook_path=approval_notebook_path,
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                    # Same concept as reference notebook:
                    # approval_tag_name = "{{task.name}}"
                    # Here task name/key is Approval_Check.
                    "approval_tag_name": "{{task.name}}",
                    "approval_required": "{{job.parameters.approval_required}}",
                }
            ),
            depends_on=[
                jobs.TaskDependency(task_key="Evaluation")
            ],
            max_retries=0,
        ),

        jobs.Task(
            task_key="Deployment",
            notebook_task=jobs.NotebookTask(
                notebook_path=deployment_notebook_path,
                base_parameters={
                    "model_name": "{{job.parameters.model_name}}",
                    "model_version": "{{job.parameters.model_version}}",
                    "source_table": "{{job.parameters.source_table}}",
                    "output_table": "{{job.parameters.output_table}}",
                    "smoke_test": "{{job.parameters.smoke_test}}",
                }
            ),
            depends_on=[
                jobs.TaskDependency(task_key="Approval_Check")
            ],
            max_retries=0,
        ),
    ],

    parameters=[
        jobs.JobParameter(
            name="model_name",
            default=model_name
        ),
        jobs.JobParameter(
            name="model_version",
            default=model_version
        ),
        jobs.JobParameter(
            name="min_f1",
            default="0.72"
        ),
        jobs.JobParameter(
            name="promotion_threshold",
            default="0.01"
        ),
        jobs.JobParameter(
            name="metric_name",
            default="f1_weighted"
        ),
        jobs.JobParameter(
            name="approval_required",
            default="true"
        ),
        jobs.JobParameter(
            name="source_table",
            default="bank.silver.data_cleaned"
        ),
        jobs.JobParameter(
            name="output_table",
            default="bank.gold.subscription_predictions"
        ),
        jobs.JobParameter(
            name="smoke_test",
            default="false"
        ),
    ],

    # Queue prevents overlapping deployment runs.
    queue=jobs.QueueSettings(enabled=True),

    # Only one deployment run should execute at a time.
    # This avoids two runs changing the champion alias simultaneously.
    max_concurrent_runs=1,
)

# COMMAND ----------

w = WorkspaceClient()

existing_jobs = w.jobs.list(name=job_name)

job_id = None

for created_job in existing_jobs:
    if created_job.settings.name == job_name and created_job.creator_user_name == current_user:
        job_id = created_job.job_id
        break

if job_id:
    print("Updating existing deployment job...")

    w.jobs.update(
        job_id=job_id,
        new_settings=job_settings
    )

else:
    print("Creating new deployment job...")

    try:
        created_job = w.jobs.create(**job_settings.as_dict())
    except Exception:
        # This fallback matches the style used in the reference notebook.
        created_job = w.jobs.create(**job_settings.__dict__)

    job_id = created_job.job_id

print("Job ID:", job_id)

# COMMAND ----------

print(
    "Use the job name "
    + job_name
    + " to connect the deployment job to the UC model "
    + model_name
    + " in the Unity Catalog Model UI."
)

print("\nFor your reference, the job ID is:", job_id)

# Programmatically link deployment job to Unity Catalog model.

try:
    model_info = client.get_registered_model(model_name)
    print(f"Model exists: {model_name}")

except mlflow.exceptions.RestException as e:
    error_message = str(e)

    if (
        "RESOURCE_DOES_NOT_EXIST" in error_message
        or "NOT_FOUND" in error_message
        or "does not exist" in error_message.lower()
    ):
        print(f"Model does not exist. Creating registered model: {model_name}")

        client.create_registered_model(model_name)

        model_info = client.get_registered_model(model_name)

    elif "PERMISSION_DENIED" in error_message:
        raise PermissionError(
            f"Permission denied while reading model `{model_name}`. "
            "Check UC model permissions."
        )

    else:
        raise


# Now link the deployment job only after model existence is confirmed.
try:
    current_deployment_job_id = getattr(model_info, "deployment_job_id", None)

    print("Current deployment job ID on model:", current_deployment_job_id)
    print("Target deployment job ID:", job_id)

    if str(current_deployment_job_id) == str(job_id):
        print("Model already linked to this deployment job. No action needed.")

    else:
        print("Updating deployment job link on model...")

        client.update_registered_model(
            name=model_name,
            deployment_job_id=str(job_id)
        )

        print("Deployment job linked to model successfully.")

except mlflow.exceptions.RestException as e:
    error_message = str(e)

    if "PERMISSION_DENIED" in error_message:
        print(f"Permission denied while linking job to model `{model_name}`.")
        print("Deployment job was created, but model link was not updated.")
        print("You can manually link the job from the Unity Catalog Model UI.")

    else:
        print("Deployment job was created, but automatic model-job linking failed.")
        print("Error:", error_message)
        print("You can manually link the job from the Unity Catalog Model UI.")
# COMMAND ----------

# Optional: trigger a test run from the notebook.
# Keep this commented until you approve the model version tag.
# run_response = w.jobs.run_now(
#     job_id=job_id,
#     job_parameters={
#         "model_name": model_name,
#         "model_version": model_version,
#         "min_f1": "0.72",
#         "promotion_threshold": "0.01",
#         "metric_name": "f1_weighted",
#         "approval_required": "true",
#         "source_table": "bank.silver.data_cleaned",
#         "output_table": "bank.gold.subscription_predictions",
#         "smoke_test": "false",
#     }
# )
# print("Triggered run:", run_response)