import argparse
from pathlib import Path
from typing import Any

from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.models.basic_model import BasicModel


def get_dbutils(spark: SparkSession) -> Any | None:
    """
    Creates a dbutils object inside a Python script task.

    In Databricks notebooks, dbutils is already available.
    But in spark_python_task files, it is safer to create it using DBUtils.

    We return None when running locally, because local Python does not have
    Databricks job task values.
    """

    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)

    except Exception as e:
        logger.warning(
            "dbutils is not available in this runtime. "
            "Task values will not be set. "
            f"Reason: {e}"
        )
        return None


def set_task_value(
    dbutils_obj: Any | None,
    key: str,
    value: str,
) -> None:
    """
    Sets a Databricks task value safely.

    Task values are how one task passes small runtime values to another task.

    In our Phase 4 pipeline:
        train_model task sets model_updated and model_version
        condition_task reads model_updated
        deploy_model task reads model_version
    """

    if dbutils_obj is None:
        logger.warning(
            f"Skipping task value because dbutils is unavailable. "
            f"{key}={value}"
        )
        return

    dbutils_obj.jobs.taskValues.set(
        key=key,
        value=value,
    )

    logger.info(f"Set Databricks task value: {key}={value}")


def parse_args() -> argparse.Namespace:
    """
    Reads parameters passed by Databricks Asset Bundles.

    These arguments will later come from resources/bank-pipeline.yml.

    Example:
        --root-path ${workspace.root_path}
        --env ${bundle.target}
        --git-sha ${var.git_sha}
        --branch ${var.branch}
        --job-run-id {{job.run_id}}
    """

    parser = argparse.ArgumentParser(
        description="Train, evaluate, and conditionally register Bank Marketing model."
    )

    parser.add_argument(
        "--root-path",
        "--root_path",
        dest="root_path",
        required=True,
        help="Bundle root path in Databricks workspace.",
    )

    parser.add_argument(
        "--env",
        required=True,
        choices=["dev", "acc", "prd"],
        help="Target environment. Example: dev, acc, prd.",
    )

    parser.add_argument(
        "--git-sha",
        "--git_sha",
        dest="git_sha",
        default="none",
        help="Git commit SHA passed by DABs or CI/CD.",
    )

    parser.add_argument(
        "--branch",
        default="local",
        help="Git branch name passed by DABs or CI/CD.",
    )

    parser.add_argument(
        "--job-run-id",
        "--job_run_id",
        dest="job_run_id",
        default="manual",
        help="Databricks job run id.",
    )

    return parser.parse_args()


def resolve_config_path(root_path: str) -> Path:
    candidates = [
        Path(root_path) / "files" / "files" / "project_config_bank.yml",
        Path(root_path) / "files" / "project_config_bank.yml",
        Path(root_path) / "project_config_bank.yml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find project_config_bank.yml. "
        f"Searched: {searched}"
    )


def main() -> None:
    """
    Main entry point for Phase 4 Step 2.

    This function intentionally does not contain model internals.
    The model internals live inside BasicModel.

    This script is responsible for orchestration:
        config loading
        Spark creation
        MLflow tags
        model lifecycle method calls
        Databricks task values
    """

    args = parse_args()

    logger.info("Starting train_register_model.py")
    logger.info(
        f"Received args: env={args.env}, "
        f"root_path={args.root_path}, "
        f"git_sha={args.git_sha}, "
        f"branch={args.branch}, "
        f"job_run_id={args.job_run_id}"
    )

    config_path = resolve_config_path(args.root_path)

    logger.info(f"Loading config from: {config_path}")

    config = ProjectConfig.from_yaml(
        config_path=str(config_path),
        env=args.env,
    )

    logger.info(
        f"Config loaded successfully. "
        f"catalog={config.catalog_name}, "
        f"schema={config.schema_name}, "
        f"experiment={config.experiment_name}"
    )

    spark = SparkSession.builder.getOrCreate()

    dbutils_obj = get_dbutils(spark)

    tags = {
        "git_sha": args.git_sha,
        "branch": args.branch,
        "job_run_id": args.job_run_id,
        "bundle_target": args.env,
        "pipeline_step": "train_register_model",
        "execution_style": "spark_python_task",
    }

    logger.info(f"MLflow tags prepared: {tags}")

    model = BasicModel(
        config=config,
        spark=spark,
        tags=tags,
    )

    logger.info("Loading training and test data.")
    model.load_data()

    logger.info("Training model.")
    model.train()

    logger.info("Checking whether candidate model improved.")

    # model_improved compares self.metrics against the current latest-model alias.
    # We pass 0 here because the candidate is not registered yet.
    improved = model.model_improved(new_version=0)

    candidate_f1 = model.metrics.get("f1_weighted", 0.0)

    set_task_value(
        dbutils_obj=dbutils_obj,
        key="candidate_f1_weighted",
        value=str(candidate_f1),
    )

    set_task_value(
        dbutils_obj=dbutils_obj,
        key="candidate_run_id",
        value=str(model.run_id),
    )

    if improved:
        logger.info(
            "Candidate model improved. Registering model to Unity Catalog."
        )

        run_id, version = model.register_model()

        set_task_value(
            dbutils_obj=dbutils_obj,
            key="model_updated",
            value="1",
        )

        set_task_value(
            dbutils_obj=dbutils_obj,
            key="model_version",
            value=str(version),
        )

        set_task_value(
            dbutils_obj=dbutils_obj,
            key="registered_run_id",
            value=str(run_id),
        )

        logger.info(
            f"Model registered successfully. "
            f"run_id={run_id}, version={version}, "
            f"f1_weighted={candidate_f1}"
        )

    else:
        logger.info(
            "Candidate model did not improve enough. "
            "Skipping model registration."
        )

        set_task_value(
            dbutils_obj=dbutils_obj,
            key="model_updated",
            value="0",
        )

        set_task_value(
            dbutils_obj=dbutils_obj,
            key="model_version",
            value="",
        )

        logger.info(
            f"No new model registered. "
            f"candidate_run_id={model.run_id}, "
            f"candidate_f1_weighted={candidate_f1}"
        )

    logger.info("train_register_model.py completed successfully.")


if __name__ == "__main__":
    main()
