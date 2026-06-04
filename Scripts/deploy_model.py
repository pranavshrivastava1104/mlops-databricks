import argparse
from pathlib import Path
from typing import Any

from loguru import logger
from pyspark.sql import SparkSession

from bank_marketing.config import ProjectConfig
from bank_marketing.serving.model_serving import ModelServing


def get_dbutils(spark: SparkSession) -> Any | None:
    """
    Creates dbutils inside a spark_python_task.

    In Databricks notebooks, dbutils is globally available.
    In Python script tasks, we create it using DBUtils.

    We return None when running locally.
    """

    try:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)

    except Exception as e:
        logger.warning(
            "dbutils is not available. "
            "This usually happens during local testing. "
            f"Reason: {e}"
        )
        return None


def get_task_value(
    dbutils_obj: Any | None,
    task_key: str,
    key: str,
    default: str = "",
) -> str:
    """
    Reads a value from a previous Databricks task.

    In this step, we read:
        task_key = train_model
        key = model_version

    That value was set by Scripts/train_register_model.py.
    """

    if dbutils_obj is None:
        logger.warning(
            f"Cannot read task value because dbutils is unavailable. "
            f"Returning default value for {task_key}.{key}: {default}"
        )
        return default

    value = dbutils_obj.jobs.taskValues.get(
        taskKey=task_key,
        key=key,
        default=default,
    )

    logger.info(
        f"Read Databricks task value: "
        f"task_key={task_key}, key={key}, value={value}"
    )

    return str(value)


def parse_args() -> argparse.Namespace:
    """
    Reads parameters passed by DABs.

    Required:
        --root-path
        --env

    Optional:
        --model-version

    model-version is optional because in real job execution we read it from
    Databricks task values. But for manual testing, passing --model-version
    is convenient.
    """

    parser = argparse.ArgumentParser(
        description="Deploy registered Bank Marketing model to Databricks Model Serving."
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
        "--model-version",
        "--model_version",
        dest="model_version",
        default="",
        help=(
            "Optional model version for manual testing. "
            "In the DABs workflow, this is normally read from task values."
        ),
    )

    parser.add_argument(
        "--workload-size",
        "--workload_size",
        dest="workload_size",
        default="Small",
        help="Databricks serving workload size. Default: Small.",
    )

    parser.add_argument(
        "--disable-scale-to-zero",
        action="store_true",
        help="Disable scale-to-zero for the serving endpoint.",
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
    Main entry point for Phase 4 Step 3.

    Responsibility:
        1. Load config
        2. Read model_version from train_model task values
        3. Create ModelServing
        4. Deploy or update endpoint
    """

    args = parse_args()

    logger.info("Starting deploy_model.py")
    logger.info(
        f"Received args: env={args.env}, "
        f"root_path={args.root_path}, "
        f"manual_model_version={args.model_version}, "
        f"workload_size={args.workload_size}"
    )

    spark = SparkSession.builder.getOrCreate()
    dbutils_obj = get_dbutils(spark)

    config_path = resolve_config_path(args.root_path)

    logger.info(f"Loading config from: {config_path}")

    config = ProjectConfig.from_yaml(
        config_path=str(config_path),
        env=args.env,
    )

    logger.info(
        f"Config loaded successfully. "
        f"catalog={config.catalog_name}, "
        f"schema={config.schema_name}"
    )

    # In real workflow execution, model_version comes from train_model task.
    model_version = args.model_version.strip()

    if not model_version:
        model_version = get_task_value(
            dbutils_obj=dbutils_obj,
            task_key="train_model",
            key="model_version",
            default="",
        )

    if not model_version:
        raise ValueError(
            "No model_version found. "
            "Either pass --model-version manually or run this task after train_model."
        )

    logger.info(f"Deploying model version: {model_version}")

    serving = ModelServing(
        config=config,
        env=args.env,
        workload_size=args.workload_size,
        scale_to_zero_enabled=not args.disable_scale_to_zero,
    )

    serving.deploy_or_update_serving_endpoint(
        version=model_version,
    )

    logger.info("deploy_model.py completed successfully.")


if __name__ == "__main__":
    main()
