import argparse
from pyspark.sql import SparkSession
from loguru import logger

from bank_marketing.config import ProjectConfig
from bank_marketing.data_processor import DataProcessor
from bank_marketing.feature_engineering import FeatureProcessor

def main():
    parser=argparse.ArgumentParser(
        description="Bank Marketing — data processing pipeline"
    )
    parser.add_argument(
        "--root-path",
        required=True,
        help="root path for project bundles resolved by (DAB au runtime)"
    )
    parser.add_argument(
        "--env",
        required=True,
        help="envirnoment to run for (dev,prod,acc)"
    )
    args=parser.parse_args()
    config = ProjectConfig.from_yaml(
        config_path=f"{args.root_path}/files/project_config_bank.yml",
        env=args.env,
    )

    logger.info(
        f"Config loaded. env={args.env}, "
        f"catalog={config.catalog_name}, "
        f"schema={config.schema_name}"
    )

    spark = SparkSession.builder.getOrCreate()

     # --- Step 1: preprocess raw bronze data into silver ---
    processor=DataProcessor(config=config,spark=spark)
    processor.load_data()
    processor.preprocess()
    processor.save_to_silver()

    train_set, test_set = processor.split_data()
    processor.save_train_test_tables(train_set=train_set, test_set=test_set)

    # --- Step 2: compute features and write to Feature Store ---

    feature_processor = FeatureProcessor(config=config, spark=spark)
    feature_processor.compute_and_save()

    logger.info("process_data.py completed successfully.")

    if __name__ == "__main__":
        main()


