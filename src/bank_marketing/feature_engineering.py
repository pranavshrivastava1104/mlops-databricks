from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient
from loguru import logger

from bank_marketing.config import ProjectConfig


def compute_customer_features(df):
    """
    Copied directly from your feature engineering notebook.
    Takes the silver DataFrame, creates a synthetic customer_id,
    computes the 7 engineered features, returns features-only DataFrame.
    y_flag is NOT included — labels and features stay separate.
    """

    # synthetic key — same window ordering as your notebook
    key_window = Window.orderBy(
        "age", "job", "balance", "duration",
        "campaign", "pdays", "previous"
    )

    df_with_key = df.withColumn(
        "customer_id",
        F.row_number().over(key_window).cast("long")  # fixed: "long" not LONG
    )

    features_df = (
        df_with_key

        .withColumn(
            "age_group",
            F.when(F.col("age") <= 30, F.lit("young"))
             .when((F.col("age") > 30) & (F.col("age") <= 50), F.lit("middle"))
             .when((F.col("age") > 50) & (F.col("age") <= 65), F.lit("senior"))
             .otherwise(F.lit("elderly"))
        )

        .withColumn(
            "balance_tier",
            F.when(F.col("balance") < 0, F.lit("negative"))
             .when((F.col("balance") >= 0) & (F.col("balance") < 500), F.lit("low"))
             .when((F.col("balance") >= 500) & (F.col("balance") <= 2000), F.lit("medium"))
             .otherwise(F.lit("high"))
        )

        # numeric ratio — same as your feature engineering notebook
        .withColumn(
            "contact_intensity",
            F.col("campaign").cast("double") /
            (F.col("previous").cast("double") + F.lit(1.0))
        )

        .withColumn(
            "was_previously_contacted",
            F.when(F.col("pdays") != -1, F.lit(1)).otherwise(F.lit(0))
        )

        .withColumn(
            "recency_score",
            F.when(F.col("pdays") == -1, F.lit(0.0))
             .otherwise(
                 F.lit(1.0) / (F.col("pdays").cast("double") + F.lit(1.0))
             )
        )

        .withColumn(
            "duration_bucket",
            F.when(F.col("duration") < 60, F.lit("very_short"))
             .when((F.col("duration") >= 60) & (F.col("duration") < 180), F.lit("short"))
             .when((F.col("duration") >= 180) & (F.col("duration") <= 500), F.lit("medium"))
             .otherwise(F.lit("long"))
        )

        # binary flag — same as your feature engineering notebook
        .withColumn(
            "financial_stress",
            F.when(
                (F.col("housing") == "yes") & (F.col("loan") == "yes"),
                F.lit(1)
            ).otherwise(F.lit(0))
        )

        .select(
            "customer_id",
            "age_group",
            "balance_tier",
            "contact_intensity",
            "was_previously_contacted",
            "recency_score",
            "duration_bucket",
            "financial_stress"
        )
    )

    return features_df


class FeatureProcessor:
    def __init__(
        self,
        config: ProjectConfig,
        spark: SparkSession,
    ):
        self.config = config
        self.spark = spark
        self.fe = FeatureEngineeringClient()
        self.feature_table = (
            f"{config.catalog_name}.feature_store.customer_features"
        )

    def compute_and_save(self) -> None:
        """
        Reads from silver table, computes features, writes to Feature Store.
        Wraps the exact same flow from your feature engineering notebook.
        """
        silver_df = self.spark.table(
            f"{self.config.catalog_name}.silver.data_cleaned"
        )
        logger.info(
            f"Read silver table. Row count: {silver_df.count()}"
        )

        features_df = compute_customer_features(silver_df)
        logger.info(
            f"Features computed. Row count: {features_df.count()}"
        )

        # create table only if it does not exist yet
        try:
            self.fe.get_table(self.feature_table)
            logger.info(
                f"Feature table already exists: {self.feature_table}"
            )
        except Exception:
            self.fe.create_table(
                name=self.feature_table,
                primary_keys=["customer_id"],
                schema=features_df.schema,
                description=(
                    "Engineered features for bank term deposit subscription "
                    "prediction. Source: bank.silver.data_cleaned."
                )
            )
            logger.info(
                f"Feature table created: {self.feature_table}"
            )

        self.fe.write_table(
            name=self.feature_table,
            df=features_df,
            mode="merge"
        )
        logger.info(
            f"Features written to {self.feature_table}"
        )