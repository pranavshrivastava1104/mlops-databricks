import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from sklearn.model_selection import train_test_split
from loguru import logger

from bank_marketing.config import ProjectConfig

class DataProcessor:
    def __init__(self,config:ProjectConfig,spark:SparkSession):
        self.config=config
        self.spark=spark
        self.df=None 
    def load_data(self)->None:
        """
        Reads raw data from the bronze Delta table.
        This is the production pattern — always read from Delta,
        never from a raw CSV file in a pipeline script.
        """
        self.df=(
            self.spark.table(f"{self.config.catalog_name}.bronze.raw_data").toPandas()
        )
        logger.info(
            f"Loaded bronze table from "
            f"{self.config.catalog_name}.bronze.raw_data. "
            f"Shape: {self.df.shape}"
        )
    def preprocess(self) -> None:
        """
        Applies the same transformations as your 01_bank_bronze_setup notebook.
        Converts the raw DataFrame into a clean, ML-ready DataFrame.
        Filters to adults only, creates y_flag, creates all 7 engineered features.
        """
        if self.df is None:
            raise ValueError("Call load_data() before preprocess().")

        # keep only adults — same filter from your notebook
        self.df = self.df[self.df["age"] >= 18].copy()

        # binary label — 1 if subscribed, 0 if not
        self.df["y_flag"] = (self.df["y"] == "yes").astype(int)
        self.df = self.df.drop(columns=["y"])

        # age_group — same buckets as your notebook
        self.df["age_group"] = pd.cut(
            self.df["age"],
            bins=[0, 30, 45, 60, 120],
            labels=["young", "adult", "middle_age", "senior"]
        ).astype(str)

        # balance_tier — same buckets as your notebook
        self.df["balance_tier"] = pd.cut(
            self.df["balance"],
            bins=[-999999, 0, 300, 1000, 999999],
            labels=["negative", "low", "medium", "high"]
        ).astype(str)

        # contact_intensity — same logic from your notebook
        self.df["contact_intensity"] = self.df["campaign"].apply(
            lambda x: "low" if x <= 1 else "high"
        )

        # was_previously_contacted — pdays == -1 means never contacted
        self.df["was_previously_contacted"] = (
            self.df["pdays"] != -1
        ).astype(int)

        # recency_score — same tier buckets from your notebook
        def recency(pdays):
            if pdays == -1:
                return 0
            elif pdays <= 30:
                return 3
            elif pdays <= 180:
                return 2
            else:
                return 1

        self.df["recency_score"] = self.df["pdays"].apply(recency)

        # duration_bucket — same buckets from your notebook
        self.df["duration_bucket"] = pd.cut(
            self.df["duration"],
            bins=[0, 120, 300, 999999],
            labels=["short", "medium", "long"]
        ).astype(str)

        # financial_stress — same composite logic from your notebook
        def fin_stress(row):
            if row["balance"] < 0 or (
                row["housing"] == "yes" and row["loan"] == "yes"
            ):
                return "high"
            elif row["housing"] == "yes" or row["loan"] == "yes":
                return "medium"
            else:
                return "low"

        self.df["financial_stress"] = self.df.apply(fin_stress, axis=1)

        logger.info(
            f"Preprocessing done. Shape: {self.df.shape}. "
            f"Subscription rate: {self.df['y_flag'].mean():.4f}"
        )

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        80/20 stratified split — same pattern as your mlflow_model_training notebook.
        Puts y_flag back into both sets so the table is self-contained.
        """
        feature_cols = self.config.num_features + self.config.cat_features
        X = self.df[feature_cols]
        y = self.df[self.config.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        train_set = X_train.copy()
        train_set[self.config.target] = y_train.values

        test_set = X_test.copy()
        test_set[self.config.target] = y_test.values

        logger.info(
            f"Split done. Train: {train_set.shape}, Test: {test_set.shape}"
        )
        return train_set, test_set

    def save_silver_table(self) -> None:
        """
        Saves the fully preprocessed DataFrame to bank.silver.data_cleaned.
        This is what your notebook saves after all transformations.
        """
        silver_spark = self.spark.createDataFrame(self.df)
        (
            silver_spark.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(
                f"{self.config.catalog_name}.silver.data_cleaned"
            )
        )
        logger.info(
            f"Saved to "
            f"{self.config.catalog_name}.silver.data_cleaned"
        )

    def save_train_test_tables(
        self,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
    ) -> None:
        """
        Saves train and test splits as separate Delta tables.
        Used by the training script to load data without re-splitting every time.
        """
        catalog = self.config.catalog_name

        self.spark.createDataFrame(train_set).write.format("delta") \
            .mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(f"{catalog}.silver.train_set")

        self.spark.createDataFrame(test_set).write.format("delta") \
            .mode("overwrite").option("overwriteSchema", "true") \
            .saveAsTable(f"{catalog}.silver.test_set")

        logger.info(
            f"Saved {catalog}.silver.train_set "
            f"and {catalog}.silver.test_set"
        )


