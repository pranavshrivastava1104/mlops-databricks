# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# FeatureEngineeringClient is the Unity Catalog-aware client for creating and using feature tables.
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS bank.feature_store")

# COMMAND ----------

# READING DATA FROM THE SILVER TABLE
silver_df=spark.table("bank.silver.data_cleaned")
print("row counts in silver table:",silver_df.count())
print(silver_df.printSchema())

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN bank.silver;

# COMMAND ----------

def compute_customer_features(df):
    """
    Takes the Bank Marketing Silver DataFrame and returns only:
    - customer_id
    - 7 engineered feature columns

    Important:
    This function does NOT return y_flag.
    Labels and features should be kept separate in the Feature Store pattern.
    """

    # Create a deterministic ordering window.
    # The UCI Bank Marketing dataset has no real customer_id.
    # For practice, we create a synthetic customer_id using row_number().
    # In production, always use a real source-system key like customer_id or lead_id.
    key_window = Window.orderBy(
        "age",
        "job",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous"
    )

    # Add synthetic customer_id.
    # This gives every row a unique key so the feature table can perform merge/upsert.
    df_with_key = df.withColumn(
        "customer_id",
        F.row_number().over(key_window).cast(LONG)
    )

    # Compute the 7 engineered features.
    features_df = (
        df_with_key

        # age_group buckets raw age into business-friendly segments.
        # This is useful because subscription behavior may differ by life stage.
        .withColumn(
            "age_group",
            F.when(F.col("age") <= 30, F.lit("young"))
             .when((F.col("age") > 30) & (F.col("age") <= 50), F.lit("middle"))
             .when((F.col("age") > 50) & (F.col("age") <= 65), F.lit("senior"))
             .otherwise(F.lit("elderly"))
        )

        # balance_tier converts skewed numeric balance into interpretable buckets.
        # This helps models and analysts reason about customer financial position.
        .withColumn(
            "balance_tier",
            F.when(F.col("balance") < 0, F.lit("negative"))
             .when((F.col("balance") >= 0) & (F.col("balance") < 500), F.lit("low"))
             .when((F.col("balance") >= 500) & (F.col("balance") <= 2000), F.lit("medium"))
             .otherwise(F.lit("high"))
        )

        # contact_intensity measures how aggressively the customer is contacted now
        # compared to previous contact history.
        # previous + 1 avoids division by zero when previous = 0.
        .withColumn(
            "contact_intensity",
            F.col("campaign").cast("double") / (F.col("previous").cast("double") + F.lit(1.0))
        )

        # was_previously_contacted converts pdays into a clean binary signal.
        # In this dataset, pdays = -1 means the customer was never contacted before.
        .withColumn(
            "was_previously_contacted",
            F.when(F.col("pdays") != -1, F.lit(1)).otherwise(F.lit(0))
        )

        # recency_score is higher when previous contact was more recent.
        # For never-contacted customers, recency_score is 0.
        .withColumn(
            "recency_score",
            F.when(F.col("pdays") == -1, F.lit(0.0))
             .otherwise(1.0 / (F.col("pdays").cast("double") + F.lit(1.0)))
        )

        # duration_bucket groups call duration into meaningful categories.
        # Long calls are usually strongly associated with subscription likelihood.
        .withColumn(
            "duration_bucket",
            F.when(F.col("duration") < 60, F.lit("very_short"))
             .when((F.col("duration") >= 60) & (F.col("duration") < 180), F.lit("short"))
             .when((F.col("duration") >= 180) & (F.col("duration") <= 500), F.lit("medium"))
             .otherwise(F.lit("long"))
        )

        # financial_stress is a binary composite feature.
        # It becomes 1 when the customer has both housing loan and personal loan.
        .withColumn(
            "financial_stress",
            F.when((F.col("housing") == "yes") & (F.col("loan") == "yes"), F.lit(1))
             .otherwise(F.lit(0))
        )

        # Select only the primary key and feature columns.
        # Do not include y_flag because labels should not be stored as features.
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

# COMMAND ----------

customer_feature_df=compute_customer_features(silver_df)
print("customer_feature_row_count:",customer_feature_df.count())
display(customer_feature_df.limit(10))

# COMMAND ----------

# creating feature store table
feature_store_table="bank.feature_store.customer_features"

# COMMAND ----------

# Create the feature table metadata in Unity Catalog.
# primary_keys tells Databricks which column uniquely identifies a feature row.
# schema tells Databricks the structure of the feature table.
# description helps teammates understand what the table is for.

fe = FeatureEngineeringClient()
fe.create_table(
    name=feature_store_table,
    primary_keys=["customer_id"],
    schema=customer_feature_df.schema,
    description=(
        "created the feature store with name customer_features all the engineered ml features are stored there."
    )
)

# COMMAND ----------

fe.write_table(
    name=feature_store_table,
    df=customer_feature_df,
    mode="merge"
)

# COMMAND ----------

# creating label dataframe 
key_window = Window.orderBy(
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous"
)
label_df=(
    silver_df
    .withColumn("customer_id",F.row_number().over(key_window))
    .select("customer_id","y_flag")
)
display(label_df.limit(10))

# COMMAND ----------

# feature lookups
from databricks.feature_engineering import FeatureLookup
customer_feature_lookup=FeatureLookup(
    table_name=feature_store_table,
    lookup_key="customer_id",
    feature_names=[
        "age_group",
        "balance_tier",
        "contact_intensity",
        "was_previously_contacted",
        "recency_score",
        "duration_bucket",
        "financial_stress"
    ]

)


# COMMAND ----------

# create training setfrom databricks 
from databricks.feature_engineering import FeatureEngineeringClient
fe=FeatureEngineeringClient()
label_df=label_df.withColumn(
    "customer_id",
    F.col("customer_id").cast("LONG")
)
training_set=fe.create_training_set(
    df=label_df,
    feature_lookups=[customer_feature_lookup],
    label="y_flag",
    exclude_columns=["customer_id"]
)
training_df=training_set.load_df()
print("training_set rows:", training_df.count())
print("training  set columns:",training_df.columns)
display(training_df.limit(10))



# COMMAND ----------

