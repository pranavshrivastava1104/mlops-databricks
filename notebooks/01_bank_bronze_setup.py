# Databricks notebook source
display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS bank")
spark.sql("CREATE SCHEMA IF NOT EXISTS bank.bronze")
spark.sql("CREATE VOLUME IF NOT EXISTS bank.bronze.raw_files")

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/bank/bronze/raw_files/"))

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/bank/bronze/raw_files/"))

# COMMAND ----------

raw_path="/Volumes/bank/bronze/raw_files/bank-full.csv"

bank_df=(
    spark.read
    .option("header",True)
    .option("inferschema",True)
    .option("sep",";")
    .csv(raw_path)
)

display(bank_df)

# COMMAND ----------

bank_df.write.mode("overwrite").format("delta").saveAsTable("bank.bronze.raw_data")

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from bank.bronze.raw_data

# COMMAND ----------

bronze_df=spark.read.table("bank.bronze.raw_data")
bronze_df.groupBy("y").count().show()
display(bronze_df.limit(10))


# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE DETAIL bank.bronze.raw_data;

# COMMAND ----------

spark.version

# COMMAND ----------

bronze_df = spark.table('bank.bronze.raw_data')
bronze_df.printSchema()


# COMMAND ----------

selected_df = bronze_df.select(
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y"
)

selected_df.show(5,truncate=False)

# COMMAND ----------

adult_df=selected_df.filter(selected_df["age"]>=18)
adult_count=adult_df.count()
print("adults counts are:",adult_count)

# COMMAND ----------

from pyspark.sql import functions as F
# creating ml columns for feature engineering using spark transformations
feature_df=(
    adult_df
    .withColumn(
        "y_flag",
        F.when(F.col("y")=="yes",1).otherwise(0)

    )

    .withColumn(
        "age_group",
        F.when(F.col("age") < 30, F.lit("young"))
         .when((F.col("age") >= 30) & (F.col("age") < 45), F.lit("adult"))
         .when((F.col("age") >= 45) & (F.col("age") < 60), F.lit("middle_age"))
         .otherwise(F.lit("senior"))
    )

    .withColumn(
        "balance_tier",
        F.when(F.col("balance") < 0, F.lit("negative"))
         .when(F.col("balance") < 300, F.lit("low"))
         .when(F.col("balance") <1000, F.lit("medium"))
         .otherwise(F.lit("high"))
    )

    .withColumn(
        "contact_intensity",
        F.when(F.col("campaign")<=1,F.lit("low"))
        .when(F.col("campaign")<=3,F.lit("high"))
        .otherwise("high")
    )
    
    .withColumn(
        "was_previously_contacted",
        F.when(F.col("pdays") == -1, F.lit(0)).otherwise(F.lit(1))
    )

    .withColumn(
        "recency_score",
        F.when(F.col("pdays") == -1, F.lit(0))
         .when(F.col("pdays") <= 30, F.lit(3))
         .when(F.col("pdays") <= 180, F.lit(2))
         .otherwise(F.lit(1))
    )

    .withColumn(
        "duration_bucket",
        F.when(F.col("duration") < 120, F.lit("short"))
         .when(F.col("duration") < 300, F.lit("medium"))
         .otherwise(F.lit("long"))
    )

    .withColumn(
        "financial_stress",
        F.when(
            (F.col("balance") < 0) | ((F.col("housing") == "yes") & (F.col("loan") == "yes")),
            F.lit("high")
        )
        .when(
            (F.col("housing") == "yes") | (F.col("loan") == "yes"),
            F.lit("medium")
        )
        .otherwise(F.lit("low"))
    )
)

display(feature_df.limit(10))

# COMMAND ----------

bronze_df.columns

# COMMAND ----------

print(bronze_df.select("balance").distinct().count())
print(bronze_df.select("balance").distinct().show())

# COMMAND ----------

class_balance=(
    feature_df
    .groupBy("y","y_flag")
    .agg(
        F.count("*").alias("raw_count"),
        F.avg("balance").alias("average_balance"),
        F.avg("duration").alias("average_duration")
    )
    .orderBy ("y_flag")
)
display(class_balance)

# COMMAND ----------

# Analyze how many customers were previously contacted.
# pdays = -1 means never contacted before.
previous_contact_df = (
    feature_df
    .groupBy("was_previously_contacted")
    .agg(
        F.count("*").alias("customer_count"),
        F.avg("y_flag").alias("subscription_rate")
    )
    .orderBy("was_previously_contacted")
)

# This action lets you compare subscription rate for never-contacted vs previously-contacted customers.
display(previous_contact_df)

# COMMAND ----------


spark.sql("CREATE SCHEMA IF NOT EXISTS bank.silver")


# COMMAND ----------

(
feature_df.write
.format("delta")
.mode("overwrite")
.option("overwriteSchema",True)
.saveAsTable("bank.silver.data_cleaned")
)

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bank.bronze.raw_data

# COMMAND ----------

spark.read.option("VersionAsof",0).table("bank.bronze.raw_data")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

silver_df=spark.table("bank.silver.data_cleaned")
print("silver row counts:",silver_df.count())
display(
    silver_df.select(
        "age",
        "job",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y",
        "y_flag",
        "age_group",
        "balance_tier",
        "contact_intensity",
        "was_previously_contacted",
        "recency_score",
        "duration_bucket",
        "financial_stress"
    ).limit(10)
)

# COMMAND ----------

from pyspark.sql import Window
window_spec = Window.orderBy(
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "y"
)

# COMMAND ----------

from pyspark.sql import functions as F
merge_demo=(
    silver_df
    .select(
        "age",
        "job",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y",
        "y_flag",
        "age_group",
        "balance_tier",
        "contact_intensity",
        "was_previously_contacted",
        "recency_score",
        "duration_bucket",
        "financial_stress"
    )
    .withColumn("contact_id",F.row_number().over(window_spec))
    .withColumn("merge_action_note",F.lit("original_record"))
)

# COMMAND ----------

# This makes the table easier to inspect during the merge demo.
merge_demo = merge_demo.select(
    "contact_id",
    "age",
    "job",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "y",
    "y_flag",
    "age_group",
    "balance_tier",
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "duration_bucket",
    "financial_stress",
    "merge_action_note"
)

# COMMAND ----------

(
    merge_demo.write
    .format("delta")
    .mode("overWrite")
    .option("overwriteSchema","true")
    .saveAsTable("bank.silver.cleaned_merge_demo")
)
print("demo new delta table created after merging in silver layer")

# COMMAND ----------

demo_df=spark.table("bank.silver.cleaned_merge_demo")
print("demo df row counts:",demo_df.count())
display(
    demo_df
    .select(
        "contact_id",
        "age",
        "job",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "y",
        "y_flag",
        "merge_action_note"
    )
    .orderBy("contact_id")
    .limit(10)

)

# COMMAND ----------

# creating incoming new batch data 
# contact id =1 , 2 == update this 
# contact _id=999999   == insert new row

exsisting_update_df=(
    demo_df
    .filter(F.col("contact_id").isin(1,2))
    .withColumn("balance",F.col("balance")+F.lit(2000))
    .withColumn("duration",F.col("duration")+F.lit(60))
    .withColumn("campaign",F.col("campaign")+F.lit(1))
    .withColumn("merge_action_note",F.lit("updated exsisting record"))
)


# COMMAND ----------

# Create one brand-new record.
# contact_id = 999999 does not exist in the target table, so MERGE should insert it.
new_record_df = spark.createDataFrame(
    [
        (
            999999,              # contact_id
            35,                  # age
            "technician",        # job
            1200,                # balance
            240,                 # duration
            2,                   # campaign
            -1,                  # pdays
            0,                   # previous
            "no",                # y
            0,                   # y_flag
            "adult",             # age_group
            "medium",            # balance_tier
            "medium",            # contact_intensity
            0,                   # was_previously_contacted
            0,                   # recency_score
            "medium",            # duration_bucket
            "medium",            # financial_stress
            "new_inserted_record" # merge_action_note
        )
    ],
    [
        "contact_id",
        "age",
        "job",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y",
        "y_flag",
        "age_group",
        "balance_tier",
        "contact_intensity",
        "was_previously_contacted",
        "recency_score",
        "duration_bucket",
        "financial_stress",
        "merge_action_note"
    ]
)

# COMMAND ----------

incoming_updated_df=exsisting_update_df.unionByName(new_record_df)
incoming_updated_df.createOrReplaceTempView("incoming_bank_campaign_updates")
display(incoming_updated_df.orderBy("contact_id"))


# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO bank.silver.cleaned_merge_demo AS target
# MAGIC USING incoming_bank_campaign_updates AS source
# MAGIC ON target.contact_id = source.contact_id
# MAGIC
# MAGIC WHEN MATCHED THEN UPDATE SET
# MAGIC   target.age = source.age,
# MAGIC   target.job = source.job,
# MAGIC   target.balance = source.balance,
# MAGIC   target.duration = source.duration,
# MAGIC   target.campaign = source.campaign,
# MAGIC   target.pdays = source.pdays,
# MAGIC   target.previous = source.previous,
# MAGIC   target.y = source.y,
# MAGIC   target.y_flag = source.y_flag,
# MAGIC   target.age_group = source.age_group,
# MAGIC   target.balance_tier = source.balance_tier,
# MAGIC   target.contact_intensity = source.contact_intensity,
# MAGIC   target.was_previously_contacted = source.was_previously_contacted,
# MAGIC   target.recency_score = source.recency_score,
# MAGIC   target.duration_bucket = source.duration_bucket,
# MAGIC   target.financial_stress = source.financial_stress,
# MAGIC   target.merge_action_note = source.merge_action_note
# MAGIC
# MAGIC WHEN NOT MATCHED THEN INSERT (
# MAGIC   contact_id,
# MAGIC   age,
# MAGIC   job,
# MAGIC   balance,
# MAGIC   duration,
# MAGIC   campaign,
# MAGIC   pdays,
# MAGIC   previous,
# MAGIC   y,
# MAGIC   y_flag,
# MAGIC   age_group,
# MAGIC   balance_tier,
# MAGIC   contact_intensity,
# MAGIC   was_previously_contacted,
# MAGIC   recency_score,
# MAGIC   duration_bucket,
# MAGIC   financial_stress,
# MAGIC   merge_action_note
# MAGIC )
# MAGIC VALUES (
# MAGIC   source.contact_id,
# MAGIC   source.age,
# MAGIC   source.job,
# MAGIC   source.balance,
# MAGIC   source.duration,
# MAGIC   source.campaign,
# MAGIC   source.pdays,
# MAGIC   source.previous,
# MAGIC   source.y,
# MAGIC   source.y_flag,
# MAGIC   source.age_group,
# MAGIC   source.balance_tier,
# MAGIC   source.contact_intensity,
# MAGIC   source.was_previously_contacted,
# MAGIC   source.recency_score,
# MAGIC   source.duration_bucket,
# MAGIC   source.financial_stress,
# MAGIC   source.merge_action_note
# MAGIC );

# COMMAND ----------

merged_df=spark.table("bank.silver.cleaned_merge_demo")
print("final row counts for merged demo:",merged_df.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY bank.silver.cleaned_merge_demo
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- this is comparing history using time travel
# MAGIC select 'VERSION_AS_OF 0' as snapshot , count(*) as row_count
# MAGIC from bank.silver.cleaned_merge_demo VERSION AS OF 0
# MAGIC union all 
# MAGIC select  'VERSION_AS_OF 1' as snapshot , count(*) as row_count
# MAGIC from bank.silver.cleaned_merge_demo VERSION AS OF 1
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F 
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup