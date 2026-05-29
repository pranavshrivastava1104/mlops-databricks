# Databricks notebook source
# MAGIC %pip install --quiet databricks-feature-engineering>=0.13.0a8 mlflow --upgrade lightgbm optuna
# MAGIC
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.sklearn
import optuna

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

catalog = "bank"
silver_table = "bank.silver.data_cleaned"
feature_table_name = "bank.feature_store.customer_features"
model_name = "bank.models.subscription_classifier"

# COMMAND ----------

silver_df = spark.table(silver_table)

print("Silver row count:", silver_df.count())

display(silver_df.limit(5))

# COMMAND ----------


key_window = Window.orderBy(
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

silver_keyed_df = silver_df.withColumn(
    "customer_id",
    F.row_number().over(key_window).cast("long")
)

# COMMAND ----------

silver_keyed_df.write.mode("overwrite").format("delta").option("overwriteSchema", "true").saveAsTable(
    "bank.silver.campaigns_keyed"
)

print("Saved keyed Silver table: bank.silver.campaigns_keyed")

# COMMAND ----------

feature_cols = [
    "age_group",
    "balance_tier",
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "duration_bucket",
    "financial_stress"
]

label_col = "y_flag"

# COMMAND ----------

customer_features_df = silver_keyed_df.select(
    "customer_id",
    *feature_cols
)

print("Feature rows:", customer_features_df.count())

display(customer_features_df.limit(10))

# COMMAND ----------

fe=FeatureEngineeringClient

# COMMAND ----------

labels_df = silver_keyed_df.select(
    "customer_id",
    label_col
)

print("Labels rows:", labels_df.count())

display(labels_df.limit(10))

# COMMAND ----------

customer_feature_lookup = FeatureLookup(
    table_name=feature_table_name,
    lookup_key="customer_id",
    feature_names=feature_cols
)

# COMMAND ----------

# fe should be an instance, not the class itself
fe = FeatureEngineeringClient()

training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=[customer_feature_lookup],
    label=label_col,
    exclude_columns=["customer_id"]
)

# COMMAND ----------

training_spark_df = training_set.load_df()

print("Training rows:", training_spark_df.count())
print("Training columns:", training_spark_df.columns)

display(training_spark_df.limit(10))

# COMMAND ----------

training_pdf = training_spark_df.toPandas()

X = training_pdf[feature_cols]
y = training_pdf[label_col]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:")
print(y.value_counts())

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_test distribution:")
print(y_test.value_counts(normalize=True))

# COMMAND ----------

categorical_cols = [
    "age_group",
    "balance_tier",
    "duration_bucket"
]

numeric_cols = [
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "financial_stress"
]

# Numeric pipeline:
# 1. Convert values to numeric
# 2. Fill missing values with mean
# 3. Standardize numerical columns
numeric_pipeline = Pipeline(
    steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]
)

# Categorical pipeline:
# 1. Fill missing categories
# 2. One-hot encode categories
# handle_unknown='ignore' is important because serving may receive categories not seen during training.
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ColumnTransformer combines categorical and numeric pipelines.
# This same preprocessing will be saved inside the final model pipeline.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_cols),
        ("numeric", numeric_pipeline, numeric_cols)
    ]
)

# COMMAND ----------

# optuna learning 

#Choose model type
#Choose hyperparameters
#Build sklearn Pipeline
#Train model
#Validate model
#Return binary F1 score

# COMMAND ----------

available_classifiers = ["LogisticRegression", "RandomForest"]
print("available_classifiers",available_classifiers)

# COMMAND ----------

# DBTITLE 1,un
class objective_function:
    def __init__(
        self,
        x_train_in: pd.DataFrame,
        y_train_in: pd.Series,
        preprocessor_in: ColumnTransformer,
        rng_seed: int = 42,
    ):
        self.preprocessor = preprocessor_in
        self.rng_seed = rng_seed

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x_train_in,
            y_train_in,
            test_size=0.33,
            random_state=rng_seed,
            stratify=y_train_in
        )

    def __call__(self, trial):
        classifier_name = trial.suggest_categorical(
            "classifier",
            available_classifiers
        )

        trial_params = {
            "classifier": classifier_name
        }

        if classifier_name == "LogisticRegression":
            C = trial.suggest_float("C", 1e-2, 10, log=True)
            tol = trial.suggest_float("tol", 1e-6, 1e-3, log=True)

            classifier = LogisticRegression(
                C=C,
                tol=tol,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.rng_seed
            )

            trial_params.update({
                "C": C,
                "tol": tol,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": self.rng_seed
            })

        elif classifier_name == "RandomForest":
            max_depth = trial.suggest_int("max_depth", 3, 15)
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=self.rng_seed,
                n_jobs=-1
            )

            trial_params.update({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "class_weight": "balanced",
                "random_state": self.rng_seed
            })

        model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", classifier)
            ]
        )

        # Disable sklearn autolog inside Optuna to avoid duplicate/noisy logs.
        mlflow.sklearn.autolog(disable=True)

        run_name = f"trial_{trial.number}_{classifier_name}"

        with mlflow.start_run(run_name=run_name, nested=False):
            mlflow.set_tag("classifier", classifier_name)
            mlflow.set_tag("run_name", run_name)
            mlflow.set_tag("hpo_tool", "optuna")

            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(trial_params)

            model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)

            f1_binary_value = f1_score(
                self.y_test,
                y_pred,
                average="binary",
                zero_division=0
            )

            f1_weighted_value = f1_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )

            precision_value = precision_score(
                self.y_test,
                y_pred,
                zero_division=0
            )

            recall_value = recall_score(
                self.y_test,
                y_pred,
                zero_division=0
            )

            accuracy_value = accuracy_score(
                self.y_test,
                y_pred
            )

            mlflow.log_metrics({
                "accuracy": accuracy_value,
                "f1_binary": f1_binary_value,
                "f1_weighted": f1_weighted_value,
                "precision": precision_value,
                "recall": recall_value
            })

            trial.set_user_attr("classifier", classifier_name)
            trial.set_user_attr("f1_weighted", f1_weighted_value)

            return f1_weighted_value

# COMMAND ----------

optuna_sampler=optuna.samplers.TPESampler(
    seed=2025
)

# COMMAND ----------

from optuna.pruners import BasePruner

class NoneValuePruner(BasePruner):
    """
    Custom pruner to prune failed trials with None values.
    Useful when distributed HPO has unstable/failed trials.
    """

    def prune(self, study, trial):
        if trial.value is None:
            return True
        return False

optuna_pruner = NoneValuePruner()

# COMMAND ----------

# create study and run 
objective_fun=objective_function(
    x_train_in=X_train,
    y_train_in=y_train,
    preprocessor_in=preprocessor
)

study_1=optuna.create_study(
    direction="maximize",
    study_name="hpo optuna",
    sampler=optuna_sampler,
    pruner=optuna_pruner
)

study_1.optimize(
    objective_fun,
    n_trials=20
)

# COMMAND ----------

# create expoeirment and store these optuna trials and study there. 
current_user=spark.sql("SELECT current_user()").collect()[0][0]
experiment_name=f"/Users/{current_user}/bank_experiment_hpo_optuna"
mlflow.get_experiment_by_name(experiment_name)
experiemnt_id=mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# mlflow storage creation for hpo experiemnts savong 
from mlflow.optuna.storage import MlflowStorage

mlflow_storage = MlflowStorage(
    experiment_id=experiemnt_id
)

# COMMAND ----------

from mlflow.pyspark.optuna.study import MlflowSparkStudy
spark_optuna_available = True

# COMMAND ----------

best_params = dict(study_1.best_params)

best_classifier_name = best_params.pop("classifier")

print("Best classifier:", best_classifier_name)
print("Best params:", best_params)

# COMMAND ----------

if best_classifier_name=="RandomForest":
    best_model=RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=-1,
        n_jobs=-1
    )
    

# COMMAND ----------

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_model)
    ]
)

# COMMAND ----------

mlflow.sklearn.autolog(
    disable=True
)

# COMMAND ----------

with mlflow.start_run(run_name="bank-hpo-best-model") as run():
    mlflow.set_tag("model_type",best_classifier_name)
    mlflow.set_tag("training_type", "optuna_hpo")
    mlflow.log_params(best_params)
    mlflow.log_param("best_classifier", best_classifier_name)
    
    best_pipeline.fit(
        X_train,
        y_train
    )

    mlflow_model=Model()
    mlflow.add_to_model(mlflow_model,loader_module=True)
    pyfunc_model=pyFuncModel(
        model_meta=mlflow_model,
        model_impl=best_pipeline
    )

    
     training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{label_col: y_train}),
        targets=label_col,
        model_type="classifier",
        evaluator_config={
            "log_model_explainability": False,
            "metric_prefix": "training_",
            "pos_label": 1
        }
    )

    # Evaluate test set.
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{label_col: y_test}),
        targets=label_col,
        model_type="classifier",
        evaluator_config={
            "log_model_explainability": True,
            "metric_prefix": "test_",
            "pos_label": 1
        }
    )

    # Manual key metrics for your pipeline quality gate.
    y_test_pred = best_pipeline.predict(X_test)

    if hasattr(best_pipeline, "predict_proba"):
        y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    else:
        test_roc_auc = None

    f1 = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    f1_binary = f1_score(y_test, y_test_pred, average="binary", pos_label=1, zero_division=0)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)

    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("f1_binary", f1_binary)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    if test_roc_auc is not None:
        mlflow.log_metric("roc_auc", test_roc_auc)

    # This is the most important line.
    # fe.log_model logs the model with Feature Store metadata.
    # This enables fe.score_batch() later.
    fe.log_model(
        model=best_pipeline,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=model_name
    )

    print("Final run ID:", final_run_id)
    print("Best classifier:", best_classifier_name)
    print("Weighted F1:", f1)
    print("Binary F1:", f1_binary)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC AUC:", test_roc_auc)