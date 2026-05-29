# Databricks notebook source
import os
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# COMMAND ----------

import importlib.util

spec = importlib.util.find_spec("mlflow")

print("MLflow origin:", spec.origin)
print("MLflow locations:", spec.submodule_search_locations)

# COMMAND ----------

silver_table="bank.silver.data_cleaned"
feature_table_name="bank.feature_store.customer_features"
registered_model_name="bank.model.subscription_classifier"

# COMMAND ----------

silver_df = spark.table(silver_table)
feature_cols = [
    "age_group",
    "balance_tier",
    "contact_intensity",
    "was_previously_contacted",
    "recency_score",
    "duration_bucket",
    "financial_stress"
]
label_cols="y_flag"

training_spark_df=silver_df.select(*feature_cols,label_cols)

# COMMAND ----------

training_df=training_spark_df.toPandas()
X=training_df[feature_cols]
Y=training_df[label_cols]
print("row count for X:",X.count())
print("row count for Y:",Y.count())
print("shape of X:",X.shape)
print("shape of Y:",Y.shape)
print("class distributuion:",Y.value_counts())

# COMMAND ----------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:")
print(Y_train.value_counts(normalize=True))
print("y_test distribution:")
print(Y_test.value_counts(normalize=True))

# COMMAND ----------

# settin mlflow experiments
current_user=spark.sql("SELECT current_user()").collect()[0][0]
experiment_name=f"/Users/{current_user}/bank_subscription_experiment"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

X=training_df.drop(columns="y_flag")
Y=training_df["y_flag"]
categorical_cols=X.select_dtypes(include=[
    "category",
    "string",
    "object"
]).columns.tolist()
numeric_cols = X.select_dtypes(
    include=["int64", "float64", "int32", "float32", "bool"]
).columns.tolist()

processor_with_scaling=ColumnTransformer(
    transformers=[
        ("categorical",OneHotEncoder(),categorical_cols),
        ("numerical",StandardScaler(),numeric_cols)
    ]
)
processor_without_scaling=ColumnTransformer(
    transformers=[
        ("categorical",OneHotEncoder,categorical_cols)
    ]
)

# COMMAND ----------

def evaluate_and_log_metrics(model, X_test, y_test, artifact_prefix):
    """
    Evaluates a trained model and logs important classification metrics to MLflow.

    artifact_prefix is used to name plot files differently for each model.
    """
    y_pred=model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    f1_wt=f1_score(y_test,y_pred,average="weighted")

    # metrics logging 
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.log_metric("f1_score_wighted",f1_wt)

    if roc_auc is not None:
        mlflow.log_metric("roc_auc", roc_auc)
    print(classification_report(y_test, y_pred))
    cm=confusion_matrix(y_test,y_pred)
    cm_display=ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm_display.plot(ax=ax)
    plt.title(f"{artifact_prefix} Confusion Matrix")


    cm_path = f"/tmp/{artifact_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(cm_path)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_binary": f1,
        "f1_weighted": f1_wt,
        "roc_auc": roc_auc
    }




# COMMAND ----------

# logistic regression with baseline model also with autologging

mlflow.sklearn.autolog(log_models=False)
with mlflow.start_run(run_name="logistic_regression_baseline") as run:
    mlflow.set_tag("model_type","logistic_regression")
    
    lr_model=Pipeline(
        steps=[
            ("preprocessor",processor_with_scaling),
            ("classifier",LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ))
        ]
    )

    lr_model.fit(X_train,Y_train)
    lr_metrics=evaluate_and_log_metrics(lr_model,X_test,Y_test,"logistic_regression")

    input_example=X_test.head(5)
    prediction_sample=lr_model.predict(input_example)
    signature=infer_signature(input_example,prediction_sample)

    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print("logistic regression per run:",run.info.run_id)
    print(lr_metrics)





# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

# Correct preprocessor instance
processor_without_scaling = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("numeric", "passthrough", numeric_cols)
    ]
)

mlflow.sklearn.autolog(log_models=False)

with mlflow.start_run(run_name="random_forest_v1") as run:
    mlflow.set_tag("model", "random_forest")

    params = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "max_depth": 10,
        "random_state": 42
    }

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_params(params)

    rf_model = Pipeline(
        steps=[
            ("preprocessor", processor_without_scaling),
            ("classifier", RandomForestClassifier(**params))
        ]
    )

    rf_model.fit(X_train, Y_train)

    rf_metrics = evaluate_and_log_metrics(
        rf_model,
        X_test,
        Y_test,
        "random_forest"
    )

    encoded_feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

    importances = rf_model.named_steps["classifier"].feature_importances_

    feature_importance_df = pd.DataFrame({
        "feature": encoded_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["feature"].head(15)[::-1],
        feature_importance_df["importance"].head(15)[::-1]
    )
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance - Top 15")

    fi_path = "/tmp/random_forest_feature_importance.png"
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(fi_path)

    input_example = X_test.head(5)
    predictions_example = rf_model.predict(input_example)
    signature = infer_signature(input_example, predictions_example)

    mlflow.sklearn.log_model(
        sk_model=rf_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    print("Random Forest run_id:", run.info.run_id)
    print(rf_metrics)

    display(feature_importance_df.head(20))

# COMMAND ----------

runs_df=mlflow.search_runs(
    experiment_names=[experiment_name],
    order_by=["metric.f1_wt"]
)

cols_to_show = [
    "run_id",
    "tags.mlflow.runName",
    "tags.model_type",
    "metrics.f1_wt",
    "metrics.f1",
    "metrics.roc_auc",
    "metrics.precision",
    "metrics.recall",
    "metrics.accuracy"
]

available_cols=[c for c in cols_to_show if c in runs_df.columns]
best_run_df=runs_df[available_cols].head(10)
best_run_df

# COMMAND ----------

