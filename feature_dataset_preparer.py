import ast
import os
import json

from classifiers.ml_models import run_random_forest_model, run_xgboost_model
from features.feature_table import (
    CKP_SOURCE,
    columns_to_remove,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)
import pandas as pd
import matplotlib.pyplot as plt

# This is the csv saving logic for the features, we shouldn't have to always run this unless we change this with the users
# or the features
# We will also need to rerun this if we ever change the source because currently the columns are fixed
#################
# source = CKP_SOURCE.ALPHA_WORDS
# rows = create_full_user_and_platform_table(source)
# cleaned = table_to_cleaned_df(rows, source)
# cleaned.to_csv("alpha_features_data.csv", mode='w+')
#################


# Deserialization of columns
def deserialize_column(df, column_name):
    df[column_name] = df[column_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


# Flattening the columns with lists into separate columns
def flatten_column(df, column_name):
    new_cols = pd.DataFrame(df[column_name].tolist(), index=df.index)
    new_cols.columns = [
        f"{column_name}_median",
        f"{column_name}_mean",
        f"{column_name}_stdev",
        f"{column_name}_q1",
        f"{column_name}_q3",
    ]

    df = pd.concat([df.drop(columns=[column_name]), new_cols], axis=1)
    return df


df = pd.read_csv(os.path.join(os.getcwd(), "kht_and_kit_features_data.csv"))
df = df.dropna()
df.drop(columns_to_remove(), inplace=True, axis=1)
# print(list(df.columns))
# input()
with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)
# print(df.columns)
# input("Printing dataframe columns")
# Columns to deserialize
# NOTE: deserialization here is different than dropping the unnecessary columns before they are passed to the model.
#       Here deserialization is to make sure the feature lists get reinterpreted from str to python lists
#       But we are not removing them from the df here because we still need them to setup the experiments
columns_to_deserialize = list(df.columns)
columns_to_deserialize.remove("user_id")
columns_to_deserialize.remove("platform_id")
# Apply deserialization and flattening to each relevant column
for col in columns_to_deserialize:
    df = deserialize_column(df, col)
    df = flatten_column(df, col)

# Converting 'user_id' and 'platform_id' to numeric values
df["user_id"] = df["user_id"].apply(
    lambda x: int(ast.literal_eval(x)[0]) if isinstance(x, str) else x
)
df["platform_id"] = df["platform_id"].apply(
    lambda x: int(ast.literal_eval(x)[0]) if isinstance(x, str) else x
)
# print(list(df.columns))
# input()
df.to_csv("cleaned_features_data.csv", mode='w+')

experiments = [
    # Dual-platform training tests (original ones)
    ([1, 2], 3, "FI vs. T"),
    # ([1, 3], 2, "FT vs. I"),
    # ([2, 1], 3, "IF vs. T"),
    # ([2, 3], 1, "IT vs. F"),
    # ([3, 1], 2, "TF vs. I"),
    # ([3, 2], 1, "TI vs. F"),
    # Single-platform training tests
    ([1], 2, "F vs. I"),
    ([1], 3, "F vs. T"),
    ([2], 1, "I vs. F"),
    ([2], 3, "I vs. T"),
    ([3], 1, "T vs. F"),
    ([3], 2, "T vs. I"),
]

for train_platforms, test_platform, experiment_name in experiments:
    print(experiment_name)

    X_train = df[df["platform_id"].isin(list(train_platforms))].drop(
        columns=["platform_id", "user_id"], errors="raise", axis=1
    )
    # print(X_train.columns)
    # input()
    y_train = df[df["platform_id"].isin(list(train_platforms))]["user_id"]

    X_test = df[df["platform_id"] == test_platform].drop(
        columns=["platform_id", "user_id"], errors="raise", axis=1
    )
    y_test = df[df["platform_id"] == test_platform]["user_id"]

    # print("Number of samples per class in training set:")
    # print(y_train.value_counts())
    # print("Number of samples per class in testing set:")
    # print(y_test.value_counts())
    # input()
    # Plot class distribution if needed
    if config["show_class_distributions"]:
        y_train.value_counts().plot(kind="bar", title=f"Train {experiment_name}")
        plt.show()
        y_test.value_counts().plot(kind="bar", title=f"Test {experiment_name}")
        plt.show()

    run_random_forest_model(X_train, X_test, y_train, y_test)

    input(f"{experiment_name} results")
