import pandas as pd
import os
import ast
from features.feature_table import (
    CKP_SOURCE,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)


def test():
    source = CKP_SOURCE.ALPHA_WORDS
    rows = create_full_user_and_platform_table(source)
    for row in rows:
        print(row)
        input("Current row")
    cleaned = table_to_cleaned_df(rows, source)
    cleaned.to_csv("small_features_data.csv")
    df = pd.read_csv(os.path.join(os.getcwd(), "small_features_data.csv"))
    nan_counts = df.isna().sum()
    print(nan_counts)

    # # df = df.dropna()
    # # print(df)
    # for row in df.index:
    #     for col in df.columns:
    #         if isinstance(df.at[row, col], float):
    #             pass
    #             # print(f"Row {row}, Column '{col}': {df.at[row, col]}")
    #         try:
    #             ast.literal_eval(df.at[row, col])
    #         except Exception:
    #             # print(f"Row {row}, Column '{col}': {df.at[row, col]}")


def test_2():
    df = pd.read_csv(os.path.join(os.getcwd(), "before_cleaning.csv"))
    print(df)
    columns_without_nans = df.dropna(axis=1).columns.tolist()
    print(columns_without_nans)
    columns_all_nans = df.columns[df.isna().all()].tolist()

    print(columns_all_nans)
    # Initialize an empty dictionary to store non-NaN values
    non_nan_values = {}

    # Iterate through each column
    for col in df.columns:
        # Check if column contains any NaN values
        if df[col].isna().any():
            # Get non-NaN values and add them to the dictionary
            non_nan_values[col] = df[col].dropna().tolist()
    print(non_nan_values)


def test_3():
    df = pd.read_csv("post_fill.csv")
    print(df.iloc[0].to_dict())


test_3()
