import pandas as pd
import os
import ast
from features.feature_table import (
    CKP_SOURCE,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)

source = CKP_SOURCE.ALPHA_WORDS
rows = create_full_user_and_platform_table(source)
cleaned = table_to_cleaned_df(rows, source)
cleaned.to_csv("kht_and_kit_features_data.csv")
# df = pd.read_csv(os.path.join(os.getcwd(), "kht_and_kit_features_data.csv"))
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
