import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sold = pd.read_csv("sold_residential.csv", low_memory=False)
listed = pd.read_csv("listings_residential.csv", low_memory=False)

print("Dataset Structure is as follows:")
print(f"\n Rows for Sold Dataset: {sold.shape[0]}")
print(f"\n Columns for Sold Dataset: {sold.shape[1]}")
print(f"\n Rows for Listed Dataset: {listed.shape[0]}")
print(f"\n Columns for Listed Dataset: {listed.shape[1]}")

print("\n Column Types for the Sold Dataset")
print(sold.dtypes.to_string())
print("\n Column Types for the Listed Dataset")
print(listed.dtypes.to_string())

properties = sold["PropertyType"].unique()
print("\n The unique Property Types:")
print("\n", properties)
print("\n ^^^ Same will be true of the Listed Dataset as we already filtered for PropertyType")

sold_null = sold.isnull().sum()
sold_null_pct = ((sold_null / len(sold)) * 100).round(2)
sold_notnull = len(sold) - sold_null
sold_null_table = pd.DataFrame({"Missing_Counts": sold_null, 
                               "Missing_Pct": sold_null_pct, 
                               "Non_Missing_Counts": sold_notnull}).sort_values("Missing_Pct", ascending = False)

listed_null = listed.isnull().sum()
listed_null_pct = ((listed_null / len(listed)) * 100).round(2)
listed_notnull = len(listed) - listed_null
listed_null_table = pd.DataFrame({"Missing_Counts": listed_null, 
                               "Missing_Pct": listed_null_pct, 
                               "Non_Missing_Counts": listed_notnull}).sort_values("Missing_Pct", ascending = False)

print(f"\n Sold Dataset Null Table: {sold_null}")
print(f"\n Listed Dataset Null Table: {listed_null}")

high_miss = sold_null_table[sold_null_table["Missing_Pct"] > 90]
if high_miss.empty:
    print("\nNo columns exceed 90% missing values.")
else:
    print(f"\n{len(high_miss)} column(s) flagged:\n")
    for col, row in high_miss.iterrows():
        print(f"{col:<40} {row['Missing_Pct']:>9.1f}%")
    print("\nRecommendation: Consider dropping these columns unless")
    print("they are core fields required for downstream analysis.")

high_miss_listed = listed_null_table[listed_null_table["Missing_Pct"] > 90]
if high_miss_listed.empty:
    print("\nNo columns exceed 90% missing values.")
else:
    print(f"\n{len(high_miss_listed)} column(s) flagged:\n")
    for col, row in high_miss_listed.iterrows():
        print(f"{col:<40} {row['Missing_Pct']:>9.1f}%")
    print("\nRecommendation: Consider dropping these columns unless")
    print("they are core fields required for downstream analysis.")