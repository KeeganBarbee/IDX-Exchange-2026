import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Load Data
sold = pd.read_csv("sold_residential.csv")
 
print(f"\nRows: {sold.shape[0]:,}  |  Columns: {sold.shape[1]}")
print("\n--- Column Data Types ---")
print(sold.dtypes.to_string())
print("\n--- First 5 Rows ---")
print(sold.head().to_string())
 
# Unique Property Types
property_types = sold["PropertyType"].unique()
print(f"\nUnique values ({len(property_types)} total):")
for pt in sorted(property_types, key=lambda x: str(x)):
    count = (sold["PropertyType"] == pt).sum()
    pct   = count / len(sold) * 100
    print(f"  {str(pt):<30} {count:>6,}  ({pct:.1f}%)")
 
residential_count = (sold["PropertyType"] == "Residential").sum()
other_count       = len(sold) - residential_count
print(f"\nResidential share : {residential_count}  ({residential_count/len(sold)*100:.1f}%)")
print(f"Other share       : {other_count}  ({other_count/len(sold)*100:.1f}%)")
 
# Filter to Residential
pre_filter  = len(sold)
sold        = sold[sold["PropertyType"] == "Residential"].copy()
post_filter = len(sold)
sold.reset_index(drop=True, inplace=True)
 
print(f"\nFilter applied : sold['PropertyType'] == 'Residential'")
print(f"Rows before    : {pre_filter:,}")
print(f"Rows after     : {post_filter:,}")
print(f"Rows removed   : {pre_filter - post_filter:,}")
 
# Date Conversion
date_columns = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
for col in date_columns:
    if col in sold.columns:
        sold[col] = pd.to_datetime(sold[col], errors="coerce")
 
print("\nDate Checks")
sold["listing_after_close_flag"]  = (sold["ListingContractDate"] > sold["CloseDate"])
sold["purchase_after_close_flag"] = (sold["PurchaseContractDate"] > sold["CloseDate"])
sold["negative_timeline_flag"]    = (sold["PurchaseContractDate"] < sold["ListingContractDate"])
 
print(f"Listing after close: {sold['listing_after_close_flag'].sum()}")
print(f"Purchase after close: {sold['purchase_after_close_flag'].sum()}")
print(f"Negative timeline: {sold['negative_timeline_flag'].sum()}")
 
print("\nGeographic Checks")
sold["missing_coords_flag"]     = (sold["Latitude"].isnull() | sold["Longitude"].isnull())
sold["zero_coords_flag"]        = ((sold["Latitude"] == 0) | (sold["Longitude"] == 0))
sold["positive_longitude_flag"] = (sold["Longitude"] > 0)
sold["out_of_bounds_flag"]      = (
    (sold["Latitude"] < 31.8) | (sold["Latitude"] > 42.2) |
    (sold["Longitude"] < -125.2) | (sold["Longitude"] > -114.8)
)
 
print(f"Missing coords: {sold['missing_coords_flag'].sum()}")
print(f"Zero coords: {sold['zero_coords_flag'].sum()}")
print(f"Positive longitude errors: {sold['positive_longitude_flag'].sum()}")
print(f"Out-of-bounds coords: {sold['out_of_bounds_flag'].sum()}")
 
# Drop Redundant Columns
repeats = [x for x in sold.columns if x.endswith(".1")]
names = [
    "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "CoListAgentFirstName", "CoListAgentLastName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId",
    "CoBuyerAgentFirstName", "ListAgentAOR", "BuyerAgentAOR",
    "ListOfficeName", "BuyerOfficeName", "CoListOfficeName",
    "BuyerOfficeAOR", "BuilderName",
]
drops = [x for x in repeats + names if x in sold.columns]
sold.drop(columns=drops, inplace=True)
 
print(f"\nDropped {len(drops)} columns")
print(f"Columns remaining: {sold.shape[1]}")
 
# Numeric Type Enforcement
numeric_cols = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "LotSizeArea", "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt", "TaxAnnualAmount", "AssociationFee", "FireplacesTotal",
    "ParkingTotal", "GarageSpaces", "CoveredSpaces", "Stories", "AboveGradeFinishedArea",
    "BelowGradeFinishedArea", "BuildingAreaTotal", "MainLevelBedrooms", "Latitude", "Longitude",
]
for col in numeric_cols:
    if col not in sold.columns:
        continue
    if sold[col].dtype in ["float64", "int64"]:
        print(f"  {col:<35} already numeric ({sold[col].dtype})")
        continue
    before_nulls = sold[col].isnull().sum()
    sold[col]    = pd.to_numeric(sold[col], errors="coerce")
    new_nulls    = sold[col].isnull().sum() - before_nulls
    print(f"  {col:<35} converted to {sold[col].dtype}  |  {new_nulls} new nulls")
 
# Drop Rows Missing Critical Fields
sold = sold.dropna(subset=["ClosePrice", "LivingArea", "Latitude", "Longitude"])
 
print("\nNumeric Validation")
before_rows = len(sold)
 
sold["invalid_closeprice_flag"]  = sold["ClosePrice"] <= 0
sold["invalid_livingarea_flag"]  = sold["LivingArea"] <= 0
sold["invalid_dom_flag"]         = sold["DaysOnMarket"] < 0
sold["invalid_bedrooms_flag"]    = sold["BedroomsTotal"] < 0
sold["invalid_bathrooms_flag"]   = sold["BathroomsTotalInteger"] < 0
 
invalid_mask = (
    (sold["ClosePrice"] <= 0) | (sold["LivingArea"] <= 0) |
    (sold["DaysOnMarket"] < 0) | (sold["BedroomsTotal"] < 0) |
    (sold["BathroomsTotalInteger"] < 0)
)
 
print(f"Invalid rows detected: {invalid_mask.sum()}")
sold       = sold[~invalid_mask].copy()
after_rows = len(sold)
print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning : {after_rows}")
print(f"Rows removed        : {before_rows - after_rows}")
 
 
# Outlier Detection
iqr_fields     = ["ClosePrice", "LivingArea", "DaysOnMarket"]
medians_before = {field: sold[field].median() for field in iqr_fields}
rows_before    = len(sold)
 
for field in iqr_fields:
    q1    = sold[field].quantile(0.25)
    q3    = sold[field].quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    sold[f"{field}_outlier_flag"] = (
        (sold[field] < lower) | (sold[field] > upper)
    )
    print(f"{field}: Q1={q1}, Q3={q3}, IQR={iqr}, Lower={lower}, Upper={upper}")
    print(f"  Outliers flagged: {sold[f'{field}_outlier_flag'].sum()}")
 
sold.to_csv("sold_flagged.csv", index=False)
print(f"Rows: {len(sold)}, Columns: {sold.shape[1]}")
 
any_outlier = (
    sold["ClosePrice_outlier_flag"] |
    sold["LivingArea_outlier_flag"] |
    sold["DaysOnMarket_outlier_flag"]
)
sold_clean = sold[~any_outlier].copy()
sold_clean.to_csv("sold_clean.csv", index=False)
print(f"Rows: {len(sold_clean)}, Columns: {sold_clean.shape[1]}")
 
print("\nBefore vs After")
for field in iqr_fields:
    print(f"{field} {medians_before[field]} {sold[field].median()} {rows_before} {len(sold)}")
 
print("\nTotal Data Summary")
print(f"Final rows: {len(sold)}")
print("\nData types:")
print(sold.dtypes)
print("\nDate flag summary:")
print(sold[["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag"]].sum())
print("\nGeographic flag summary:")
print(sold[["missing_coords_flag", "zero_coords_flag", "positive_longitude_flag", "out_of_bounds_flag"]].sum())
 
# Missing Value Analysis
null_counts  = sold.isnull().sum()
null_pct     = (null_counts / len(sold) * 100).round(2)
null_summary = pd.DataFrame({
    "missing_count": null_counts,
    "missing_pct"  : null_pct,
    "present_count": len(sold) - null_counts,
}).sort_values("missing_pct", ascending=False)
 
print(f"\n{'Column':<40} {'Missing':>8} {'Missing %':>10} {'Present':>8}")
print("-" * 70)
for col, row in null_summary.iterrows():
    flag = "  *** >90% ***" if row["missing_pct"] > 90 else ""
    print(f"{col:<40} {int(row['missing_count']):>8,} "
          f"{row['missing_pct']:>9.1f}% {int(row['present_count']):>8,}{flag}")
 
high_missing = null_summary[null_summary["missing_pct"] > 90]
if high_missing.empty:
    print("\nNo columns exceed 90% missing.")
else:
    print(f"\n{len(high_missing)} column(s) flagged:\n")
    print(f"{'Column':<40} {'Missing %':>10}")
    print("-" * 52)
    for col, row in high_missing.iterrows():
        print(f"{col:<40} {row['missing_pct']:>9.1f}%")
    print("\nRecommendation: consider dropping unless required downstream.")
 
# Numeric Distribution Summary
numeric_fields = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]
percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
 
for field in numeric_fields:
    if field not in sold.columns:
        print(f"\n[WARNING] '{field}' not found — skipping.")
        continue
    s = sold[field].dropna()
    print(f"\n--- {field} ---")
    print(f"  Count   : {len(s):,}")
    print(f"  Missing : {sold[field].isnull().sum():,}")
    print(f"  Min     : {s.min():,.2f}")
    print(f"  Max     : {s.max():,.2f}")
    print(f"  Mean    : {s.mean():,.2f}")
    print(f"  Median  : {s.median():,.2f}")
    print(f"  Std Dev : {s.std():,.2f}")
    print(f"  Percentiles:")
    for p in percentiles:
        print(f"    p{int(p*100):>2}  : {s.quantile(p):,.2f}")
 
# EDA Questions
if "ClosePrice" in sold.columns:
    print(f"\nMedian Close Price : ${sold['ClosePrice'].median():,.2f}")
    print(f"Average Close Price: ${sold['ClosePrice'].mean():,.2f}")
 
if "DaysOnMarket" in sold.columns:
    dom = sold["DaysOnMarket"].dropna()
    print(f"\nDays on Market:")
    print(f"  Median : {dom.median():.0f} days")
    print(f"  Mean   : {dom.mean():.1f} days")
    print(f"  Max    : {dom.max():.0f} days")
    print(f"  Within 30 days : {(dom <= 30).sum()/len(dom)*100:.1f}%")
    print(f"  Within 90 days : {(dom <= 90).sum()/len(dom)*100:.1f}%")
 
if "ClosePrice" in sold.columns and "ListPrice" in sold.columns:
    valid   = sold[["ClosePrice", "ListPrice"]].dropna()
    above   = (valid["ClosePrice"] > valid["ListPrice"]).sum()
    below   = (valid["ClosePrice"] < valid["ListPrice"]).sum()
    at_list = (valid["ClosePrice"] == valid["ListPrice"]).sum()
    total   = len(valid)
    print(f"\nSold vs List Price (n={total:,}):")
    print(f"  Above list : {above:,}  ({above/total*100:.1f}%)")
    print(f"  At list    : {at_list:,}  ({at_list/total*100:.1f}%)")
    print(f"  Below list : {below:,}  ({below/total*100:.1f}%)")
 
if all(c in sold.columns for c in ["CloseDate", "ListingContractDate"]):
    bad_dates = sold[sold["CloseDate"] < sold["ListingContractDate"]]
    print(f"\nDate Consistency:")
    print(f"  CloseDate before ListingContractDate: {len(bad_dates):,} records")
    if len(bad_dates) > 0:
        print("  [FLAG] Date inconsistencies detected.")
 
if "ClosePrice" in sold.columns and "CountyOrParish" in sold.columns:
    top_counties = (
        sold.groupby("CountyOrParish")["ClosePrice"]
        .median().sort_values(ascending=False).head(10)
    )
    print(f"\nTop 10 Counties by Median Close Price:")
    for county, med in top_counties.items():
        print(f"  {str(county):<30} ${med:,.0f}")
 
# Plots
plot_fields = ["ClosePrice", "LivingArea", "DaysOnMarket"]
 
for field in plot_fields:
    if field not in sold.columns:
        continue
    series = sold[field].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"SOLD — {field}", fontsize=14, fontweight="bold")
    axes[0].hist(series, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel(field)
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(series.median(), color="red", linestyle="--", label="Median")
    axes[0].axvline(series.mean(), color="orange", linestyle="--", label="Mean")
    axes[0].legend()
    axes[1].boxplot(
        series, vert=True, patch_artist=True,
        boxprops=dict(facecolor="steelblue", color="navy"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_title("Boxplot")
    axes[1].set_ylabel(field)
    plt.tight_layout()
    plt.show()
 
# Feature Engineering
sold["price_ratio"]    = sold["ClosePrice"] / sold["OriginalListPrice"]
sold["price_per_sqft"] = sold["ClosePrice"] / sold["LivingArea"]
sold["close_year"]     = sold["CloseDate"].dt.year
sold["close_month"]    = sold["CloseDate"].dt.month
sold["close_yrmo"]     = sold["CloseDate"].dt.to_period("M")
 
if "PurchaseContractDate" in sold.columns and "ListingContractDate" in sold.columns:
    sold["listing_to_contract_days"] = (
        sold["PurchaseContractDate"] - sold["ListingContractDate"]
    ).dt.days
 
if "PurchaseContractDate" in sold.columns and "CloseDate" in sold.columns:
    sold["contract_to_close_days"] = (
        sold["CloseDate"] - sold["PurchaseContractDate"]
    ).dt.days
 
sample_cols = [c for c in [
    "ClosePrice", "OriginalListPrice", "LivingArea", "price_ratio", "price_per_sqft",
    "close_year", "close_month", "close_yrmo", "listing_to_contract_days", "contract_to_close_days",
] if c in sold.columns]
print(sold[sample_cols].head(10).to_string())
 
seg_cols = [c for c in [
    "price_ratio", "price_per_sqft", "DaysOnMarket",
    "listing_to_contract_days", "contract_to_close_days",
] if c in sold.columns]
 
if "CountyOrParish" in sold.columns and seg_cols:
    county_summary = (
        sold.groupby("CountyOrParish")[seg_cols]
        .median().round(2).sort_values("price_per_sqft", ascending=False)
    )
    print(county_summary.to_string())
 
if "MLSAreaMajor" in sold.columns and seg_cols:
    mls_summary = (
        sold.groupby("MLSAreaMajor")[seg_cols]
        .median().round(2).sort_values("price_per_sqft", ascending=False)
    )
    print(mls_summary.to_string())
 
sold.to_csv("sold_featured.csv", index=False)
print(f"\nFeature engineered dataset saved -> sold_featured.csv")
print(f"  Rows: {len(sold):,}  |  Columns: {sold.shape[1]}")