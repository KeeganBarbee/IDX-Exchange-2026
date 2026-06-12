import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
listed = pd.read_csv("listings_residential.csv")

print(f"\nRows: {listed.shape[0]:,}  |  Columns: {listed.shape[1]}")
print("\n--- Column Data Types ---")
print(listed.dtypes.to_string())
print("\n--- First 5 Rows ---")
print(listed.head().to_string())

# Unique Property Types
property_types = listed["PropertyType"].unique()
print(f"\nUnique values ({len(property_types)} total):")
for pt in sorted(property_types, key=lambda x: str(x)):
    count = (listed["PropertyType"] == pt).sum()
    pct   = count / len(listed) * 100
    print(f"  {str(pt):<30} {count:>6,}  ({pct:.1f}%)")

residential_count = (listed["PropertyType"] == "Residential").sum()
other_count       = len(listed) - residential_count
print(f"\nResidential share : {residential_count}  ({residential_count/len(listed)*100:.1f}%)")
print(f"Other share       : {other_count}  ({other_count/len(listed)*100:.1f}%)")

# Filter to Residential
pre_filter  = len(listed)
listed      = listed[listed["PropertyType"] == "Residential"].copy()
post_filter = len(listed)
listed.reset_index(drop=True, inplace=True)

print(f"\nFilter applied : listed['PropertyType'] == 'Residential'")
print(f"Rows before    : {pre_filter:,}")
print(f"Rows after     : {post_filter:,}")
print(f"Rows removed   : {pre_filter - post_filter:,}")

# Date Conversion
date_columns = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
for col in date_columns:
    if col in listed.columns:
        listed[col] = pd.to_datetime(listed[col], errors="coerce")

print("\nDate Checks")
listed["listing_after_close_flag"]   = (listed["ListingContractDate"] > listed["CloseDate"])
listed["purchase_after_close_flag"]  = (listed["PurchaseContractDate"] > listed["CloseDate"])
listed["negative_timeline_flag"]     = (listed["PurchaseContractDate"] < listed["ListingContractDate"])

print(f"Listing after close: {listed['listing_after_close_flag'].sum()}")
print(f"Purchase after close: {listed['purchase_after_close_flag'].sum()}")
print(f"Negative timeline: {listed['negative_timeline_flag'].sum()}")

print("\nGeographic Checks")
listed["missing_coords_flag"]    = (listed["Latitude"].isnull() | listed["Longitude"].isnull())
listed["zero_coords_flag"]       = ((listed["Latitude"] == 0) | (listed["Longitude"] == 0))
listed["positive_longitude_flag"] = (listed["Longitude"] > 0)
listed["out_of_bounds_flag"]     = (
    (listed["Latitude"] < 31.8) | (listed["Latitude"] > 42.2) |
    (listed["Longitude"] < -125.2) | (listed["Longitude"] > -114.8)
)

print(f"Missing coords: {listed['missing_coords_flag'].sum()}")
print(f"Zero coords: {listed['zero_coords_flag'].sum()}")
print(f"Positive longitude errors: {listed['positive_longitude_flag'].sum()}")
print(f"Out-of-bounds coords: {listed['out_of_bounds_flag'].sum()}")

# Drop Redundant Columns
repeats = [x for x in listed.columns if x.endswith(".1")]
names = [
    "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "CoListAgentFirstName", "CoListAgentLastName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId",
    "CoBuyerAgentFirstName", "ListAgentAOR", "BuyerAgentAOR",
    "ListOfficeName", "BuyerOfficeName", "CoListOfficeName",
    "BuyerOfficeAOR", "BuilderName",
]
drops = [x for x in repeats + names if x in listed.columns]
listed.drop(columns=drops, inplace=True)

print(f"\nDropped {len(drops)} columns")
print(f"Columns remaining: {listed.shape[1]}")

# Numeric Type Enforcement
numeric_cols = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "LotSizeArea", "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt", "TaxAnnualAmount", "AssociationFee", "FireplacesTotal",
    "ParkingTotal", "GarageSpaces", "CoveredSpaces", "Stories", "AboveGradeFinishedArea",
    "BelowGradeFinishedArea", "BuildingAreaTotal", "MainLevelBedrooms", "Latitude", "Longitude",
]
for col in numeric_cols:
    if col not in listed.columns:
        continue
    if listed[col].dtype in ["float64", "int64"]:
        print(f"  {col:<35} already numeric ({listed[col].dtype})")
        continue
    before_nulls = listed[col].isnull().sum()
    listed[col]  = pd.to_numeric(listed[col], errors="coerce")
    new_nulls    = listed[col].isnull().sum() - before_nulls
    print(f"  {col:<35} converted to {listed[col].dtype}  |  {new_nulls} new nulls")

# Drop Rows Missing Critical Fields
listed = listed.dropna(subset=["ClosePrice", "LivingArea", "Latitude", "Longitude"])

print("\nNumeric Validation")
before_rows = len(listed)

listed["invalid_closeprice_flag"]  = listed["ClosePrice"] <= 0
listed["invalid_livingarea_flag"]  = listed["LivingArea"] <= 0
listed["invalid_dom_flag"]         = listed["DaysOnMarket"] < 0
listed["invalid_bedrooms_flag"]    = listed["BedroomsTotal"] < 0
listed["invalid_bathrooms_flag"]   = listed["BathroomsTotalInteger"] < 0

invalid_mask = (
    (listed["ClosePrice"] <= 0) | (listed["LivingArea"] <= 0) |
    (listed["DaysOnMarket"] < 0) | (listed["BedroomsTotal"] < 0) |
    (listed["BathroomsTotalInteger"] < 0)
)

print(f"Invalid rows detected: {invalid_mask.sum()}")
listed      = listed[~invalid_mask].copy()
after_rows  = len(listed)
print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning : {after_rows}")
print(f"Rows removed        : {before_rows - after_rows}")


# Outlier Detection
iqr_fields     = ["ClosePrice", "LivingArea", "DaysOnMarket"]
medians_before = {field: listed[field].median() for field in iqr_fields}
rows_before    = len(listed)

for field in iqr_fields:
    q1    = listed[field].quantile(0.25)
    q3    = listed[field].quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    listed[f"{field}_outlier_flag"] = (
        (listed[field] < lower) | (listed[field] > upper)
    )
    print(f"{field}: Q1={q1}, Q3={q3}, IQR={iqr}, Lower={lower}, Upper={upper}")
    print(f"  Outliers flagged: {listed[f'{field}_outlier_flag'].sum()}")

listed.to_csv("listed_flagged.csv", index=False)
print(f"Rows: {len(listed)}, Columns: {listed.shape[1]}")

any_outlier  = (
    listed["ClosePrice_outlier_flag"] |
    listed["LivingArea_outlier_flag"] |
    listed["DaysOnMarket_outlier_flag"]
)
listed_clean = listed[~any_outlier].copy()
listed_clean.to_csv("listed_clean.csv", index=False)
print(f"Rows: {len(listed_clean)}, Columns: {listed_clean.shape[1]}")

print("\nBefore vs After")
for field in iqr_fields:
    print(f"{field} {medians_before[field]} {listed[field].median()} {rows_before} {len(listed)}")

print("\nTotal Data Summary")
print(f"Final rows: {len(listed)}")
print("\nData types:")
print(listed.dtypes)
print("\nDate flag summary:")
print(listed[["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag"]].sum())
print("\nGeographic flag summary:")
print(listed[["missing_coords_flag", "zero_coords_flag", "positive_longitude_flag", "out_of_bounds_flag"]].sum())

# Missing Value Analysis
null_counts  = listed.isnull().sum()
null_pct     = (null_counts / len(listed) * 100).round(2)
null_summary = pd.DataFrame({
    "missing_count": null_counts,
    "missing_pct"  : null_pct,
    "present_count": len(listed) - null_counts,
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
    "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]
percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

for field in numeric_fields:
    if field not in listed.columns:
        print(f"\n[WARNING] '{field}' not found — skipping.")
        continue
    s = listed[field].dropna()
    print(f"\n--- {field} ---")
    print(f"  Count   : {len(s):,}")
    print(f"  Missing : {listed[field].isnull().sum():,}")
    print(f"  Min     : {s.min():,.4f}")
    print(f"  Max     : {s.max():,.4f}")
    print(f"  Mean    : {s.mean():,.4f}")
    print(f"  Median  : {s.median():,.4f}")
    print(f"  Std Dev : {s.std():,.4f}")
    print(f"  Percentiles:")
    for p in percentiles:
        print(f"    p{int(p*100):>2}  : {s.quantile(p):,.4f}")

# EDA Questions
if "ListPrice" in listed.columns:
    print(f"\nMedian List Price : ${listed['ListPrice'].median():,.2f}")
    print(f"Average List Price: ${listed['ListPrice'].mean():,.2f}")

if "DaysOnMarket" in listed.columns:
    dom = listed["DaysOnMarket"].dropna()
    print(f"\nDays on Market:")
    print(f"  Median : {dom.median():.0f} days")
    print(f"  Mean   : {dom.mean():.1f} days")
    print(f"  Max    : {dom.max():.0f} days")
    print(f"  Within 30 days : {(dom <= 30).sum()/len(dom)*100:.1f}%")
    print(f"  Within 90 days : {(dom <= 90).sum()/len(dom)*100:.1f}%")

if "ListPrice" in listed.columns and "OriginalListPrice" in listed.columns:
    valid   = listed[["ListPrice", "OriginalListPrice"]].dropna()
    reduced = (valid["ListPrice"] < valid["OriginalListPrice"]).sum()
    same    = (valid["ListPrice"] == valid["OriginalListPrice"]).sum()
    raised  = (valid["ListPrice"] > valid["OriginalListPrice"]).sum()
    total   = len(valid)
    print(f"\nList Price vs Original List Price (n={total:,}):")
    print(f"  Price reduced : {reduced:,}  ({reduced/total*100:.1f}%)")
    print(f"  No change     : {same:,}  ({same/total*100:.1f}%)")
    print(f"  Price raised  : {raised:,}  ({raised/total*100:.1f}%)")

if "ListingContractDate" in listed.columns and "ExpirationDate" in listed.columns:
    listed["ExpirationDate"] = pd.to_datetime(listed["ExpirationDate"], errors="coerce")
    bad = listed[listed["ExpirationDate"] < listed["ListingContractDate"]]
    print(f"\nDate Consistency:")
    print(f"  ExpirationDate before ListingContractDate: {len(bad):,} record(s)")
    if len(bad) > 0:
        print("  [FLAG] Date inconsistencies detected.")

if "ListPrice" in listed.columns and "CountyOrParish" in listed.columns:
    top_counties = (
        listed.groupby("CountyOrParish")["ListPrice"]
        .median().sort_values(ascending=False).head(10)
    )
    print(f"\nTop 10 Counties by Median List Price:")
    for county, med in top_counties.items():
        print(f"  {str(county):<30} ${med:,.0f}")

# Plots
plot_fields = ["ClosePrice", "LivingArea", "DaysOnMarket"]

for field in plot_fields:
    if field not in listed.columns:
        continue
    series = listed[field].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LISTED — {field}", fontsize=14, fontweight="bold")
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
listed["price_ratio"]     = listed["ClosePrice"] / listed["OriginalListPrice"]
listed["price_per_sqft"]  = listed["ClosePrice"] / listed["LivingArea"]
listed["close_year"]      = listed["CloseDate"].dt.year
listed["close_month"]     = listed["CloseDate"].dt.month
listed["close_yrmo"]      = listed["CloseDate"].dt.to_period("M")

if "PurchaseContractDate" in listed.columns and "ListingContractDate" in listed.columns:
    listed["listing_to_contract_days"] = (
        listed["PurchaseContractDate"] - listed["ListingContractDate"]
    ).dt.days

if "PurchaseContractDate" in listed.columns and "CloseDate" in listed.columns:
    listed["contract_to_close_days"] = (
        listed["CloseDate"] - listed["PurchaseContractDate"]
    ).dt.days

sample_cols = [c for c in [
    "ClosePrice", "OriginalListPrice", "LivingArea", "price_ratio", "price_per_sqft",
    "close_year", "close_month", "close_yrmo", "listing_to_contract_days", "contract_to_close_days",
] if c in listed.columns]
print(listed[sample_cols].head(10).to_string())

seg_cols = [c for c in [
    "price_ratio", "price_per_sqft", "DaysOnMarket",
    "listing_to_contract_days", "contract_to_close_days",
] if c in listed.columns]

if "CountyOrParish" in listed.columns and seg_cols:
    county_summary = (
        listed.groupby("CountyOrParish")[seg_cols]
        .median().round(2).sort_values("price_per_sqft", ascending=False)
    )
    print(county_summary.to_string())

if "MLSAreaMajor" in listed.columns and seg_cols:
    mls_summary = (
        listed.groupby("MLSAreaMajor")[seg_cols]
        .median().round(2).sort_values("price_per_sqft", ascending=False)
    )
    print(mls_summary.to_string())

listed.to_csv("listed_featured.csv", index=False)
print(f"\nFeature engineered dataset saved -> listed_featured.csv")
print(f"  Rows: {len(listed):,}  |  Columns: {listed.shape[1]}")