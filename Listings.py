import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
 
# Load Data 
listed = pd.read_csv("listings_residential.csv")
# Dataset Structure
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
print(f"\nResidential share : {residential_count:,}  ({residential_count/len(listed)*100:.1f}%)")
print(f"Other share       : {other_count:,}  ({other_count/len(listed)*100:.1f}%)")
 
date_columns = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]

for col in date_columns:
    if col in listed.columns:
        listed[col] = pd.to_datetime(listed[col], errors="coerce")

repeats = [x for x in listed.columns if x.endswith(".1")]

names = ["ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",  "CoListAgentFirstName", "CoListAgentLastName",
    "BuyerAgentFirstName", "BuyerAgentLastName", "BuyerAgentMlsId", "CoBuyerAgentFirstName", "ListAgentAOR", "BuyerAgentAOR",
    "ListOfficeName", "BuyerOfficeName", "CoListOfficeName", "BuyerOfficeAOR", "BuilderName",]

drops = [x for x in repeats + names if x in listed.columns]

listed.drop(columns=drops, inplace=True)

print(f"\nDropped {len(drops)} columns")
print(f"Columns remaining: {listed.shape[1]}")

# Check Data Types
numeric_cols = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres", "LotSizeArea", 
            "LotSizeSquareFeet", "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt", "TaxAnnualAmount", 
            "AssociationFee", "FireplacesTotal", "ParkingTotal", "GarageSpaces", "CoveredSpaces", "Stories", "AboveGradeFinishedArea", 
            "BelowGradeFinishedArea", "BuildingAreaTotal", "MainLevelBedrooms", "Latitude", "Longitude"]

for col in numeric_cols:
    if listed[col].dtype in ["float64", "int64"]:
        print(f"  {col:<35} already numeric ({listed[col].dtype})")
        continue
    before_nulls = listed[col].isnull().sum()
    listed[col] = pd.to_numeric(listed[col], errors="coerce")
    new_nulls = listed[col].isnull().sum() - before_nulls
    print(f"  {col:<35} converted to {listed[col].dtype}  |  {new_nulls} new nulls")
# Filter to Residential 
pre    = len(listed)
listed = listed[listed["PropertyType"] == "Residential"].copy()
post   = len(listed)
listed.reset_index(drop=True, inplace=True)
print(f"\nFilter : listed['PropertyType'] == 'Residential'")
print(f"Before : {pre:,} rows")
print(f"After  : {post:,} rows")
print(f"Removed: {pre - post:,} rows")
 
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
    "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres", "BedroomsTotal",
    "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
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
 
# Intern EDA Questions 

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
    listed["ListingContractDate"] = pd.to_datetime(listed["ListingContractDate"], errors="coerce")
    listed["ExpirationDate"]      = pd.to_datetime(listed["ExpirationDate"],      errors="coerce")
    bad = listed[listed["ExpirationDate"] < listed["ListingContractDate"]]
    print(f"\nDate Consistency:")
    print(f"  ExpirationDate before ListingContractDate: {len(bad):,} record(s)")
    if len(bad) > 0:
        print("  [FLAG] Date inconsistencies detected.")
 
if "ListPrice" in listed.columns and "CountyOrParish" in listed.columns:
    top_counties = (listed.groupby("CountyOrParish")["ListPrice"]
                          .median().sort_values(ascending=False).head(10))
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
        series,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", color="navy"),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1].set_title("Boxplot")
    axes[1].set_ylabel(field)

    plt.tight_layout()
    plt.show()
# Save Filtered Dataset
 
listed.to_csv("listed_residential_filtered.csv", index=False)
print(f"\nFiltered dataset saved -> listed_residential_filtered.csv")
print(f"  Rows: {len(listed):,}  |  Columns: {listed.shape[1]}")
 
# Mortgage Rate Enrichment 
 
url      = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]
mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["rate_30yr_fixed"], errors="coerce")
 
print(f"\nFRED data fetched: {len(mortgage):,} weekly observations")
print(f"Date range: {mortgage['date'].min().date()} to {mortgage['date'].max().date()}")
 
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)
print(f"Resampled to {len(mortgage_monthly):,} monthly averages")
 
if "ListingContractDate" in listed.columns:
    listed["year_month"]  = pd.to_datetime(listed["ListingContractDate"], errors="coerce").dt.to_period("M")
    listed_with_rates     = listed.merge(mortgage_monthly, on="year_month", how="left")
 
    unmatched = listed_with_rates["rate_30yr_fixed"].isnull().sum()
    print(f"\nMerge validation:")
    print(f"  Rows in listed        : {len(listed):,}")
    print(f"  Rows after merge      : {len(listed_with_rates):,}")
    print(f"  Unmatched (null rate) : {unmatched:,}")
 
    preview_cols = [c for c in ["ListingContractDate", "year_month", "ListPrice", "rate_30yr_fixed"]
                    if c in listed_with_rates.columns]
    print("\nPreview:")
    print(listed_with_rates[preview_cols].head().to_string())
 
    listed_with_rates.to_csv("listed_enriched.csv", index=False)
    print(f"\nEnriched dataset saved -> listed_enriched.csv")
    print(f"  Rows: {len(listed_with_rates):,}  |  Columns: {listed_with_rates.shape[1]}")
else:
    print("\n[WARNING] 'ListingContractDate' not found — skipping mortgage merge.")
 
print("\nListed EDA complete.")