import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
 
# ── Load Data ─────────────────────────────────────────────────────────────────
 
sold = pd.read_csv("sold_residential.csv")
 
# Dataset Sturcture

print(f"\nRows    : {sold.shape[0]:,}")
print(f"Columns : {sold.shape[1]}")
print("\n--- Column Data Types ---")
print(sold.dtypes.to_string())
print("\n--- First 5 Rows ---")
print(sold.head().to_string())
 
# Unique Property Types 

property_types = sold["PropertyType"].unique()
print(f"\nUnique values found ({len(property_types)} total):")
for pt in sorted(property_types, key=lambda x: str(x)):
    count = (sold["PropertyType"] == pt).sum()
    pct   = count / len(sold) * 100
    print(f"  {str(pt):<30} {count:>6,} records  ({pct:.1f}%)")
 
residential_count = (sold["PropertyType"] == "Residential").sum()
other_count       = len(sold) - residential_count
print(f"\nResidential share : {residential_count:,}  ({residential_count/len(sold)*100:.1f}%)")
print(f"Other share       : {other_count:,}  ({other_count/len(sold)*100:.1f}%)")
 
# Filter to Residential Only 
 
pre_filter  = len(sold)
sold        = sold[sold["PropertyType"] == "Residential"].copy()
post_filter = len(sold)
 
print(f"\nFilter applied : sold['PropertyType'] == 'Residential'")
print(f"Rows before    : {pre_filter:,}")
print(f"Rows after     : {post_filter:,}")
print(f"Rows removed   : {pre_filter - post_filter:,}")
 
sold.reset_index(drop=True, inplace=True)
# Missing Value Analysis 
null_counts  = sold.isnull().sum()
null_pct     = (null_counts / len(sold) * 100).round(2)
null_summary = pd.DataFrame({
    "missing_count" : null_counts,
    "missing_pct"   : null_pct,
    "present_count" : len(sold) - null_counts,
}).sort_values("missing_pct", ascending=False)
print(f"\n{'Column':<40} {'Missing':>8} {'Missing %':>10} {'Present':>8}")
print("-" * 70)
for col, row in null_summary.iterrows():
    flag = "  *** >90% ***" if row["missing_pct"] > 90 else ""
    print(f"{col:<40} {int(row['missing_count']):>8,} "
          f"{row['missing_pct']:>9.1f}% {int(row['present_count']):>8,}{flag}")
# Flag Columns Above 90% Missing
high_missing = null_summary[null_summary["missing_pct"] > 90]
if high_missing.empty:
    print("\nNo columns exceed 90% missing values.")
else:
    print(f"\n{len(high_missing)} column(s) flagged:\n")
    print(f"{'Column':<40} {'Missing %':>10}")
    print("-" * 52)
    for col, row in high_missing.iterrows():
        print(f"{col:<40} {row['missing_pct']:>9.1f}%")
    print("\nRecommendation: Consider dropping unless core to analysis.")
 
numeric_fields = [
    "ClosePrice", "ListPrice", "OriginalListPrice",
    "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt"
]
percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
 
for field in numeric_fields:
    if field not in sold.columns:
        print(f"\n[WARNING] '{field}' not found — skipping.")
        continue
    series = sold[field].dropna()
    print(f"\n--- {field} ---")
    print(f"  Count   : {len(series):,}")
    print(f"  Missing : {sold[field].isnull().sum():,}")
    print(f"  Min     : {series.min():,.2f}")
    print(f"  Max     : {series.max():,.2f}")
    print(f"  Mean    : {series.mean():,.2f}")
    print(f"  Median  : {series.median():,.2f}")
    print(f"  Std Dev : {series.std():,.2f}")
    print(f"  Percentiles:")
    for p in percentiles:
        print(f"    p{int(p*100):>2}  : {series.quantile(p):,.2f}")
 
# Intern EDA Questions
 
if "ClosePrice" in sold.columns:
    print(f"\nMedian Close Price : ${sold['ClosePrice'].median():,.2f}")
    print(f"Average Close Price: ${sold['ClosePrice'].mean():,.2f}")
 
if "DaysOnMarket" in sold.columns:
    dom = sold["DaysOnMarket"].dropna()
    print(f"\nDays on Market:")
    print(f"  Median              : {dom.median():.0f} days")
    print(f"  Mean                : {dom.mean():.1f} days")
    print(f"  Max                 : {dom.max():.0f} days")
    print(f"  Sold within 30 days : {(dom <= 30).sum() / len(dom) * 100:.1f}%")
    print(f"  Sold within 90 days : {(dom <= 90).sum() / len(dom) * 100:.1f}%")
 
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
date_cols = ["CloseDate", "ListingContractDate"]
if all(c in sold.columns for c in date_cols):
    sold["CloseDate"]           = pd.to_datetime(sold["CloseDate"],           errors="coerce")
    sold["ListingContractDate"] = pd.to_datetime(sold["ListingContractDate"], errors="coerce")
    bad_dates = sold[sold["CloseDate"] < sold["ListingContractDate"]]
    print(f"\nDate Consistency:")
    print(f"  CloseDate before ListingContractDate: {len(bad_dates):,} records")
    if len(bad_dates) > 0:
        print("  [FLAG] Date inconsistencies detected.")
 
if "ClosePrice" in sold.columns and "CountyOrParish" in sold.columns:
    county_medians = (sold.groupby("CountyOrParish")["ClosePrice"]
                         .median()
                         .sort_values(ascending=False)
                         .head(10))
    print(f"\nTop 10 Counties by Median Close Price:")
    for county, med in county_medians.items():
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
 
sold.to_csv("sold_residential_filtered.csv", index=False)
print(f"\nFiltered dataset saved -> sold_residential_filtered.csv")
print(f"  Rows: {len(sold):,}  |  Columns: {sold.shape[1]}")
 
# Fetch from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=["observation_date"])
mortgage.columns = ["date", "rate_30yr_fixed"]
mortgage = mortgage.dropna(subset=["rate_30yr_fixed"])
print(f"\nFetched {len(mortgage):,} weekly mortgage rate observations from FRED.")
print(f"Date range: {mortgage['date'].min().date()} to {mortgage['date'].max().date()}")
 
# Resample weekly to monthly average
mortgage["year_month"] = mortgage["date"].dt.to_period("M")
mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)
print(f"Resampled to {len(mortgage_monthly):,} monthly averages.")
 
# Create year_month key on sold
if "CloseDate" in sold.columns:
    sold["year_month"] = pd.to_datetime(sold["CloseDate"], errors="coerce").dt.to_period("M")
else:
    print("[WARNING] CloseDate not found — cannot create year_month key.")
 
# Merge
sold_with_rates = sold.merge(mortgage_monthly, on="year_month", how="left")
 
# Validate
unmatched = sold_with_rates["rate_30yr_fixed"].isnull().sum()
print(f"\nMerge validation:")
print(f"  Rows with matched rate  : {len(sold_with_rates) - unmatched:,}")
print(f"  Rows missing rate       : {unmatched:,}")
 
if "CloseDate" in sold_with_rates.columns:
    print("\nPreview (CloseDate, year_month, ClosePrice, rate_30yr_fixed):")
    preview_cols = [c for c in ["CloseDate", "year_month", "ClosePrice", "rate_30yr_fixed"]
                    if c in sold_with_rates.columns]
    print(sold_with_rates[preview_cols].head().to_string(index=False))
 
# Save enriched dataset
sold_with_rates.to_csv("sold_with_mortgage_rates.csv", index=False)
print(f"\nEnriched dataset saved -> sold_with_mortgage_rates.csv")
print(f"  Rows: {len(sold_with_rates):,}  |  Columns: {sold_with_rates.shape[1]}")
 
print("\nSold EDA complete.")