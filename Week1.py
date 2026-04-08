import pandas as pd

# Define the range of months from January 2024 to February 2026 (most recently completed month as of April 2026)
months = []
for year in range(2024, 2027):
    for month in range(1, 13):
        if not (year == 2026 and month > 2):
            months.append(f"{year}{month:02d}")
# Load and concatenate listings
listing_dfs = []
for month in months:
    file = f"CRMLSListing{month}.csv"
    df = pd.read_csv(file, encoding="cp1252")
    listing_dfs.append(df)
    print(f"Loaded {file} with {len(df)} rows") # Counting rows before concatenation
listings_combined = pd.concat(listing_dfs, ignore_index=True)
print(f"After concatenation, the listings dataframe has {len(listings_combined)} rows") # Counting rows after concatenation
# Filter to Residential
listings_res = listings_combined[listings_combined['PropertyType'] == 'Residential']
print(f"After filtering to Residential, listings have {len(listings_residential)} rows") # Counting rows after filtering
# Save to CSV
listings_res.to_csv("listings_residential.csv", index=False)
# Load and concatenate sold
sold_dfs = []
for month in months:
    file = f"CRMLSSold{month}.csv"
    df = pd.read_csv(file, encoding="cp1252")
    sold_dfs.append(df)
    print(f"Loaded {file} with {len(df)} rows") # Counting rows before concatenation
sold_combined = pd.concat(sold_dfs, ignore_index=True)
print(f"After concatenation, the sold dataframe has {len(sold_combined)} rows")
# Filter to Residential only
sold_res = sold_combined[sold_combined['PropertyType'] == 'Residential'] # Counting rows after filtering
print(f"After filtering to Residential, the sold dataframe {len(sold_residential)} rows")
# Save to CSV
sold_res .to_csv("sold_residential.csv", index=False)