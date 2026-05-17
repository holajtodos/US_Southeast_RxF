"""Extract SE (FL, GA, SC) FIREID data from 2017-2019 and create yearly CSV files."""

import pandas as pd
from pathlib import Path

# Define paths
rx_folder = Path("/work/chflab/jthuang/breadcrumbs/RxEmissionData/FINN/Rx_WF")
output_folder = Path("/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/FINN_rxf_inv")

# States to process
states = ['FL', 'GA', 'SC']

# Years to process
years = [2017, 2018, 2019]

# Read all state files
all_data = []

for state in states:
    file_path = rx_folder / f"{state}_Combined_MODIS_VIIRS_rx_wf.csv"
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    all_data.append(df)
    print(f"  Loaded {len(df)} rows")

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal rows combined: {len(combined_df)}")

# Convert DAY to datetime
combined_df['DAY'] = pd.to_datetime(combined_df['DAY'])

# Extract year and month
combined_df['YEAR'] = combined_df['DAY'].dt.year
combined_df['MONTH'] = combined_df['DAY'].dt.month

# Filter for FIREIDs that start with FL_, GA_, or SC_
filtered_df = combined_df[
    (combined_df['FIREID'].str.startswith('FL_', na=False)) |
    (combined_df['FIREID'].str.startswith('GA_', na=False)) |
    (combined_df['FIREID'].str.startswith('SC_', na=False))
].copy()

print(f"\nRows with FL_, GA_, or SC_ FIREID: {len(filtered_df)}")

# Filter for years 2017-2019
filtered_df = filtered_df[filtered_df['YEAR'].isin(years)]

# Filter for Jan-Apr (months 1-4)
filtered_df = filtered_df[filtered_df['MONTH'].isin([1, 2, 3, 4])]

print(f"Rows after year and month filtering (2017-2019, Jan-Apr): {len(filtered_df)}")

# Add STATE column based on FIREID prefix
filtered_df['STATE'] = filtered_df['FIREID'].str[:2]

# Create separate files for each year
for year in years:
    year_data = filtered_df[filtered_df['YEAR'] == year].copy()
    
    # Remove the YEAR and MONTH columns before saving, but keep STATE
    year_data = year_data.drop(columns=['YEAR', 'MONTH'])
    
    output_file = output_folder / f"SE_Combined_FINN_rx_wf_{year}_Jan-Apr.csv"
    year_data.to_csv(output_file, index=False)
    print(f"\nSaved {year}: {output_file}")
    print(f"  Rows: {len(year_data)}")

print("\nDone!")