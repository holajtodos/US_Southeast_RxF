# GA 2018 burn permit data is stopped on Nov 19th, 2018
# using GeorgiaTech data to replace the missing data
import os
import sys
import glob
import pandas as pd
import numpy as np
import rasterio
import pyproj
import math
from shapely import geometry
from shapely.geometry import Point
from rasterstats import zonal_stats

# Change directory
os.getcwd()
print('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020'

# Append the location of our function directory
dir_python_scripts = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis/step4_RxFireEmissionCode'
sys.path.append(os.path.join(dir_python_scripts, 'RxFireEmission'))

from util import GeoHelper
from BurnTypeDifferentiation.LandTypeHelper import max_counts
from BurnTypeDifferentiation.LandTypeHelper import counts

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print('cwd is %s ' % (os.getcwd()))

###############################################################################
# Helper: string-or-list to regex OR pattern (for agriculture keywords)
def _to_regex(term_or_list):
    if isinstance(term_or_list, (list, tuple, np.ndarray, pd.Series)):
        parts = [str(x) for x in term_or_list if pd.notna(x) and str(x).strip() != ""]
        return "|".join(parts) if parts else ""
    return str(term_or_list)
###############################################################################
SHOW_PLOTS = False

new_df = pd.read_csv("/work/chflab/jthuang/breadcrumbs/RxEmissionData/BlueSky_Input/SE_Combined_Permit_rx.csv")

df = new_df.loc[new_df['State'] == 'GA', :].copy()

# make sure DATE is datetime
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

df_sub = df.query("Time >= '2018-11-20' and Time <= '2018-12-31'")

# 1. Rename columns
df_sub = df_sub.rename(columns={
    'State': 'STATE',
    'Lat': 'LATITUDE',
    'Lon': 'LONGITUDE',
    'Time': 'DATE',
    'Burned_Area': 'ACRES'
})

# 2. Ensure DATE is a datetime and extract YEAR
df_sub['YEAR'] = df_sub['DATE'].dt.year

# 3. Drop the Id column (safe if missing)
df_sub = df_sub.drop(columns=['Id'], errors='ignore')

# 4. Reset the index (so row numbers go 0,1,2,…) and create OBJECTID
df_sub = df_sub.reset_index(drop=True)
df_sub['OBJECTID'] = df_sub.index + 42180

# Now df has columns: STATE, DATE, LATITUDE, LONGITUDE, ACRES, YEAR, OBJECTID
df_sub['BURN_TYPE'] = "Other"

# lf .tif data (FBFM40)
lf_files = {
    2014: "/work/chflab/jthuang/breadcrumbs/LandFire/US_140_FBFM40/Tif/us_140fbfm40.tif",
    2016: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    2020: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2022_FBFM40_220_CONUS/Tif/LC22_F40_220.tif"
}

year_range = {
    2014: [2010, 2014],  # LF 2014 Update--Includes disturbances for the years 1999-2014
    2016: [2015, 2016],  # LF 2016 Remap--Includes disturbances for the years 2015-2016--2.0.0 (200) = new base map
    2020: [2017, 2020]   # LF 2020--Includes disturbances for the years 2017-2020--2.2.0 (220) = 2nd update to LF 2016 Remap
}

# Create transformers once
lf_file = lf_files[2020]

with rasterio.open(lf_file) as r:
    transformer_r = pyproj.Transformer.from_crs('epsg:4326', r.crs, always_xy=True)

    permit_id = df_sub['OBJECTID'].to_numpy()
    permit_lat = df_sub['LATITUDE'].to_numpy()
    permit_lon = df_sub['LONGITUDE'].to_numpy()
    permit_time = pd.to_datetime(df_sub['DATE'], errors='coerce')
    burn_area = df_sub["ACRES"].to_numpy()

    num_fires = len(df_sub)
    print("#Total:", num_fires)

    # What if burned area is nan?
    burn_area = np.nan_to_num(burn_area)

    # Process coordinates and data
    coords = np.vstack((permit_lon, permit_lat)).T
    coords_r = np.array([transformer_r.transform(lon, lat) for lon, lat in coords])

    lf_attr_res = np.zeros(len(df_sub))

    for i, coord_r in enumerate(coords_r):
        permit_x, permit_y = coord_r
        # change the buffer size based on burned area
        # convert acres to m2
        this_burn_area = burn_area[i] * 4046.8564224
        # calculate the buffer size
        buffer_length = max(math.sqrt(this_burn_area) / 2, 15)

        # Create points for buffer calculation
        buffer_point_ctr = Point(permit_x, permit_y)
        if buffer_length <= 15:
            print(f"Too small: {permit_id[i]}, set the buffer length as 15m")
            buffer_length = 15
        buffer_grid = buffer_point_ctr.buffer(buffer_length, cap_style=3)

        # Get the pixel values at the given coordinates
        # row, col are the row and column indices in the raster matrix
        row, col = r.index(permit_x, permit_y)
        point_type = r.read(1, window=((row, row + 1), (col, col + 1))).item()

        # Handling zonal stats and raster reads based on area
        if this_burn_area <= 900:
            lf_attr_res[i] = point_type
        else:
            stats = zonal_stats(buffer_grid, lf_file, add_stats={'counts': counts})
            main_type = max_counts(stats[0]["counts"])
            # Conditions based on types and validity checks
            lf_attr_res[i] = main_type

    # Save results
    df_sub['fuel_type'] = lf_attr_res

df_sub['DATE'] = pd.to_datetime(df_sub['DATE'], utc=True, errors='coerce')

# Read burn permits
state_abbr = 'GA'
num_fires = len(df_sub)
print("#Total:", num_fires)

# Agricultural, Invalid and Rx
# LF
legend = np.array([
    -9999, 91, 92, 93, 98, 99,
    101, 102, 103, 104, 105, 106,
    107, 108, 109, 121, 122, 123,
    124, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 161, 162,
    163, 164, 165, 181, 182, 183,
    184, 185, 186, 187, 188, 189,
    201, 202, 203, 204
])
leg_str = np.array(
    ['No Data', 'Urban/Developed', 'Snow/Ice', 'Agricultural', 'Open Water', 'Barren',
     'GR1', 'GR2', 'GR3', 'GR4', 'GR5', 'GR6', 'GR7', 'GR8', 'GR9', 'GS1', 'GS2', 'GS3',
     'GS4', 'SH1', 'SH2', 'SH3', 'SH4', 'SH5', 'SH6', 'SH7', 'SH8', 'SH9', 'TU1', 'TU2',
     'TU3', 'TU4', 'TU5', 'TL1', 'TL2', 'TL3', 'TL4', 'TL5', 'TL6', 'TL7', 'TL8', 'TL9',
     'SB1', 'SB2', 'SB3', 'SB4']
)
# LF map
lf_map = {}
for i in range(0, len(leg_str)):
    lf_map[leg_str[i]] = legend[i]

invalid_type = ["No Data", "Snow/Ice", "Open Water", "Barren"]
agricultural_type = ["Agricultural"]

invalid_type_num = []
agricultural_type_num = lf_map[agricultural_type[0]]
for type_tmp in invalid_type:
    invalid_type_num.append(lf_map[type_tmp])

agri_index = []
invalid_index = []
valid_index = []

# Function to check conditions and populate indices lists
def check_conditions(state_abbr_local, contains_agriculture_keyword=None, require_keywords_for_agri=None):
    # Build keyword matcher #1: legacy/other-state "contains agriculture" rule
    pattern = _to_regex(contains_agriculture_keyword) if contains_agriculture_keyword else None
    if pattern:
        contains_agriculture = df_sub['BURN_TYPE'].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
    else:
        contains_agriculture = pd.Series(False, index=permit_df.index)

    # Build keyword matcher #2: GA/SC/FL restriction — only classify agri when fuel_type is agri AND BURN_TYPE hits these
    if require_keywords_for_agri:
        pattern_req = _to_regex(require_keywords_for_agri)
        require_hit = df_sub['BURN_TYPE'].fillna("").astype(str).str.contains(pattern_req, case=False, regex=True)
    else:
        require_hit = pd.Series(True, index=permit_df.index)  # no restriction if not provided

    for index, row in df_sub.iterrows():
        fuel_type = row["fuel_type"]

        # --- New logic: if a "require_keywords_for_agri" list is provided (GA/SC/FL),
        #     fuel_type must be agricultural_type_num AND BURN_TYPE must match those keywords
        if require_keywords_for_agri is not None:
            if contains_agriculture_keyword is not None:
                if contains_agriculture.iloc[index]:
                    agri_index.append(index)
                elif (fuel_type == agricultural_type_num) and require_hit.iloc[index]:
                    agri_index.append(index)
                elif fuel_type in invalid_type_num:
                    invalid_index.append(index)
                else:
                    valid_index.append(index)
            else:
                if (fuel_type == agricultural_type_num) and require_hit.iloc[index]:
                    agri_index.append(index)
                elif fuel_type in invalid_type_num:
                    invalid_index.append(index)
                else:
                    valid_index.append(index)
        else:
            # Original behavior for other states:
            if contains_agriculture.iloc[index]:
                agri_index.append(index)
            elif fuel_type in invalid_type_num:
                invalid_index.append(index)
            else:
                valid_index.append(index)

# Apply conditions based on state abbreviation
if state_abbr == 'FL':
    # Keep your original FL rule (substring 'Agricultur')
    # FL: ONLY when BURN_TYPE contains "Other" AND fuel_type == agricultural_type_num
    check_conditions('FL', contains_agriculture_keyword='Agricultur', require_keywords_for_agri=['Other'])

elif state_abbr == 'GA':
    # Keep your original GA rule (substring 'CROP', 'ORCHARD')
    # GA: ONLY when BURN_TYPE contains "PASTURE" or "Other" AND fuel_type == agricultural_type_num
    check_conditions('GA', contains_agriculture_keyword=['CROP', 'ORCHARD'], require_keywords_for_agri=['PASTURE', 'Other'])

elif state_abbr == 'SC':
    # Keep your original SC rule (substring 'PASTURE', 'DITCH')
    # SC: ONLY when BURN_TYPE contains "DISEASE" or "Other" AND fuel_type == agricultural_type_num
    check_conditions('SC', contains_agriculture_keyword=['PASTURE', 'DITCH'], require_keywords_for_agri=['DISEASE', 'Other'])

elif state_abbr == 'MS':
    # Keep your original MS rule (substring 'Agricultur')
    check_conditions('MS', contains_agriculture_keyword='Agricultur', require_keywords_for_agri=None)

elif state_abbr == 'TN':
    # Keep your original TN rule (substring 'Agricultur')
    check_conditions('TN', contains_agriculture_keyword='Agricultur', require_keywords_for_agri=None)

elif state_abbr == 'NC':
    # Keep your original NC rule (substring 'Other')
    check_conditions('NC', contains_agriculture_keyword='Other', require_keywords_for_agri=None)

else:
    # Fallback: original else branch
    for index, row in permit_df.iterrows():
        fuel_type = row["fuel_type"]
        if fuel_type == agricultural_type_num:
            agri_index.append(index)
        elif fuel_type in invalid_type_num:
            invalid_index.append(index)
        else:
            valid_index.append(index)

agri_df = df_sub.iloc[agri_index]
invalid_df = df_sub.iloc[invalid_index]
valid_df = df_sub.iloc[valid_index]

agri_df = agri_df.reset_index(drop=True)
invalid_df = invalid_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

old_rx_df = pd.read_csv("GA_2018_permits_lf_rx.csv")
old_agr_df = pd.read_csv("GA_2018_permits_lf_agr.csv")
old_invalid_df = pd.read_csv("GA_2018_permits_lf_invalid.csv")

valid_df = valid_df.reindex(columns=old_rx_df.columns)
combined_rx_df = pd.concat([old_rx_df, valid_df], ignore_index=True)
combined_rx_df.to_csv("GA_2018_permits_lf_rx_new.csv", index=False, header=True)

agri_df = agri_df.reindex(columns=old_agr_df.columns)
combined_agr_df = pd.concat([old_agr_df, agri_df], ignore_index=True)
combined_agr_df.to_csv("GA_2018_permits_lf_agr_new.csv", index=False, header=True)

invalid_df = invalid_df.reindex(columns=old_invalid_df.columns)
combined_invalid_df = pd.concat([old_invalid_df, invalid_df], ignore_index=True)
combined_invalid_df.to_csv("GA_2018_permits_lf_invalid_new.csv", index=False, header=True)

# Combine all rx records
filenames = []
selected_year = 2018

# List of states to process
states = ["Florida", "Georgia", "South Carolina"]
state_abbreviations = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for state_name in states:
    state_abbr = state_abbreviations[state_name]
    if state_abbr == 'GA':
        filenames.append(f"{state_abbr}_{selected_year}_permits_lf_rx_new.csv")
    filenames.append(f"{state_abbr}_{selected_year}_permits_lf_rx.csv")

idx = 0
res_df = None
for filename in filenames:
    if idx == 0:
        res_df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename)
        res_df = pd.concat([res_df, df], ignore_index=True)
    idx = idx + 1

res_df.to_csv("SE_Combined_Permit_lf_3states_rx_" + str(selected_year) + ".csv", index=False, header=True)

# Combine all agricultural records
filenames = []
selected_year = 2018

# List of states to process
states = ["Florida", "Georgia", "South Carolina"]
state_abbreviations = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for state_name in states:
    state_abbr = state_abbreviations[state_name]
    if state_abbr == 'GA':
        filenames.append(f"{state_abbr}_{selected_year}_permits_lf_agr_new.csv")
    filenames.append(f"{state_abbr}_{selected_year}_permits_lf_agr.csv")

idx = 0
res_df = None
for filename in filenames:
    if idx == 0:
        res_df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename)
        res_df = pd.concat([res_df, df], ignore_index=True)
    idx = idx + 1

res_df.to_csv("SE_Combined_Permit_lf_3states_agr_" + str(selected_year) + ".csv", index=False, header=True)

# Combine all invalid records
filenames = []
selected_year = 2018

# List of states to process
states = ["Florida", "Georgia", "South Carolina"]
state_abbreviations = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for state_name in states:
    state_abbr = state_abbreviations[state_name]
    if state_abbr == 'GA':
        filenames.append(f"{state_abbr}_{selected_year}_permits_lf_invalid_new.csv")
    filenames.append(f"{state_abbr}_{selected_year}_permits_lf_invalid.csv")

idx = 0
res_df = None
for filename in filenames:
    if idx == 0:
        res_df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename)
        res_df = pd.concat([res_df, df], ignore_index=True)
    idx = idx + 1

res_df.to_csv("SE_Combined_Permit_lf_3states_invalid_" + str(selected_year) + ".csv", index=False, header=True)
