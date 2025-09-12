# -*- coding: utf-8 -*-
###############################################################################
# clean_SE_burns.py
# author: Jingting HUANG
# purpose: To clean/extract burn permit data during 2010-2020
#          To separate RxFire and AgFire in burn permit data
# version history:
#   02/17/2025 - original
# data required:
#   SE(seven states: FL, GA, AL, SC, NC, MS, TN) burn permits
#   LANDFIRE Fire Behavior Fuel Model 13
#   National Land Cover Database (NLCD)
# usage:
#   -
# to do:
#   -
# notes:
#   -
# debugging:
#   -
###############################################################################

import os
import sys
import glob
import pandas as pd
import numpy as np
import math
from datetime import datetime
import pyproj
from shapely import geometry
from shapely.geometry import Point
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
###############################################################################
# Helper: string-or-list to regex OR pattern (for agriculture keywords)
def _to_regex(term_or_list):
    if isinstance(term_or_list, (list, tuple, np.ndarray, pd.Series)):
        parts = [str(x) for x in term_or_list if pd.notna(x) and str(x).strip() != ""]
        return "|".join(parts) if parts else ""
    return str(term_or_list)
###############################################################################
# Helper: combine files across states for a given year & suffix
def combine_by_state_year(states_list, state_abbrev_map, year, suffix, out_prefix):
    filenames = []
    for state_name in states_list:
        abbr = state_abbrev_map[state_name]
        filenames.append(f"{abbr}_{year}_permits_lf_{suffix}.csv")

    idx = 0
    res_df = None
    for filename in filenames:
        if not os.path.exists(filename):
            continue
        if idx == 0:
            res_df = pd.read_csv(filename)
        else:
            df = pd.read_csv(filename)
            res_df = pd.concat([res_df, df], ignore_index=True)
        idx += 1

    if res_df is not None:
        out_name = f"{out_prefix}_{suffix}_{year}.csv"
        res_df.to_csv(out_name, index=False, header=True)
        print(f"Saved {out_name}")
    else:
        print(f"No files found for {year} with suffix '{suffix}'.")
###############################################################################
SHOW_PLOTS = False

# List of states to process
states = ["Florida", "Alabama", "Mississippi", "Georgia", "Tennessee", "South Carolina", "North Carolina"]
state_abbreviations = {"Florida": "FL", "Alabama": "AL", "Mississippi": "MS", "Georgia": "GA", "Tennessee": "TN", "South Carolina": "SC", "North Carolina": "NC"}

permit_file = os.path.join("SE_BurnData2010_2020-AllStates.csv")

# Filter rows where the year is during 2010 - 2020
for selected_year in range(2010, 2021):
    print('Processing for the year of ' + str(selected_year))
    valid_df = pd.read_csv(permit_file)
    valid_df['DATE'] = pd.to_datetime(valid_df['DATE'])
    valid_df = valid_df.sort_values(by='DATE')
    valid_df = valid_df[valid_df['DATE'].dt.year == selected_year]

    for state_name in states:
        state = GeoHelper.StatePolygon(state_name)
        state_abbr = state_abbreviations[state_name]

        state_df = valid_df[valid_df["STATE"] == state_abbr]

        invalid_idx = []
        # remove the fire out of the state boundary
        for index, row in state_df.iterrows():
            fire_point = geometry.Point(row["LONGITUDE"], row["LATITUDE"])
            if not state.contains(fire_point):
                invalid_idx.append(index)

        print(f"{len(invalid_idx)} Fires are out of state for {state_name}")
        state_df = state_df.drop(invalid_idx)

        # --- remove burn area: zero or missing (robust numeric) ---
        state_df["ACRES"] = pd.to_numeric(state_df["ACRES"], errors="coerce")
        invalid_burned_area = state_df[state_df["ACRES"] <= 0]
        print(f"{len(invalid_burned_area)} Fires are less than or equal to 0 acres for {state_name}")
        state_df = state_df[state_df["ACRES"] > 0]

        num_fires = len(state_df)
        print("#Total > 0 acres:", num_fires)

        state_df = state_df[state_df["LATITUDE"] > 0]

        num_fires = len(state_df)
        print(state_abbr)
        print("#Total:", num_fires)

        # Check if there is data in the BURN_TYPE column and fill missing data with "Other"
        state_df = state_df[state_df["BURN_TYPE"] != "PILED DEBRIS"]

        state_df['BURN_TYPE'] = state_df['BURN_TYPE'].fillna('Other').apply(lambda x: 'Other' if str(x).strip() == '' else str(x).strip())
        state_df['BURN_TYPE'] = state_df['BURN_TYPE'].replace(['None', 'none', ''], 'Other').apply(lambda x: 'Other' if str(x).strip() == '' else str(x).strip())

        merged_df = state_df.groupby(['LATITUDE', 'LONGITUDE', 'BURN_TYPE', 'STATE', 'YEAR', 'DATE'], as_index=False)['ACRES'].sum()

        num_fires = len(merged_df)
        print("#Total after merged:", num_fires)

        # Add a column 'OBJECTID' that represents the index of each row
        merged_df = merged_df.sort_values(by='DATE', ascending=True)
        merged_df.reset_index(drop=True, inplace=True)  # Reset index to 0...n-1
        merged_df['OBJECTID'] = merged_df.index + 1     # Assign a 1-based sequence

        if SHOW_PLOTS:
            # Optional quick-look plot (disable on HPC if slow)
            if state.geom_type == "MultiPolygon":
                for geom in state.geoms:
                    xs, ys = geom.exterior.xy
                    plt.plot(xs, ys, 'k')
            else:
                xs, ys = state.exterior.xy
                plt.plot(xs, ys, 'k')
            plt.scatter(merged_df["LONGITUDE"], merged_df["LATITUDE"], s=1)
            plt.show()

        if not merged_df.empty:
            # Save the cleaned data for the state
            output_file = f"{state_abbr}_{selected_year}_permits.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"Saved {output_file}")

# LANDFIRE inputs by era
lf_files = {
    2014: "/work/chflab/jthuang/breadcrumbs/LandFire/US_140_FBFM40/Tif/us_140fbfm40.tif",  # FBFM40
    2016: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    2020: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2022_FBFM40_220_CONUS/Tif/LC22_F40_220.tif"
}

year_range = {
    2014: [2010, 2014],  # LF 2014 Update--Includes disturbances for the years 1999-2014
    2016: [2015, 2016],  # LF 2016 Remap--Includes disturbances for the years 2015-2016--2.0.0 (200) = new base map
    2020: [2017, 2020]   # LF 2020--Includes disturbances for the years 2017-2020--2.2.0 (220) = 2nd update to LF 2016 Remap
}

# Create transformers once per LF era/raster
for year in lf_files.keys():
    lf_file = lf_files[year]
    with rasterio.open(lf_file) as r:
        transformer_r = pyproj.Transformer.from_crs('epsg:4326', r.crs, always_xy=True)

        lower_time = year_range[year][0]
        upper_time = year_range[year][1] + 1

        for selected_year in range(lower_time, upper_time):
            for state_name in states:
                state_abbr = state_abbreviations[state_name]
                permit_file = f"{state_abbr}_{selected_year}_permits.csv"
                output_file = f"{state_abbr}_{selected_year}_permits_lf.csv"

                # Read burn permits
                # Check if the file exists
                if os.path.exists(permit_file):
                    print("File exists.")
                    permit_df = pd.read_csv(permit_file)
                    permit_id = permit_df['OBJECTID'].to_numpy()
                    permit_lat = permit_df['LATITUDE'].to_numpy()
                    permit_lon = permit_df['LONGITUDE'].to_numpy()
                    permit_time = pd.to_datetime(permit_df['DATE'], errors='coerce')
                    burn_area = permit_df["ACRES"].to_numpy()

                    num_fires = len(permit_df)
                    print(state_abbr)
                    print("#Total:", num_fires)

                    # What if burned area is nan?
                    burn_area = np.nan_to_num(burn_area)

                    # Process coordinates and data
                    coords = np.vstack((permit_lon, permit_lat)).T
                    coords_r = np.array([transformer_r.transform(lon, lat) for lon, lat in coords])

                    lf_attr_res = np.zeros(len(permit_df))

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
                            print(f"In {state_abbr}, Too small: " + str(permit_id[i]) + ", set the buffer length as 15m")
                            buffer_length = 15
                        buffer_grid = buffer_point_ctr.buffer(buffer_length, cap_style=3)

                        # Get the pixel values at the given coordinates (scalar + bounds guard)
                        try:
                            row, col = r.index(permit_x, permit_y)
                            point_type = r.read(1, window=((row, row + 1), (col, col + 1))).item()
                        except Exception:
                            point_type = -9999  # sentinel

                        # Handling zonal stats and raster reads based on area
                        if this_burn_area <= 900:
                            lf_attr_res[i] = point_type
                        else:
                            # Use the buffered polygon for zonal stats (not the point)
                            stats = zonal_stats(buffer_grid, lf_file, add_stats={'counts': counts})
                            main_type = max_counts(stats[0]["counts"])
                            lf_attr_res[i] = main_type

                    # Save results
                    permit_df['fuel_type'] = lf_attr_res
                    permit_df.to_csv(output_file, index=False)

                else:
                    print("File not found.")

for selected_year in range(2010, 2021):
    print('Processing for the year of ' + str(selected_year))
    for state_name in states:
        state_abbr = state_abbreviations[state_name]
        permit_file = f"{state_abbr}_{selected_year}_permits_lf.csv"

        # Check if the file exists
        if os.path.exists(permit_file):
            print("File exists.")

            # Read burn permits
            permit_df = pd.read_csv(permit_file)
            num_fires = len(permit_df)
            print(state_abbr)
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
                    contains_agriculture = permit_df['BURN_TYPE'].fillna("").astype(str).str.contains(pattern, case=False, regex=True)
                else:
                    contains_agriculture = pd.Series(False, index=permit_df.index)
            
                # Build keyword matcher #2: GA/SC/FL restriction — only classify agri when fuel_type is agri AND BURN_TYPE hits these
                if require_keywords_for_agri:
                    pattern_req = _to_regex(require_keywords_for_agri)
                    require_hit = permit_df['BURN_TYPE'].fillna("").astype(str).str.contains(pattern_req, case=False, regex=True)
                else:
                    require_hit = pd.Series(True, index=permit_df.index)  # no restriction if not provided
            
                for index, row in permit_df.iterrows():
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


            agri_df = permit_df.iloc[agri_index]
            invalid_df = permit_df.iloc[invalid_index]
            valid_df = permit_df.iloc[valid_index]

            agri_df = agri_df.reset_index(drop=True)
            invalid_df = invalid_df.reset_index(drop=True)
            valid_df = valid_df.reset_index(drop=True)

            # save to csv files
            agri_df.to_csv(f"{state_abbr}_{selected_year}_permits_lf_agr.csv", index=False)
            invalid_df.to_csv(f"{state_abbr}_{selected_year}_permits_lf_invalid.csv", index=False)
            valid_df.to_csv(f"{state_abbr}_{selected_year}_permits_lf_rx.csv", index=False)

# -----------------------------------------------------------------------------
# Flexible combine examples
# -----------------------------------------------------------------------------

# All 7 states (rx, agr, invalid)
states7 = ["Florida", "Alabama", "Mississippi", "Georgia", "Tennessee", "South Carolina", "North Carolina"]
abbr7 = {"Florida": "FL", "Alabama": "AL", "Mississippi": "MS", "Georgia": "GA", "Tennessee": "TN", "South Carolina": "SC", "North Carolina": "NC"}

for selected_year in range(2013, 2021):
    combine_by_state_year(states7, abbr7, selected_year, "rx", "SE_Combined_Permit_lf")
    combine_by_state_year(states7, abbr7, selected_year, "agr", "SE_Combined_Permit_lf")
    combine_by_state_year(states7, abbr7, selected_year, "invalid", "SE_Combined_Permit_lf")

# 3-state subsets at specific years
states3 = ["Florida", "Georgia", "South Carolina"]
abbr3 = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for selected_year in range(2017, 2020):
    combine_by_state_year(states3, abbr3, selected_year, "rx", "SE_Combined_Permit_lf_3states")
    combine_by_state_year(states3, abbr3, selected_year, "agr", "SE_Combined_Permit_lf_3states")
    combine_by_state_year(states3, abbr3, selected_year, "invalid", "SE_Combined_Permit_lf_3states")