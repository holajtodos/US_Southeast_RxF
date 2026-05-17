# -*- coding: utf-8 -*-
###############################################################################
# extract_SE_NEI.py 
# author: Jingting HUANG
# purpose: To extract acres burned for FL, GA, and SC from NEI prescribed fire inventories during 2017-2019
# version history: 
#   06/20/2025 - original  
# data required: 
#   Rx:
#   ptday_ptfire_2017NEI_20200206_CONUS_CAPs_07apr2020_nf_v1_rxfire
#   ptday_ptfire_2018gc_31mar2021_nf_v2
#   ptday_ptfire_sf2_2019ge_bsp_16dec2021_nf_v1
#   ptinv_ptfire_2017NEI_20200206_CONUS_CAPs_07apr2020_nf_v1_rxfire.csv
#   ptinv_ptfire_2018gc_31mar2021_nf_v2.csv
#   ptinv_ptfire_sf2_2019ge_bsp_16dec2021_nf_v1.csv
#   Ag Fire:
#   ptday_agburn_2017_MYR_ff10_22apr2020_v0
#   ptday_agburn_2018_ff10_csv_08mar2021_v0
#   ptday_agfire_CONUS_2019ge_23dec2021_v0
#   ptday_agfire_FL_2019ge_23dec2021_v0
#   ptday_fl_agburn_2018_ff10_csv_08mar2021_v0
#   ptinv_agburn_2017_MYR_ff10_22apr2020_v0.csv
#   ptinv_agburn_2018_ff10_08mar2021_v0.csv
#   ptinv_agfire_CONUS_2019ge_23dec2021_v0.csv
#   ptinv_agfire_FL_2019ge_23dec2021_v0.csv
#   ptinv_fl_agburn_2018_ff10_08mar2021_v0.csv
# usage:  
#   -
# to do: 
#   - 
# notes: 
#   -
# debugging: 
#   -
###############################################################################
from io import StringIO
import sys
import os
import math
import numpy as np
import pandas as pd
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
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/NEI_rxf_inv'

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
SHOW_PLOTS = False

# --- Setup RX daily (ptday_ptfire) ---
fire_inv = [
    "ptday_ptfire_2017NEI_20200206_CONUS_CAPs_07apr2020_nf_v1_rxfire",
    "ptday_ptfire_2018gc_31mar2021_nf_v2",
    "ptday_ptfire_sf2_2019ge_bsp_16dec2021_nf_v1",
]

# List of day columns and metadata
day_cols = [f"dayval{d}" for d in range(1, 32)]
meta_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'poll', 'monthnum']

# Unique ID columns + DATE (used after melt)
grp_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'DATE']

# pollutants (daily has CO2)
pollutant_cols = ['ACRESBURNED', 'CO', 'CO2', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']

df_daily_list = []

for k, fname in enumerate(fire_inv):
    input_path = os.path.join(dir_python_local, fname)
    if not os.path.isfile(input_path):
        print(f"Missing file: {fname} (skipping)")
        continue
    Year = 2017 + k
    print(f"\nProcessing '{fname}'... (Year={Year})")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    lines = [line.replace('\r\n', '\n').replace('\r', '\n').replace('\x0d', '') for line in lines]

    # locate header row
    for i, line in enumerate(lines):
        if line.startswith('country_cd'):
            header_index = i
            break
    else:
        raise RuntimeError(f"Cannot find column headers in '{input_path}'")

    data_str = ''.join(lines[header_index:])
    df = pd.read_csv(StringIO(data_str))

    # Harmonize pollutant names before any processing
    df['poll'] = df['poll'].replace({
        'PM25-PRI': 'PM2_5',
        'PM10-PRI': 'PM10'
    })

    # Melt day columns long
    df_melted = df[meta_cols + day_cols].melt(
        id_vars=meta_cols, value_vars=day_cols, var_name='day_column', value_name='value'
    )
    df_melted = df_melted[df_melted['value'].notna() & (df_melted['value'] != 0)]
    df_melted['day'] = df_melted['day_column'].str.extract(r'dayval(\d+)').astype(int)

    # Build DATE
    df_melted['DATE'] = pd.to_datetime(
        {'year': Year, 'month': df_melted['monthnum'], 'day': df_melted['day']},
        errors='coerce'
    )
    df_melted = df_melted[df_melted['DATE'].notna()]

    # Pivot to pollutants as columns
    df_pivot = df_melted.pivot_table(
        index=['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'DATE'],
        columns='poll',
        values='value',
        aggfunc='sum'
    ).reset_index()
    df_pivot.columns.name = None

    # Ensure missing pollutant columns exist, then order/select
    for col in pollutant_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
    df_pivot = df_pivot[grp_cols + pollutant_cols]
    df_pivot[pollutant_cols] = df_pivot[pollutant_cols].fillna(0)

    # RX SCC filter
    rxfire_scc_set = {2811020002, 2811015001, 2811015002}
    df_merged = df_pivot[df_pivot['scc'].isin(rxfire_scc_set)].copy()
    df_daily_list.append(df_merged)

###############################################################################
# --- Setup RX annual (ptinv_ptfire) ---
fire_inv = [
    "ptinv_ptfire_2017NEI_20200206_CONUS_CAPs_07apr2020_nf_v1_rxfire.csv",
    "ptinv_ptfire_2018gc_31mar2021_nf_v2.csv",
    "ptinv_ptfire_sf2_2019ge_bsp_16dec2021_nf_v1.csv",
]

# Annual meta / id columns
meta_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'poll', 'longitude', 'latitude', 'ann_value']
grp_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'longitude', 'latitude']

# pollutants (annual list—note: no CO2 field in some inventories)
pollutant_cols = ['ACRESBURNED', 'CO', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']

df_annual_list = []

for k, fname in enumerate(fire_inv):
    input_path = os.path.join(dir_python_local, fname)
    if not os.path.isfile(input_path):
        print(f"Missing file: {fname} (skipping)")
        continue
    print(f"\nProcessing '{fname}'...")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    lines = [line.replace('\r\n', '\n').replace('\r', '\n').replace('\x0d', '') for line in lines]

    for i, line in enumerate(lines):
        if line.startswith('country_cd'):
            header_index = i
            break
    else:
        raise RuntimeError(f"Cannot find column headers in '{input_path}'")

    data_str = ''.join(lines[header_index:])
    df = pd.read_csv(StringIO(data_str))

    # Harmonize pollutant names before any processing
    df['poll'] = df['poll'].replace({
        'PM25-PRI': 'PM2_5',
        'PM10-PRI': 'PM10'
    })

    df_pivot = df.pivot_table(
        index=['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'longitude', 'latitude'],
        columns='poll',
        values='ann_value',
        aggfunc='sum'
    ).reset_index()
    df_pivot.columns.name = None

    # Ensure missing pollutant columns exist, then order/select
    for col in pollutant_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
    df_pivot = df_pivot[grp_cols + pollutant_cols]
    df_pivot[pollutant_cols] = df_pivot[pollutant_cols].fillna(0)

    rxfire_scc_set = {2811020002, 2811015001, 2811015002}
    df_merged = df_pivot[df_pivot['scc'].isin(rxfire_scc_set)].copy()
    df_annual_list.append(df_merged)

# --- Merge RX daily+annual by year (order is aligned by loops above) ---
merged_all_years = []
for df_day, df_ann in zip(df_daily_list, df_annual_list):
    merged_df = pd.merge(
        df_day, df_ann,
        on=['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc'] + pollutant_cols,
        how='inner',
        suffixes=('', '_ann')
    )
    merged_all_years.append(merged_df)

df_final = pd.concat(merged_all_years, ignore_index=True)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 7-state list for RX extraction
states = ["Florida", "Alabama", "Mississippi", "Georgia", "Tennessee", "South Carolina", "North Carolina"]
state_abbreviations = {"Florida": "FL", "Alabama": "AL", "Mississippi": "MS", "Georgia": "GA", "Tennessee": "TN", "South Carolina": "SC", "North Carolina": "NC"}

for selected_year in range(2017, 2020):
    print(f'\nProcessing for the year of {selected_year}')
    valid_df = df_final.copy()
    valid_df['DATE'] = pd.to_datetime(valid_df['DATE'])
    valid_df = valid_df[valid_df['DATE'].dt.year == selected_year]
    valid_df = valid_df.sort_values(by='DATE')

    for state_name in states:
        print(f'\n--- {state_name} ---')
        state_abbr = state_abbreviations[state_name]
        state_geom = GeoHelper.StatePolygon(state_name)

        # Filter by spatial location
        invalid_idx = []
        for idx, row in valid_df.iterrows():
            fire_point = geometry.Point(row["longitude"], row["latitude"])
            if isinstance(state_geom, geometry.MultiPolygon):
                if not any(poly.contains(fire_point) for poly in state_geom.geoms):
                    invalid_idx.append(idx)
            else:
                if not state_geom.contains(fire_point):
                    invalid_idx.append(idx)

        print(f"{len(invalid_idx)} fires out of {state_name}")
        state_df = valid_df.drop(index=invalid_idx).copy()

        # Remove invalid or missing ACRESBURNED
        state_df = state_df[state_df["ACRESBURNED"].apply(lambda x: is_number(x))]
        state_df["ACRESBURNED"] = state_df["ACRESBURNED"].astype(float)
        state_df = state_df[state_df["ACRESBURNED"] > 0]
        state_df = state_df[state_df["latitude"] > 0]

        print(f"#Valid fires for {state_abbr} in {selected_year}: {len(state_df)}")

        # Group to unique events, summing SCC phases (2811015001 + 2811015002)
        pol_cols = ['CO', 'CO2', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']
        out_cols = ['ACRESBURNED'] + pol_cols

        # Ensure numeric + fill missing
        for c in out_cols:
            if c not in state_df.columns:
                state_df[c] = 0.0
        state_df[out_cols] = state_df[out_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Map phases to a single SCC for grouping (choose 2811015002 as the combined label)
        phase_map = {2811015001: 2811015002, 2811015002: 2811015002}
        state_df['scc_group'] = state_df['scc'].map(phase_map).fillna(state_df['scc']).astype(int)

        # Aggregation: ACRESBURNED should not double-count; pollutants should add across phases
        agg_dict = {'ACRESBURNED': 'max'}
        agg_dict.update({c: 'sum' for c in pol_cols})

        merged_df = state_df.groupby(
            ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id',
             'process_id', 'scc_group', 'latitude', 'longitude', 'DATE'],
            as_index=False
        ).agg(agg_dict)

        # Keep column name 'scc' for downstream compatibility / output format
        merged_df = merged_df.rename(columns={'scc_group': 'scc'})

        merged_df['STATE'] = state_abbr
        merged_df['YEAR'] = selected_year

        print(f"#Total after merging: {len(merged_df)}")
        print(merged_df.scc.unique())

        if SHOW_PLOTS:
            # Optional plot
            if state_geom.geom_type == "MultiPolygon":
                for geom in state_geom.geoms:
                    xs, ys = geom.exterior.xy
                    plt.plot(xs, ys, 'k')
            else:
                xs, ys = state_geom.exterior.xy
                plt.plot(xs, ys, 'k')
            plt.scatter(merged_df["longitude"], merged_df["latitude"], s=1)
            plt.title(f"{state_abbr} Fires {selected_year}")
            plt.show()

        # Save cleaned RX daily
        if not merged_df.empty:
            output_file = f"{state_abbr}_{selected_year}_NEI_rx.csv"
            merged_df.to_csv(os.path.join(dir_python_local, output_file), index=False)
            print(f"Saved {output_file}")

# ---------------- Combine RX for multiple years (3-state subset) ----------------
combine_years_rx = [2017, 2018, 2019]
states_3 = ["Florida", "Georgia", "South Carolina"]
abbr_3 = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for selected_year in combine_years_rx:
    filenames = [f"{abbr_3[s]}_{selected_year}_NEI_rx.csv" for s in states_3]
    res_df = None
    for ix, filename in enumerate(filenames):
        path = os.path.join(dir_python_local, filename)
        if not os.path.exists(path):
            print(f"Missing: {filename} (skip)")
            continue
        df = pd.read_csv(path)
        res_df = df if res_df is None else pd.concat([res_df, df], ignore_index=True)
    if res_df is not None:
        out = f"SE_Combined_NEI_rx_3states_{selected_year}.csv"
        res_df.to_csv(out, index=False, header=True)
        print(f"Saved {out}")

# LANDFIRE .tif (FBFM40)
lf_files = {
    2014: "/work/chflab/jthuang/breadcrumbs/LandFire/US_140_FBFM40/Tif/us_140fbfm40.tif",
    2016: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    2020: "/work/chflab/jthuang/breadcrumbs/LandFire/LF2022_FBFM40_220_CONUS/Tif/LC22_F40_220.tif"
}
year_range = {
    2014: [2010, 2014],
    2016: [2015, 2016],
    2020: [2017, 2020]
}

# Restrict to 3 states for LANDFIRE step (matches your later combine logic)
states = ["Florida", "Georgia", "South Carolina"]
state_abbreviations = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for year in lf_files.keys():
    lf_file = lf_files[year]
    with rasterio.open(lf_file) as r:
        transformer_r = pyproj.Transformer.from_crs('epsg:4326', r.crs, always_xy=True)
        lower_time, upper_time = year_range[year][0], year_range[year][1] + 1

        for selected_year in range(lower_time, upper_time):
            for state_name in states:
                state_abbr = state_abbreviations[state_name]
                permit_file = f"{state_abbr}_{selected_year}_NEI_rx.csv"
                output_file = f"{state_abbr}_{selected_year}_NEI_rx_lf.csv"

                if os.path.exists(permit_file):
                    print("File exists.")
                    permit_df = pd.read_csv(permit_file)
                    permit_lat = permit_df['latitude'].to_numpy()
                    permit_lon = permit_df['longitude'].to_numpy()
                    permit_time = pd.to_datetime(permit_df['DATE'], errors='coerce')
                    burn_area = permit_df["ACRESBURNED"].to_numpy()

                    num_fires = len(permit_df)
                    print(state_abbr)
                    print("#Total:", num_fires)

                    burn_area = np.nan_to_num(burn_area)
                    coords = np.vstack((permit_lon, permit_lat)).T
                    coords_r = np.array([transformer_r.transform(lon, lat) for lon, lat in coords])

                    lf_attr_res = np.zeros(len(permit_df))

                    for i, coord_r in enumerate(coords_r):
                        permit_x, permit_y = coord_r
                        this_burn_area = burn_area[i] * 4046.8564224
                        buffer_length = max(math.sqrt(this_burn_area) / 2, 15)

                        buffer_point_ctr = Point(permit_x, permit_y)
                        if buffer_length <= 15:
                            print(f"In {state_abbr}, Too small: set the buffer length as 15m")
                            buffer_length = 15
                        buffer_grid = buffer_point_ctr.buffer(buffer_length, cap_style=3)

                        row, col = r.index(permit_x, permit_y)
                        point_type = r.read(1, window=((row, row + 1), (col, col + 1))).item()  # scalar

                        if this_burn_area <= 900:
                            lf_attr_res[i] = point_type
                        else:
                            stats = zonal_stats(buffer_grid, lf_file, add_stats={'counts': counts})
                            main_type = max_counts(stats[0]["counts"])
                            lf_attr_res[i] = main_type

                    permit_df['fuel_type'] = lf_attr_res
                    permit_df.to_csv(output_file, index=False)
                else:
                    print("File not found.")

# Classify to agr / invalid / rx for RX files
for selected_year in range(2010, 2021):
    print('Processing for the year of ' + str(selected_year))
    for state_name in states:
        state_abbr = state_abbreviations[state_name]
        permit_file = f"{state_abbr}_{selected_year}_NEI_rx_lf.csv"

        if os.path.exists(permit_file):
            print("File exists.")
            permit_df = pd.read_csv(permit_file)
            num_fires = len(permit_df)
            print(state_abbr)
            print("#Total:", num_fires)

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
            lf_map = {leg_str[i]: legend[i] for i in range(len(leg_str))}

            invalid_type = ["No Data", "Snow/Ice", "Open Water", "Barren"]
            agricultural_type = ["Agricultural"]

            invalid_type_num = [lf_map[t] for t in invalid_type]
            agricultural_type_num = lf_map[agricultural_type[0]]

            agri_index, invalid_index, valid_index = [], [], []
            for index, row in permit_df.iterrows():
                fuel_type = row["fuel_type"]
                if fuel_type == agricultural_type_num:
                    agri_index.append(index)
                elif fuel_type in invalid_type_num:
                    invalid_index.append(index)
                else:
                    valid_index.append(index)

            agri_df = permit_df.iloc[agri_index].reset_index(drop=True)
            invalid_df = permit_df.iloc[invalid_index].reset_index(drop=True)
            valid_df = permit_df.iloc[valid_index].reset_index(drop=True)

            agri_df.to_csv(f"{state_abbr}_{selected_year}_NEI_rx_lf_agr.csv", index=False)
            invalid_df.to_csv(f"{state_abbr}_{selected_year}_NEI_rx_lf_invalid.csv", index=False)
            valid_df.to_csv(f"{state_abbr}_{selected_year}_NEI_rx_lf_rx.csv", index=False)

# ---------------- AG daily (ptday_agburn/agfire) ----------------
fire_inv = [
    "ptday_agburn_2017_MYR_ff10_22apr2020_v0",
    "ptday_agburn_2018_ff10_csv_08mar2021_v0",
    "ptday_agfire_CONUS_2019ge_23dec2021_v0"
]
fl_fire_inv = [
    "ptday_fl_agburn_2018_ff10_csv_08mar2021_v0",
    "ptday_agfire_FL_2019ge_23dec2021_v0"
]

day_cols = [f"dayval{d}" for d in range(1, 32)]
meta_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'poll', 'monthnum']
grp_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'DATE']
pollutant_cols = ['ACRESBURNED', 'CO', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']
df_daily_list = []

def read_and_melt_csv(path, year):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.replace('\r\n', '\n').replace('\r', '\n').replace('\x0d', '') for line in f.readlines()]
    for i, line in enumerate(lines):
        if line.startswith('country_cd'):
            header_index = i
            break
    else:
        raise RuntimeError(f"Cannot find column headers in '{path}'")
    df = pd.read_csv(StringIO(''.join(lines[header_index:])))
    # Harmonize pollutant names before any processing
    df['poll'] = df['poll'].replace({
        'PM25-PRI': 'PM2_5',
        'PM10-PRI': 'PM10'
    })
    df_melt = df[meta_cols + day_cols].melt(id_vars=meta_cols, var_name='day_column', value_name='value')
    df_melt = df_melt[df_melt['value'].notna() & (df_melt['value'] != 0)]
    df_melt['day'] = df_melt['day_column'].str.extract(r'dayval(\d+)').astype(int)
    df_melt['DATE'] = pd.to_datetime({'year': year, 'month': df_melt['monthnum'], 'day': df_melt['day']}, errors='coerce')
    return df_melt[df_melt['DATE'].notna()]

def pivot_and_clean(df_melt):
    df_pivot = df_melt.pivot_table(index=grp_cols[:-1] + ['DATE'], columns='poll', values='value', aggfunc='sum').reset_index()
    df_pivot.columns.name = None
    for col in pollutant_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
    df_pivot = df_pivot[grp_cols + pollutant_cols].fillna(0)
    return df_pivot

Year = 2017
for k, fname in enumerate(fire_inv):
    input_path = os.path.join(dir_python_local, fname)
    if not os.path.isfile(input_path):
        print(f"Missing file: {fname} (skipping)")
        Year += 1
        continue
    print(f"\nProcessing '{fname}'...")
    df_melt = read_and_melt_csv(input_path, Year)
    df_pivot = pivot_and_clean(df_melt)

    if k > 0:
        fl_fname = fl_fire_inv[k-1]
        fl_path = os.path.join(dir_python_local, fl_fname)
        if os.path.isfile(fl_path):
            print(f"Processing '{fl_fname}'...")
            df_fl_melt = read_and_melt_csv(fl_path, Year)
            df_fl_pivot = pivot_and_clean(df_fl_melt)
            df_daily_list.extend([df_pivot, df_fl_pivot])
        else:
            df_daily_list.append(df_pivot)
    else:
        df_daily_list.append(df_pivot)
    Year += 1

# ---------------- AG annual (ptinv_agburn/agfire) ----------------
fire_inv = [
    "ptinv_agburn_2017_MYR_ff10_22apr2020_v0.csv",
    "ptinv_agburn_2018_ff10_08mar2021_v0.csv",
    "ptinv_agfire_CONUS_2019ge_23dec2021_v0.csv"
]
fl_fire_inv = [
    "ptinv_fl_agburn_2018_ff10_08mar2021_v0.csv",
    "ptinv_agfire_FL_2019ge_23dec2021_v0.csv"
]

all_files = [os.path.join(dir_python_local, f) for f in fire_inv]
fl_files = [os.path.join(dir_python_local, f) for f in fl_fire_inv]
meta_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'poll', 'longitude', 'latitude', 'ann_value']
grp_cols = ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc', 'longitude', 'latitude']
pollutant_cols = ['ACRESBURNED', 'CO', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']
df_annual_list = []

def read_and_pivot_csv(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.replace('\r\n', '\n').replace('\r', '\n').replace('\x0d', '') for line in f.readlines()]
    for i, line in enumerate(lines):
        if line.startswith('country_cd'):
            header_index = i
            break
    else:
        raise RuntimeError(f"Cannot find column headers in '{path}'")
    df = pd.read_csv(StringIO(''.join(lines[header_index:])))
    # Harmonize pollutant names before any processing
    df['poll'] = df['poll'].replace({
        'PM25-PRI': 'PM2_5',
        'PM10-PRI': 'PM10'
    })
    df_pivot = df.pivot_table(index=grp_cols, columns='poll', values='ann_value', aggfunc='sum').reset_index()
    df_pivot.columns.name = None
    for col in pollutant_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = 0
    df_pivot = df_pivot[grp_cols + pollutant_cols].fillna(0)
    return df_pivot

for k, path in enumerate(all_files):
    if not os.path.isfile(path):
        print(f"Missing file: {os.path.basename(path)} (skipping)")
        continue
    print(f"\nProcessing '{os.path.basename(path)}'...")
    if k == 0:
        df_ag = read_and_pivot_csv(path)
        df_annual_list.append(df_ag)
    else:
        df_conus = read_and_pivot_csv(path)
        fl_path = fl_files[k-1]
        if os.path.isfile(fl_path):
            df_fl = read_and_pivot_csv(fl_path)
            df_annual_list.extend([df_conus, df_fl])
        else:
            df_annual_list.append(df_conus)

# Merge AG daily+annual in order
merged_all_years = []
for df_day, df_ann in zip(df_daily_list, df_annual_list):
    merged_df = pd.merge(
        df_day, df_ann,
        on=['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id', 'scc'] + pollutant_cols,
        how='inner',
        suffixes=('', '_ann')
    )
    merged_all_years.append(merged_df)

df_final = pd.concat(merged_all_years, ignore_index=True)

# 7-state list for AG extraction
states = ["Florida", "Alabama", "Mississippi", "Georgia", "Tennessee", "South Carolina", "North Carolina"]
state_abbreviations = {"Florida": "FL", "Alabama": "AL", "Mississippi": "MS", "Georgia": "GA", "Tennessee": "TN", "South Carolina": "SC", "North Carolina": "NC"}

for selected_year in range(2017, 2020):
    print(f'\nProcessing for the year of {selected_year}')
    valid_df = df_final.copy()
    valid_df['DATE'] = pd.to_datetime(valid_df['DATE'])
    valid_df = valid_df[valid_df['DATE'].dt.year == selected_year]
    valid_df = valid_df.sort_values(by='DATE')

    for state_name in states:
        print(f'\n--- {state_name} ---')
        state_abbr = state_abbreviations[state_name]
        state_geom = GeoHelper.StatePolygon(state_name)

        invalid_idx = []
        for idx, row in valid_df.iterrows():
            fire_point = geometry.Point(row["longitude"], row["latitude"])
            if isinstance(state_geom, geometry.MultiPolygon):
                if not any(poly.contains(fire_point) for poly in state_geom.geoms):
                    invalid_idx.append(idx)
            else:
                if not state_geom.contains(fire_point):
                    invalid_idx.append(idx)

        print(f"{len(invalid_idx)} fires out of {state_name}")
        state_df = valid_df.drop(index=invalid_idx).copy()

        state_df = state_df[state_df["ACRESBURNED"].apply(lambda x: is_number(x))]
        state_df["ACRESBURNED"] = state_df["ACRESBURNED"].astype(float)
        state_df = state_df[state_df["ACRESBURNED"] > 0]
        state_df = state_df[state_df["latitude"] > 0]

        print(f"#Valid fires for {state_abbr} in {selected_year}: {len(state_df)}")

        # Group to unique events (include pollutants)
        out_cols = ['ACRESBURNED', 'CO', 'CO2', 'HFLUX', 'NH3', 'NOX', 'PM10', 'PM2_5', 'SO2', 'VOC']

        for c in out_cols:
            if c not in state_df.columns:
                state_df[c] = 0.0
        state_df[out_cols] = state_df[out_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        merged_df = state_df.groupby(
            ['country_cd', 'region_cd', 'facility_id', 'unit_id', 'rel_point_id', 'process_id',
             'scc', 'latitude', 'longitude', 'DATE'],
            as_index=False
        )[out_cols].sum()

        merged_df['STATE'] = state_abbr
        merged_df['YEAR'] = selected_year

        print(f"#Total after merging: {len(merged_df)}")
        print(merged_df.scc.unique())

        if SHOW_PLOTS:
            if state_geom.geom_type == "MultiPolygon":
                for geom in state_geom.geoms:
                    xs, ys = geom.exterior.xy
                    plt.plot(xs, ys, 'k')
            else:
                xs, ys = state_geom.exterior.xy
                plt.plot(xs, ys, 'k')
            plt.scatter(merged_df["longitude"], merged_df["latitude"], s=1)
            plt.title(f"{state_abbr} Fires {selected_year}")
            plt.show()

        if not merged_df.empty:
            output_file = f"{state_abbr}_{selected_year}_NEI_ag.csv"
            merged_df.to_csv(os.path.join(dir_python_local, output_file), index=False)
            print(f"Saved {output_file}")

# ---------------- Combine AG for multiple years (3-state subset) ----------------
combine_years_ag = [2017, 2018, 2019]  # extend as needed
states_3 = ["Florida", "Georgia", "South Carolina"]
abbr_3 = {"Florida": "FL", "Georgia": "GA", "South Carolina": "SC"}

for selected_year in combine_years_ag:
    filenames = [f"{abbr_3[s]}_{selected_year}_NEI_ag.csv" for s in states_3]
    res_df = None
    for ix, filename in enumerate(filenames):
        path = os.path.join(dir_python_local, filename)
        if not os.path.exists(path):
            print(f"Missing: {filename} (skip)")
            continue
        df = pd.read_csv(path)
        res_df = df if res_df is None else pd.concat([res_df, df], ignore_index=True)
    if res_df is not None:
        out = f"SE_Combined_NEI_ag_3states_{selected_year}.csv"
        res_df.to_csv(out, index=False, header=True)
        print(f"Saved {out}")