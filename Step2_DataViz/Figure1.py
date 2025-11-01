# -*- coding: utf-8 -*-
###############################################################################
# regrid_rxburn_acres.py 
# author: Jingting HUANG
# purpose: Regrid prescribed fire burned area over FL, GA, SC to CMAQ grid
# figures: Figure 1a (mean annual % burned from permits),
#          Figure 1b (permits - NEI difference in % burned),
#          Figure 1c (annual total acres by state & inventory, broken x-axis)
###############################################################################

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ------------------------- Environment / Paths -------------------------
# Use os.environ (not !export) so this works in scripts
os.environ["PROJ_LIB"]  = "/home/jh94030/.conda/envs/myenv/share/proj"
os.environ["PROJ_DATA"] = "/home/jh94030/.conda/envs/myenv/share/proj"

from pyproj import datadir
datadir.set_data_dir("/home/jh94030/.conda/envs/myenv/share/proj")  # keep consistent

dir_python_local   = "/home/jh94030/scripts/python/postdoc_project/rxfire/figure"
dir_python_scripts = "/home/jh94030/scripts/python/postdoc_project/rxfire/analysis"

sys.path.append(os.path.join(dir_python_scripts, "step3_BurnDataSelection"))
from util import CMAQGrid2D  # assumes this returns dict with 'Lon' and 'Lat'

os.chdir(dir_python_local)

# ------------------------- Config -------------------------
YEARS = [2017, 2018, 2019]
CELL_AREA_KM2    = 12 * 12             # 12km grid
CELL_AREA_ACRES  = CELL_AREA_KM2 * 247.105

# Data templates
NEI_TEMPLATE     = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/NEI_rxf_inv/SE_Combined_NEI_rx_3states_{}.csv"
PERMIT_TEMPLATE  = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/update_criteria/SE_Combined_Permit_lf_3states_rx_{}.csv"

# Shapefiles
STATES_SHP   = "/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
COUNTY_SHP   = "/work/chflab/jthuang/breadcrumbs/us_demo_county_2020/cb_2020_us_county_500k.shp"
# FEDLANDS_SHP = "/work/chflab/jthuang/breadcrumbs/federal_lands/USA_Federal_Lands.shp"

SE_ST_ABBR = ["FL", "GA", "SC"]
SE_ST_FIPS = ["12", "13", "45"]

# MCIP grid file
met_filedir        = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename  = f"{met_filedir}/METCRO2D_20170101.nc"

# ------------------------- Fonts -------------------------
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ------------------------- Helpers -------------------------
def load_cmaq_grid(metcro_file):
    info = CMAQGrid2D(metcro_file)
    lon, lat = info["Lon"], info["Lat"]
    nrows, ncols = lat.shape
    # Suggested (your prior approach):
    bot_left_lat  = lat[0, 0]   + 2
    bot_left_lon  = lon[0, 0]   + 32
    top_right_lat = lat[-1, -1] - 15
    top_right_lon = lon[-1, -1] - 22
    return lon, lat, nrows, ncols, (bot_left_lon, top_right_lon, bot_left_lat, top_right_lat)

def grid_kdtree(lat_grid, lon_grid):
    # Note: Euclidean in degrees; acceptable at ~12 km spacing.
    grid_pts = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    return cKDTree(grid_pts)

def build_state_mask(lon_grid, lat_grid, states_gdf):
    """Fast mask using a single union polygon and vectorized point-in-polygon."""
    # Ensure geodf is in lon/lat
    states_ll = states_gdf.to_crs(epsg=4326)
    union_geom = unary_union(states_ll.geometry.values)
    # Use grid-cell centers
    pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    # Vectorized contains check
    mask_flat = np.fromiter((union_geom.contains(Point(xy)) or union_geom.touches(Point(xy))
                             for xy in pts), dtype=bool, count=pts.shape[0])
    return mask_flat.reshape(lon_grid.shape)

def percent_grid_from_points(df, lat_col, lon_col, acres_col, years, tree, nrows, ncols):
    df = df.copy()
    df = df.dropna(subset=[lat_col, lon_col, acres_col])
    # map to nearest grid cell (lat, lon order consistent with KDTree)
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols

    # Sum by YEAR/ROW/COL, then compute per-cell annual mean across YEARS
    grouped = (df.groupby(["YEAR", "ROW", "COL"], observed=True)[acres_col]
                 .sum()
                 .unstack(level=0)
                 .reindex(columns=years)
                 .fillna(0.0))

    annual_mean_acres = grouped.mean(axis=1).values
    annual_mean_percent = (annual_mean_acres / CELL_AREA_ACRES) * 100.0

    grid = np.full((nrows, ncols), np.nan)
    for (row, col), pct in zip(grouped.index, annual_mean_percent):
        grid[row, col] = pct

    # Optional: clip extreme repeated-burn cells (commented to preserve your classes)
    # grid = np.clip(grid, 0, 100)

    return grid

def load_year_concat(template, years, parse_dates_col="DATE"):
    dfs = []
    for yr in years:
        df = pd.read_csv(template.format(yr), parse_dates=[parse_dates_col])
        df["YEAR"] = yr
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ------------------------- Load grid & shapes once -------------------------
cmaq_lon, cmaq_lat, nrows, ncols, (bl_lon, tr_lon, bl_lat, tr_lat) = load_cmaq_grid(metcro2d_filename)
tree = grid_kdtree(cmaq_lat, cmaq_lon)

gdf_states_all  = gpd.read_file(STATES_SHP)
gdf_SE          = gdf_states_all[gdf_states_all["STUSPS"].isin(SE_ST_ABBR)]
gdf_county_all  = gpd.read_file(COUNTY_SHP)
gdf_SE_county   = gdf_county_all[gdf_county_all["STATEFP"].isin(SE_ST_FIPS)]
# gdf_federal     = gpd.read_file(FEDLANDS_SHP)

# # Match CRS for fedlands selection and plotting
# target_ll = gdf_SE[["STUSPS", "geometry"]].to_crs(gdf_federal.crs).copy()
# target_ll = target_ll.dissolve(by="STUSPS", as_index=False)
# # Keep ONLY polygons strictly within each state boundary
# fed_fl_ga_sc = (gpd.sjoin(gdf_federal, target_ll, how="inner", predicate="intersects")
#                   .rename(columns={"STUSPS": "STATE"})
#                   .drop(columns=["index_right"]))

# ------------------------- Figure 1a: Mean annual % burned (permits) -------------------------
permits_df = load_year_concat(PERMIT_TEMPLATE, YEARS, parse_dates_col="DATE")
permits_pct_grid = percent_grid_from_points(
    df=permits_df, lat_col="LATITUDE", lon_col="LONGITUDE",
    acres_col="ACRES", years=YEARS, tree=tree, nrows=nrows, ncols=ncols
)
state_mask = build_state_mask(cmaq_lon, cmaq_lat, gdf_SE)
permits_pct_masked = np.ma.masked_where(~state_mask, permits_pct_grid)

# Colormap / classes
colors_a = ['#e6e4e6', '#eed5bb', '#f6c690', '#ee9e6b', '#d55e4d', '#bd1e2f']
cmap_a   = LinearSegmentedColormap.from_list('burned_percent', colors_a)
bounds_a = [0, 5, 10, 25, 50, 75, 100]
norm_a   = BoundaryNorm(bounds_a, cmap_a.N)

fig, ax = plt.subplots(
    1, figsize=(7, 5.25), dpi=600,
    subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)}
)
ax.set_extent([bl_lon, tr_lon, bl_lat, tr_lat], crs=ccrs.PlateCarree())
ax.axis('off')

# States fill + edges
ax.add_geometries(gdf_SE.to_crs(epsg=4326).geometry, crs=ccrs.PlateCarree(),
                  facecolor='#e6e4e6', edgecolor='k', linewidth=0, zorder=2)
ax.add_geometries(gdf_SE.to_crs(epsg=4326).geometry, crs=ccrs.PlateCarree(),
                  facecolor='none', edgecolor='k', linewidth=1.2, zorder=3)

# Grid raster
im = ax.pcolormesh(cmaq_lon, cmaq_lat, permits_pct_masked,
                   cmap=cmap_a, norm=norm_a, shading='auto',
                   transform=ccrs.PlateCarree(), zorder=2)

# State labels
for (txt, xy) in [("FL", (-84.5, 28.5)), ("GA", (-87.0, 32.5)), ("SC", (-78.5, 34.2))]:
    t = ax.text(xy[0], xy[1], txt, color='k', fontweight="bold", fontsize=16, transform=ccrs.Geodetic())
    t.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

# # Federal lands (plot in their native CRS if lon/lat; here enforce lon/lat)
# ax.add_geometries(fed_fl_ga_sc.to_crs(epsg=4326).geometry,
#                   crs=ccrs.PlateCarree(), facecolor="none",
#                   edgecolor="#4F8A8B", linewidth=0.8, zorder=2)

ax.add_feature(cfeature.LAKES, facecolor='w', edgecolor='k', linewidth=0.5, zorder=2)

# Cities
cities = {
    'Atlanta': (33.7490, -84.3880), 'Orlando': (28.5383, -81.3792),
    'Tallahassee': (30.4383, -84.2807), 'Columbia': (34.0007, -81.0348),
    'Jacksonville': (30.3322, -81.6557), 'Savannah': (32.0809, -81.0912),
    'Pensacola': (30.4213, -87.2169), 'Tampa': (27.9506, -82.4572),
    'Miami': (25.7617, -80.1918), 'Columbus': (32.4600, -84.9877),
    'Albany': (31.5785, -84.1557), 'Charleston': (32.7765, -79.9311)
}
for name, (lat, lon) in cities.items():
    ax.scatter(lon, lat, marker='o', facecolor='none', edgecolor='k', s=10,
               transform=ccrs.PlateCarree(), zorder=3)
    ax.text(lon + 0.15, lat - 0.1, name, fontsize=7, fontweight='bold', color='black',
            transform=ccrs.PlateCarree(), zorder=3,
            path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

# Colorbar (same label & tick size)
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.35, pad=0.05)
cbar.set_label('Mean annual reported burned area (%)', fontsize=9)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_fontsize(7)

plt.tight_layout()
plt.savefig(os.path.join(dir_python_local, "spatial_ba_permits.png"), bbox_inches='tight', dpi=600)
print("Max % Burned:", np.nanmax(permits_pct_masked.compressed()))
print("Min % Burned:", np.nanmin(permits_pct_masked.compressed()))

# ------------------------- Figure 1b: Permits - NEI (% burned difference) -------------------------
nei_df = load_year_concat(NEI_TEMPLATE, YEARS, parse_dates_col="DATE")

nei_pct_grid = percent_grid_from_points(
    df=nei_df, lat_col="latitude", lon_col="longitude",
    acres_col="ACRESBURNED", years=YEARS, tree=tree, nrows=nrows, ncols=ncols
)
nei_pct_masked     = np.ma.masked_where(~state_mask, nei_pct_grid)
permits_pct_masked = np.ma.masked_where(~state_mask, permits_pct_grid)

diff_pct = permits_pct_masked - nei_pct_masked  # masked arithmetic is fine

# colors_b = [
#     '#403E4B', '#746170', '#99879C', '#c2b7c6', '#C5E3F6',  # negatives
#     '#fee0b6', '#fdb863', '#e08214', '#b35806', '#7f3b08'   # positives
# ]
# bounds_b = [-100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100]

colors_b = [
    '#403E4B', '#746170', '#99879C', '#c2b7c6',  # negatives
    '#fee0b6', '#e08214', '#7f3b08'   # positives
]
bounds_b = [-72, -48, -24, -12, 0, 12, 24, 48]

cmap_b   = LinearSegmentedColormap.from_list('burned_percent_diff', colors_b)
norm_b   = BoundaryNorm(bounds_b, cmap_b.N)

fig, ax = plt.subplots(
    1, figsize=(7, 5.25), dpi=600,
    subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)}
)
ax.set_extent([bl_lon, tr_lon, bl_lat, tr_lat], crs=ccrs.PlateCarree())
ax.axis('off')

ax.add_geometries(gdf_SE.to_crs(epsg=4326).geometry, crs=ccrs.PlateCarree(),
                  facecolor='w', edgecolor='k', linewidth=0, zorder=2)
ax.add_geometries(gdf_SE.to_crs(epsg=4326).geometry, crs=ccrs.PlateCarree(),
                  facecolor='none', edgecolor='k', linewidth=1.2, zorder=3)

im = ax.pcolormesh(cmaq_lon, cmaq_lat, diff_pct,
                   cmap=cmap_b, norm=norm_b, shading='auto',
                   transform=ccrs.PlateCarree(), zorder=2)

for (txt, xy) in [("FL", (-84.5, 28.5)), ("GA", (-87.0, 32.5)), ("SC", (-78.5, 34.2))]:
    t = ax.text(xy[0], xy[1], txt, color='k', fontweight="bold", fontsize=16, transform=ccrs.Geodetic())
    t.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

ax.add_feature(cfeature.LAKES, facecolor='w', edgecolor='k', linewidth=0.5, zorder=2)

for name, (lat, lon) in cities.items():
    ax.scatter(lon, lat, marker='o', facecolor='none', edgecolor='k', s=10,
               transform=ccrs.PlateCarree(), zorder=3)
    ax.text(lon + 0.15, lat - 0.1, name, fontsize=7, fontweight='bold', color='black',
            transform=ccrs.PlateCarree(), zorder=3,
            path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.35, pad=0.05)
cbar.set_ticks(bounds_b)
cbar.set_ticklabels(bounds_b)
cbar.set_label('Difference between permits and NEI (%)', fontsize=9)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_fontsize(7)

plt.tight_layout()
plt.savefig(os.path.join(dir_python_local, "spatial_ba_diff.png"), bbox_inches='tight', dpi=600)
print("Max difference:", np.nanmax(diff_pct.compressed()))
print("Min difference:", np.nanmin(diff_pct.compressed()))

# ------------------------- Figure 1c: Annual total acres by state & inventory -------------------------
# Keep your original settings (figsize, font sizes, colors)
years  = [2017, 2018, 2019]
states = ['SC', 'GA', 'FL']
colors_year = {2017: '#C9E8ED', 2018: '#84B7D6', 2019: '#508CB6'}

# Summaries
df_NEI_summary = (nei_df.groupby(['STATE', 'YEAR'], observed=True)['ACRESBURNED']
                  .sum().reset_index())
df_permit_summary = (permits_df.groupby(['STATE', 'YEAR'], observed=True)['ACRES']
                     .sum().reset_index())

df_NEI_summary['STATE']    = pd.Categorical(df_NEI_summary['STATE'],    categories=states, ordered=True)
df_permit_summary['STATE'] = pd.Categorical(df_permit_summary['STATE'], categories=states, ordered=True)

#####################################################################################################
# --- Harmonize & combine summaries ---
tmp_nei = df_NEI_summary.rename(columns={"ACRESBURNED": "ACRES"}).copy()
tmp_nei["Inventory"] = "NEI"

tmp_per = df_permit_summary.rename(columns={"ACRES": "ACRES"}).copy()
tmp_per["Inventory"] = "Permits"

df_all = pd.concat([tmp_per, tmp_nei], ignore_index=True)

# Keep only the 3 states in desired order
state_order = ["FL", "GA", "SC"]
df_all = df_all[df_all["STATE"].isin(state_order)].copy()
df_all["STATE"] = pd.Categorical(df_all["STATE"], categories=state_order, ordered=True)

# --- Long table of annual totals (nice for SI / CSV) ---
annual_table_long = (
    df_all.groupby(["Inventory", "STATE", "YEAR"], observed=True)["ACRES"]
          .sum()
          .reset_index()
          .sort_values(["Inventory", "STATE", "YEAR"])
          .rename(columns={"STATE": "State", "YEAR": "Year", "ACRES": "Acres"})
)

# --- Wide table with mean & std across 2017-2019 (per state & inventory) ---
years_cols = [2017, 2018, 2019]
wide = (
    annual_table_long
      .pivot_table(index=["Inventory", "State"], columns="Year", values="Acres", aggfunc="sum")
      .reindex(columns=years_cols)  # ensure 2017, 2018, 2019 order
)

wide["Mean_2017_2019"] = wide[years_cols].mean(axis=1)                # arithmetic mean
wide["Std_2017_2019"]  = wide[years_cols].std(axis=1, ddof=1)         # sample std (n-1)
wide["Total_2017_2019"] = wide[years_cols].sum(axis=1)

summary_table = wide.reset_index()

# --- (Optional) pretty printing ---
with pd.option_context("display.max_rows", None, "display.float_format", "{:,.0f}".format):
    print("\nAnnual burned acres by state & inventory (2017–2019):")
    print(annual_table_long)

    print("\nPer-state mean, std, and totals across 2017–2019 (by inventory):")
    print(summary_table)

# --- Save CSVs next to figures ---
annual_csv  = os.path.join(dir_python_local, "annual_burned_acres_by_state_2017_2019.csv")
summary_csv = os.path.join(dir_python_local, "burned_acres_state_stats_2017_2019.csv")
annual_table_long.to_csv(annual_csv, index=False)
summary_table.to_csv(os.path.join(dir_python_local, summary_csv), index=False)
print(f"\nWrote:\n  {annual_csv}\n  {summary_csv}")
#####################################################################################################
# y positions and labels
y_labels    = []
y_positions = {}
for i, st in enumerate(states):
    y_positions[(st, 'Permits')] = i * 2 + 0.3
    y_positions[(st, 'NEI')]     = i * 2 - 0.3
    y_labels.extend([f"{st} (Permits)", f"{st} (NEI)"])

# Determine break points (same as yours)
break_low  = 4.4e5
break_high = 0.9e6

fig = plt.figure(figsize=(7, 2.8))
gs  = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.02)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

def plot_dataset_broken(df, val_col, label, size, ax_low, ax_high):
    for st in states:
        sub = df[df['STATE'] == st].sort_values("YEAR")
        if sub.empty:
            continue
        y = y_positions[(st, label)]
        # left axis
        sub_low = sub[sub[val_col] <= break_low]
        if not sub_low.empty:
            ax_low.plot(sub_low[val_col], np.full(len(sub_low), y), color='grey', lw=3, zorder=2)
            ax_low.scatter(sub_low[val_col], np.full(len(sub_low), y), s=size,
                           c=[colors_year[yr] for yr in sub_low['YEAR']],
                           edgecolor='#312738', zorder=3)
        # right axis
        sub_high = sub[sub[val_col] >= break_high]
        if not sub_high.empty:
            ax_high.plot(sub_high[val_col], np.full(len(sub_high), y), color='grey', lw=3, zorder=2)
            ax_high.scatter(sub_high[val_col], np.full(len(sub_high), y), s=size,
                            c=[colors_year[yr] for yr in sub_high['YEAR']],
                            edgecolor='#312738', zorder=3)

plot_dataset_broken(df_permit_summary, 'ACRES',        'Permits', size=200, ax_low=ax1, ax_high=ax2)
plot_dataset_broken(df_NEI_summary,    'ACRESBURNED',  'NEI',     size=50,  ax_low=ax1, ax_high=ax2)

# axis limits
ax1.set_xlim(3.49e5, break_low)
ax2.set_xlim(break_high, max(df_permit_summary['ACRES'].max(),
                             df_NEI_summary['ACRESBURNED'].max()) * 1.05)

# formatting (kept)
ax1.set_xlabel('', fontsize=14)
ax1.set_yticks(list(y_positions.values()))
ax1.set_yticklabels(y_labels, fontsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.ticklabel_format(axis='x', style='sci', scilimits=(5, 5))
if isinstance(ax1.xaxis.get_major_formatter(), ScalarFormatter):
    ax1.xaxis.get_major_formatter().set_useMathText(True)

ax2.set_xlabel('Reported acres burned per year', fontsize=14, labelpad=20)
ax2.set_yticks(list(y_positions.values()))
ax2.set_yticklabels([])
ax2.tick_params(axis='x', labelsize=12)
ax2.ticklabel_format(axis='x', style='sci', scilimits=(5, 5))
if isinstance(ax2.xaxis.get_major_formatter(), ScalarFormatter):
    ax2.xaxis.get_major_formatter().set_useMathText(True)

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.grid(axis='y', linestyle=':', linewidth=1, zorder=1)
    ax.set_ylim(-1, len(states) * 2)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax2.tick_params(axis='y', which='both', length=0)

# break marks
d = 0.01
ax1.plot((1-d*2, 1+d*2), (-d*2, +d*2), transform=ax1.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d*2, +d*2),       transform=ax2.transAxes, color='k', clip_on=False)

legend_dots = [Line2D([0], [0], marker='o', color='w', label=str(yr),
                      markerfacecolor=colors_year[yr], markeredgecolor='#312738', markersize=10)
               for yr in years]
ax2.legend(handles=legend_dots, loc='upper left', bbox_to_anchor=(0, 1.05),
           ncol=3, frameon=False, fontsize=12)

plt.savefig(os.path.join(dir_python_local, "annual_ba_diff.png"), bbox_inches='tight', dpi=600)