###############################################################################
# spatial_maps_pm2.5.py  (refactored + season toggle)
# author: Jingting HUANG
###############################################################################
import os
import sys
import glob
import numpy as np
import pandas as pd

from datetime import datetime
from netCDF4 import Dataset
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm

import cartopy.crs as ccrs

import colorcet as cc
from scipy.stats import gaussian_kde

# ------------------------- Fonts & Matplotlib -------------------------
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
plt.rcParams['font.family'] = 'Arial'

# ------------------------- Paths & Setup -------------------------
print('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'

dir_python_scripts = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
sys.path.append(os.path.join(dir_python_scripts, 'step3_BurnDataSelection'))
from util import CMAQGrid2D  # noqa: E402

os.chdir(dir_python_local)
print('cwd is %s ' % (os.getcwd()))

# CMAQ output directories
CMAQ_ALL_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift_old/combined/hr2dy'
CMAQ_RMV_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'

# Years & threshold
YEARS = [2017, 2018, 2019]
SMOKE_THRESHOLD = 3.5  # µg/m3

# ---- Season toggle ----
SEASON = "high"  # options: "all", "high", "low"
SEASON_MONTHS = {
    "all":  tuple(range(1, 13)),
    "high": (1, 2, 3, 4),                    # Jan–Apr
    "low":  (5, 6, 7, 8, 9, 10, 11, 12),     # May–Dec
}
def season_suffix(season: str) -> str:
    return {"all": "", "high": "_highburn", "low": "_lowburn"}[season]

# Take the CET_L11 colormap
cmap_orig = cc.cm.CET_L11
cmap_new = LinearSegmentedColormap.from_list(
    "modifiedCET_L11",
    [ (0.0, cmap_orig(0.0)),     # low end = green
      (0.25, cmap_orig(0.25)),  
      (0.5, cmap_orig(0.5)),     # mid = yellow/white
      (1.0, "darkorange") ]      # high end = dark orange
)

# ------------------------- Load Shapefiles -------------------------
# States
shapefile_path_states = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'
gdf_states = gpd.read_file(shapefile_path_states)
gdf_SE = gdf_states[gdf_states['STUSPS'].isin(['FL', 'GA', 'SC'])]
gdf_FL = gdf_states[gdf_states['STUSPS'].isin(['FL'])]
gdf_GA = gdf_states[gdf_states['STUSPS'].isin(['GA'])]
gdf_SC = gdf_states[gdf_states['STUSPS'].isin(['SC'])]

# Counties (for overlay)
shapefile_path_county = os.path.join('/work/chflab/jthuang/breadcrumbs', 'us_demo_county_2020', 'cb_2020_us_county_500k.shp')
gdf_county_bounds = gpd.read_file(shapefile_path_county)
SE_states_fips = ['12', '13', '45']
gdf_SE_county = gdf_county_bounds[gdf_county_bounds['STATEFP'].isin(SE_states_fips)]

# Nonattainment county overlay (GA only in your list)
selected_counties = ['Dougherty', 'Fulton', 'Gwinnett', 'Richmond', 'Walker']
selected_counties_gdf = gdf_SE_county[
    (gdf_SE_county['NAME'].isin(selected_counties)) &
    (gdf_SE_county['STATE_NAME'] == 'Georgia')
].copy()
selected_counties_gdf.to_file('nonattainment_counties_obs.shp')

# ------------------------- CMAQ Grid -------------------------
met_filedir = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename = f"{met_filedir}/METCRO2D_20170101.nc"
cmaq_info = CMAQGrid2D(metcro2d_filename)
cmaq_lon, cmaq_lat = cmaq_info['Lon'], cmaq_info['Lat']

# Map extent (unchanged)
bot_left_lat  = cmaq_lat[0, 0] + 2
bot_left_lon  = cmaq_lon[0, 0] + 32
top_right_lat = cmaq_lat[-1, -1] - 15
top_right_lon = cmaq_lon[-1, -1] - 22

# ------------------------- Helpers -------------------------
def mask_from_gdf(lon_grid, lat_grid, gdf):
    """
    Build a boolean mask for grid cells within/touching geometries in gdf.
    Uses a prepared unary union to speed up point-in-polygon checks.
    """
    union_geom = gdf.geometry.unary_union
    prepared = prep(union_geom)
    flat_points = (lon_grid.ravel(), lat_grid.ravel())
    mask_flat = np.fromiter(
        (prepared.contains(Point(x, y)) or prepared.touches(Point(x, y)) for x, y in zip(*flat_points)),
        dtype=bool,
        count=lon_grid.size
    )
    return mask_flat.reshape(lon_grid.shape)

# Build masks once
mask    = mask_from_gdf(cmaq_lon, cmaq_lat, gdf_SE)
mask_fl = mask_from_gdf(cmaq_lon, cmaq_lat, gdf_FL)
mask_ga = mask_from_gdf(cmaq_lon, cmaq_lat, gdf_GA)
mask_sc = mask_from_gdf(cmaq_lon, cmaq_lat, gdf_SC)

def open_cmaq(year):
    """Open paired ALL and RMV datasets for a year."""
    f_all = os.path.join(CMAQ_ALL_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
    f_rmv = os.path.join(CMAQ_RMV_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
    ds_all = xr.open_dataset(f_all)
    ds_rmv = xr.open_dataset(f_rmv)
    return ds_all, ds_rmv

def fire_pm25_year(ds_all, ds_rmv):
    """Compute daily Rx PM2.5 (>=0), surface layer only, for a year."""
    fp = (ds_all['PM25_TOT_AVG'] - ds_rmv['PM25_TOT_AVG']).isel(LAY=0)
    return fp.where(fp > 0, 0)

def month_indices(ds_all, months):
    """
    Return numpy indices for TSTEPs whose month is in `months`.
    Uses IOAPI TFLAG[:, 0, 0] (YYYYDDD).
    """
    tflag = ds_all['TFLAG'][:, 0, 0].values  # shape (TSTEP,)
    months_arr = []
    for v in tflag:
        m = datetime.strptime(str(int(v)), '%Y%j').month
        months_arr.append(m)
    months_arr = np.array(months_arr)
    return np.where(np.isin(months_arr, months))[0]

def multi_year_annual_mean_pm25(years, season="all"):
    """
    For each year: select season months -> mean over days (TSTEP) of daily Rx PM2.5.
    Then average these annual means across years.
    """
    months = SEASON_MONTHS[season]
    annual_means = []
    for yr in years:
        print(f"[PM2.5] Processing year: {yr} (season={season})")
        ds_all, ds_rmv = open_cmaq(yr)
        fp = fire_pm25_year(ds_all, ds_rmv)  # (TSTEP, Y, X)
        idx = month_indices(ds_all, months)
        if idx.size > 0:
            fp_sel = fp.isel(TSTEP=idx)
            annual_means.append(fp_sel.mean(dim='TSTEP').values)  # (Y, X)
        else:
            annual_means.append(np.zeros_like(fp.isel(TSTEP=0).values))
        ds_all.close(); ds_rmv.close()
    annual_means = np.stack(annual_means, axis=0)  # (year, Y, X)
    return np.nanmean(annual_means, axis=0)        # (Y, X)

def multi_year_avg_smoke_days(years, threshold, season="all"):
    """
    For each year: select season months -> count of days with Rx PM2.5 > threshold.
    Then average the counts across years to get 'annual average high Rx fire smoke days'.
    """
    months = SEASON_MONTHS[season]
    per_year_counts = []
    for yr in years:
        print(f"[SmokeDays] Processing year: {yr} (season={season})")
        ds_all, ds_rmv = open_cmaq(yr)
        fp = fire_pm25_year(ds_all, ds_rmv)  # (TSTEP, Y, X)
        idx = month_indices(ds_all, months)
        if idx.size > 0:
            fp_sel = fp.isel(TSTEP=idx)
            counts = (fp_sel > threshold).sum(dim='TSTEP').values  # (Y, X)
        else:
            counts = np.zeros_like(fp.isel(TSTEP=0).values)
        per_year_counts.append(counts.astype(float))
        ds_all.close(); ds_rmv.close()
    per_year_counts = np.stack(per_year_counts, axis=0)  # (year, Y, X)
    return np.mean(per_year_counts, axis=0)              # (Y, X) float

def clean_flat(data, threshold=0.01):
    """Flatten, drop NaN, and keep values > threshold."""
    data = data.flatten()
    return data[~np.isnan(data) & (data > threshold)]

# ------------------------- Compute data (season-aware) -------------------------
_suffix = season_suffix(SEASON)

# PM2.5 seasonal multi-year average of per-year means
mean_fire_pm25 = multi_year_annual_mean_pm25(YEARS, season=SEASON)
mean_fire_pm25_masked = np.where(mask, mean_fire_pm25, np.nan)

# Smoke-days seasonal multi-year average of per-year counts
mean_smoke_day_counts = multi_year_avg_smoke_days(YEARS, SMOKE_THRESHOLD, season=SEASON)
mean_smoke_day_counts_masked = np.where(mask, mean_smoke_day_counts, np.nan)

# ------------------------- Figure 3a: Annual avg Rx fire PM2.5 -------------------------
fig, ax = plt.subplots(figsize=(6, 5), dpi=600,
                       subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
ax.set_extent([-91, -75, 24, 37])
ax.axis('off')

ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                  edgecolor='k', linewidth=1.2, zorder=3)
ax.add_geometries(selected_counties_gdf.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                  edgecolor='#8C0909', linewidth=1, zorder=3)

if SEASON == "all":
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_fire_pm25_masked, vmin=0, vmax=3.5,
                   cmap=cmap_new, transform=ccrs.PlateCarree(), zorder=2)

elif SEASON == "high":
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_fire_pm25_masked, vmin=0, vmax=9,
                   cmap=cmap_new, transform=ccrs.PlateCarree(), zorder=2)

elif SEASON == "low":
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_fire_pm25_masked, vmin=0, vmax=1,
                   cmap=cmap_new, transform=ccrs.PlateCarree(), zorder=2)

# ------------------------- Colorbar & x-axis settings based on season -------------------------
if SEASON == "all":
    # Original all-season settings
    bounds_pm = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    xticks = np.linspace(0, 3.5, 8)
    xtick_labels = [f"{v:.2g}" for v in xticks]

elif SEASON == "high":
    # High-burn season settings
    bounds_pm = [0, 3, 6, 9]   # keep same colorbar bounds
    xticks = np.linspace(0, 9, 4)               # custom x-axis ticks for gradient plot
    xtick_labels = [f"{v:.2g}" for v in xticks]

elif SEASON == "low":
    # Low-burn season settings
    bounds_pm = [0, 0.25, 0.5, 0.75, 1]            # low-burn colorbar
    xticks = np.linspace(0, 1, 5)               # align gradient plot x-axis to colorbar
    xtick_labels = [f"{v:.2g}" for v in xticks]

# ------------------------- Apply to colorbar -------------------------
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
cbar.set_ticks(bounds_pm)
cbar.set_ticklabels(bounds_pm)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_fontsize(7)

cbar.set_label('Annual average Rx fire $\\mathrm{PM}_{2.5}$ ($\\mu g/m^3$)', fontsize=7)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_fontsize(7)

plt.savefig(os.path.join(dir_python_local, f"pm25_fire_2017_2019{_suffix}.png"), bbox_inches='tight', dpi=600)
plt.close()

# ------------------------- Joyplot (gradient) for PM2.5 -------------------------
mean_fire_pm25_fl = np.where(mask_fl, mean_fire_pm25, np.nan)
mean_fire_pm25_ga = np.where(mask_ga, mean_fire_pm25, np.nan)
mean_fire_pm25_sc = np.where(mask_sc, mean_fire_pm25, np.nan)

data_fl = clean_flat(mean_fire_pm25_fl)
data_ga = clean_flat(mean_fire_pm25_ga)
data_sc = clean_flat(mean_fire_pm25_sc)

df_long = pd.concat([
    pd.DataFrame({'State': 'FL', 'PM25': data_fl}),
    pd.DataFrame({'State': 'GA', 'PM25': data_ga}),
    pd.DataFrame({'State': 'SC', 'PM25': data_sc}),
], ignore_index=True)

cmap = cmap_new

if SEASON == "all":
    norm = Normalize(vmin=0, vmax=3.5)

elif SEASON == "high":
    norm = Normalize(vmin=0, vmax=9)

elif SEASON == "low":
    norm = Normalize(vmin=0, vmax=1)

state_list = ['FL', 'GA', 'SC']
kde_results = {}
x_grid = np.linspace(0, 10, 500)

for state in state_list:
    values = df_long[df_long['State'] == state]['PM25'].values
    kde = gaussian_kde(values, bw_method=0.3)
    kde_results[state] = kde(x_grid)

fig, axarr = plt.subplots(len(state_list), 1, figsize=(6, 4), dpi=600, sharex=True)

for i, (ax, state) in enumerate(zip(axarr, state_list)):
    y_vals = kde_results[state]
    offset = i * 1.0
    y_offset = y_vals + offset

    for j in range(len(x_grid) - 1):
        x_seg = x_grid[j:j+2]
        y_seg = y_offset[j:j+2]
        color = cmap(norm(np.mean(x_seg)))
        ax.fill_between(x_seg, y_seg, offset, color=color, linewidth=0)

    ax.plot(x_grid, y_offset, color='black', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.set_yticks([])
    ax.set_ylabel(state, fontsize=20, fontweight='semibold', rotation=0, labelpad=20, va='center')

    if state == 'SC':
        ax.tick_params(axis='x', labelsize=18, direction='out', bottom=True)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)

    if SEASON == "all":
        ax.set_xlim(0, 3.5)
        ax.set_ylim(offset, offset + 2)

    elif SEASON == "high":
        ax.set_xlim(0, 9)
        ax.set_ylim(offset, offset + 0.8)
    
    elif SEASON == "low":
        ax.set_xlim(0, 1)
        ax.set_ylim(offset, offset + 8)
    
axarr[-1].set_xticks(xticks)
axarr[-1].set_xticklabels(xtick_labels, fontsize=10)
axarr[-1].tick_params(axis='x', bottom=True, direction='out')
axarr[-1].set_xlabel("Annual average Rx fire $\\mathrm{PM}_{2.5}$ ($\\mu g/m^3$)", fontsize=15)

plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15, hspace=0.1)
plt.savefig(os.path.join(dir_python_local, f"joyplot_gradient_rx_fire_pm25{_suffix}.png"), dpi=600)
plt.close()

# ------------------------- Figure 3b: Annual average high Rx fire smoke days -------------------------
# Colormap (unchanged)
colors = [
    '#B6B3D6', '#CFCCE3', '#D5D3DE', '#D5D1D1',
    '#F6DFD6', '#F8B2A2', '#F1837A', '#E9687A'
]

if SEASON == "all":
    bounds_sd = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    cmap_sd = LinearSegmentedColormap.from_list('smoke_days', colors)
    norm_sd = BoundaryNorm(bounds_sd, cmap_sd.N)

elif SEASON == "high":
    bounds_sd = [0, 10, 20, 30, 40, 50, 60, 70]
    cmap_sd = LinearSegmentedColormap.from_list('smoke_days', colors)
    norm_sd = BoundaryNorm(bounds_sd, cmap_sd.N)

elif SEASON == "low":
    bounds_sd = [0, 10, 20]
    cmap_sd = LinearSegmentedColormap.from_list('smoke_days', colors)
    norm_sd = BoundaryNorm(bounds_sd, cmap_sd.N)

fig, ax = plt.subplots(figsize=(6, 5), dpi=600,
                       subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
ax.set_extent([-91, -75, 24, 37])
ax.axis('off')

ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                  edgecolor='k', linewidth=1.2, zorder=3)

im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_smoke_day_counts_masked,
                   cmap=cmap_sd, norm=norm_sd, transform=ccrs.PlateCarree(), zorder=2)

cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05)
cbar.set_ticks(bounds_sd)
cbar.set_ticklabels(bounds_sd)
cbar.set_label('Annual average high Rx fire smoke days', fontsize=7)
for l in cbar.ax.xaxis.get_ticklabels():
    l.set_fontsize(7)

plt.savefig(os.path.join(dir_python_local, f"pm25_fire_smoke_days_2017_2019{_suffix}.png"),
            bbox_inches='tight', dpi=600)
plt.close()

# ------------------------- Joyplot (gradient) for smoke days -------------------------
mean_smoke_day_counts_fl = np.where(mask_fl, mean_smoke_day_counts_masked, np.nan)
mean_smoke_day_counts_ga = np.where(mask_ga, mean_smoke_day_counts_masked, np.nan)
mean_smoke_day_counts_sc = np.where(mask_sc, mean_smoke_day_counts_masked, np.nan)

data_fl_sd = clean_flat(mean_smoke_day_counts_fl)
data_ga_sd = clean_flat(mean_smoke_day_counts_ga)
data_sc_sd = clean_flat(mean_smoke_day_counts_sc)

df_long_sd = pd.concat([
    pd.DataFrame({'State': 'FL', 'Smoke_Days': data_fl_sd}),
    pd.DataFrame({'State': 'GA', 'Smoke_Days': data_ga_sd}),
    pd.DataFrame({'State': 'SC', 'Smoke_Days': data_sc_sd}),
], ignore_index=True)

state_list_sd = ['FL', 'GA', 'SC']
kde_results_sd = {}

if SEASON == "all":
    x_grid_sd = np.linspace(0, 80, 500)

elif SEASON == "high":
    x_grid_sd = np.linspace(0, 70, 500)

elif SEASON == "low":
    x_grid_sd = np.linspace(0, 20, 500)

for state in state_list_sd:
    values = df_long_sd[df_long_sd['State'] == state]['Smoke_Days'].values
    kde = gaussian_kde(values, bw_method=0.3)
    kde_results_sd[state] = kde(x_grid_sd)

fig, axarr = plt.subplots(len(state_list_sd), 1, figsize=(6, 4), dpi=600, sharex=True)

for i, (ax, state) in enumerate(zip(axarr, state_list_sd)):
    y_vals = kde_results_sd[state]
    offset = i * 1.0
    y_offset = y_vals + offset

    for j in range(len(x_grid_sd) - 1):
        x_seg = x_grid_sd[j:j+2]
        y_seg = y_offset[j:j+2]
        color = cmap_sd(norm_sd(np.mean(x_seg)))
        ax.fill_between(x_seg, y_seg, offset, color=color, linewidth=0)

    ax.plot(x_grid_sd, y_offset, color='black', linewidth=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.set_yticks([])
    ax.set_ylabel(state, fontsize=20, fontweight='semibold', rotation=0, labelpad=20, va='center')

    if state == 'SC':
        ax.tick_params(axis='x', labelsize=18, direction='out', bottom=True)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)

    if SEASON == "all":
        ax.set_xlim(0, 80)
        ax.set_ylim(offset, offset + 0.08)
        xticks_sd = np.linspace(0, 80, 9)

    elif SEASON == "high":
        ax.set_xlim(0, 70)
        ax.set_ylim(offset, offset + 0.08)
        xticks_sd = np.linspace(0, 70, 8)

    elif SEASON == "low":
        ax.set_xlim(0, 20)
        ax.set_ylim(offset, offset + 0.36)
        xticks_sd = np.linspace(0, 20, 3)

xtick_labels_sd = [f"{v:.2g}" for v in xticks_sd]
axarr[-1].set_xticks(xticks_sd)
axarr[-1].set_xticklabels(xtick_labels_sd, fontsize=10)
axarr[-1].tick_params(axis='x', bottom=True, direction='out')
axarr[-1].set_xlabel("Annual average high Rx fire smoke days", fontsize=15)

plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15, hspace=0.1)
plt.savefig(os.path.join(dir_python_local, f"joyplot_gradient_rx_fire_smoke_days{_suffix}.png"), dpi=600)
plt.close()