# -*- coding: utf-8 -*-
"""
graphic_abstract_all.py
Replot four figures (transparent background) and compose a final graphical abstract
- Figure 1a: Mean annual % burned from permits
- Figure 3a: Annual avg Rx fire PM2.5 (2017–2019)
- Figure 3b: Annual avg high Rx fire smoke days
- Figure 4a: Population aggregated to 12 km
Then export: a 2x2 grid + a tilted, layered "stacked cards" composition.

Author: Jingting HUANG (edits packaged)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.spatial import cKDTree
from datetime import datetime

import xarray as xr
import rasterio

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, LogNorm
from matplotlib import font_manager, transforms
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import colorcet as cc

# ------------------------- Environment / Fonts -------------------------
os.environ["PROJ_LIB"]  = "/home/jh94030/.conda/envs/myenv/share/proj"
os.environ["PROJ_DATA"] = "/home/jh94030/.conda/envs/myenv/share/proj"

font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ------------------------- Paths -------------------------
dir_python_local   = "/home/jh94030/scripts/python/postdoc_project/rxfire/figure"
dir_python_scripts = "/home/jh94030/scripts/python/postdoc_project/rxfire/analysis"
os.makedirs(dir_python_local, exist_ok=True)
os.chdir(dir_python_local)

# Data templates
YEARS = [2017, 2018, 2019]
NEI_TEMPLATE     = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/NEI_rxf_inv/SE_Combined_NEI_rx_3states_{}.csv"
PERMIT_TEMPLATE  = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/SE_Combined_Permit_lf_3states_rx_{}.csv"

# Shapefiles
STATES_SHP = "/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
COUNTY_SHP = "/work/chflab/jthuang/breadcrumbs/us_demo_county_2020/cb_2020_us_county_500k.shp"
SE_ST_ABBR = ["FL", "GA", "SC"]
SE_ST_FIPS = ["12", "13", "45"]

# CMAQ Grid / MET
met_filedir       = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename = f"{met_filedir}/METCRO2D_20170101.nc"

# CMAQ outputs
CMAQ_ALL_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift_old/combined/hr2dy'
CMAQ_RMV_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'
SMOKE_THRESHOLD = 3.5  # µg/m3

# Population raster
POP_TIF = "/work/chflab/jthuang/breadcrumbs/ciesen_nasa/ciesen_nasa_gpw_v4_population_count_2020.tif"

# Take the CET_L11 colormap
cmap_orig = cc.cm.CET_L11
cmap_new = LinearSegmentedColormap.from_list(
    "modifiedCET_L11",
    [ (0.0, cmap_orig(0.0)),     # low end = green
      (0.25, cmap_orig(0.25)),  
      (0.5, cmap_orig(0.5)),     # mid = yellow/white
      (1.0, "darkorange") ]      # high end = dark orange
)

# Output filenames (transparent)
OUT_1A = os.path.join(dir_python_local, "spatial_ba_permits_GA.png")
OUT_3A = os.path.join(dir_python_local, "pm25_fire_2017_2019_graphic_GA.png")
OUT_3B = os.path.join(dir_python_local, "pm25_fire_smoke_days_2017_2019_GA.png")
OUT_4A = os.path.join(dir_python_local, "reaggregated_pop_12km_GA.png")
OUT_DONUT = os.path.join(dir_python_local, "monthly_burn_donut_2017_2019_GA.png")

# ------------------------- Utils -------------------------
sys.path.append(os.path.join(dir_python_scripts, "step3_BurnDataSelection"))
from util import CMAQGrid2D  # noqa: E402

CELL_AREA_KM2   = 12 * 12
CELL_AREA_ACRES = CELL_AREA_KM2 * 247.105

def load_cmaq_grid(metcro_file):
    info = CMAQGrid2D(metcro_file)
    lon, lat = info["Lon"], info["Lat"]
    return lon, lat

def grid_kdtree(lat_grid, lon_grid):
    grid_pts = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    return cKDTree(grid_pts)

def build_state_mask(lon_grid, lat_grid, states_gdf):
    states_ll = states_gdf.to_crs(epsg=4326)
    union_geom = unary_union(states_ll.geometry.values)
    prepared = prep(union_geom)
    pts_lon = lon_grid.ravel(); pts_lat = lat_grid.ravel()
    mask_flat = np.fromiter(
        (prepared.contains(Point(x, y)) or prepared.touches(Point(x, y))
         for x, y in zip(pts_lon, pts_lat)),
        dtype=bool, count=lon_grid.size
    )
    return mask_flat.reshape(lon_grid.shape)

def load_year_concat(template, years, parse_dates_col="DATE"):
    dfs = []
    for yr in years:
        df = pd.read_csv(template.format(yr), parse_dates=[parse_dates_col])
        df["YEAR"] = yr
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def percent_grid_from_points(df, lat_col, lon_col, acres_col, years, tree, nrows, ncols):
    df = df.dropna(subset=[lat_col, lon_col, acres_col]).copy()
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols
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
    return grid

def open_cmaq(year):
    f_all = os.path.join(CMAQ_ALL_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
    f_rmv = os.path.join(CMAQ_RMV_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
    return xr.open_dataset(f_all), xr.open_dataset(f_rmv)

def fire_pm25_year(ds_all, ds_rmv):
    fp = (ds_all['PM25_TOT_AVG'] - ds_rmv['PM25_TOT_AVG']).isel(LAY=0)
    return fp.where(fp > 0, 0)

def multi_year_annual_mean_pm25(years):
    annual_means = []
    for yr in years:
        ds_all, ds_rmv = open_cmaq(yr)
        fp = fire_pm25_year(ds_all, ds_rmv)  # (TSTEP, Y, X)
        annual_means.append(fp.mean(dim='TSTEP').values)
        ds_all.close(); ds_rmv.close()
    annual_means = np.stack(annual_means, axis=0)
    return np.nanmean(annual_means, axis=0)

def multi_year_avg_smoke_days(years, threshold):
    per_year_counts = []
    for yr in years:
        ds_all, ds_rmv = open_cmaq(yr)
        fp = fire_pm25_year(ds_all, ds_rmv)
        counts = (fp > threshold).sum(dim='TSTEP').values
        per_year_counts.append(counts.astype(float))
        ds_all.close(); ds_rmv.close()
    per_year_counts = np.stack(per_year_counts, axis=0)
    return np.mean(per_year_counts, axis=0)

def load_population(pop_tif):
    with rasterio.open(pop_tif) as src:
        pop = src.read(1).astype(np.float64)
        if src.nodata is not None:
            pop = np.where(pop == src.nodata, 0.0, pop)
        width, height = src.width, src.height
        x_coords = np.arange(width) + 0.5
        y_coords = np.arange(height) + 0.5
        lon_vals = src.transform.c + x_coords * src.transform.a
        lat_vals = src.transform.f + y_coords * src.transform.e
        if lat_vals[0] > lat_vals[-1]:
            lat_vals = lat_vals[::-1]
            pop = pop[::-1, :]
    lat_mask = (lat_vals >= 20) & (lat_vals <= 57)
    lon_mask = (lon_vals >= -135) & (lon_vals <= -53)
    return lat_vals[lat_mask], lon_vals[lon_mask], pop[np.ix_(lat_mask, lon_mask)]

def aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon, radius_deg=0.06):
    pop_coords = np.column_stack([np.repeat(lat_vals, len(lon_vals)),
                                  np.tile(lon_vals, len(lat_vals))])
    tree = cKDTree(pop_coords)
    pop_agg = np.zeros(cmaq_lat.shape)
    nlon = len(lon_vals)
    for i in range(cmaq_lat.shape[0]):
        for j in range(cmaq_lat.shape[1]):
            idx = tree.query_ball_point([cmaq_lat[i, j], cmaq_lon[i, j]], r=radius_deg)
            if idx:
                lat_idx = [k // nlon for k in idx]
                lon_idx = [k % nlon for k in idx]
                pop_agg[i, j] = np.sum(pop_data[lat_idx, lon_idx])
    return pop_agg

def make_whitened_colormap(hex_color: str):
    """White → base color colormap."""
    return LinearSegmentedColormap.from_list('custom', ['#FFFFFF', hex_color])

def plot_rx_burn_donut_grid(
    csv_template=PERMIT_TEMPLATE,
    years=YEARS,
    states=("FL", "GA", "SC"),
    base_color="#764F51",   # single color for all states 
    figsize=(12, 5),
    wedge_width=0.4,
    out_png=OUT_DONUT,
):
    """
    Donut charts of monthly burned area (sum over years) for each state.
    Uses the SAME white→base color colormap for all states,
    no subplot titles, and one shared colorbar with label.
    """

    # Load & combine data
    dfs = []
    for yr in years:
        df = pd.read_csv(csv_template.format(yr), parse_dates=['DATE'])
        df['YEAR'] = yr
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Month (1–12)
    data['month'] = data['DATE'].dt.month

    # Aggregate: sum ACRES by STATE × month (pooled across years)
    monthly_acres = (
        data.groupby(['STATE', 'month'])['ACRES']
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=range(1, 13))
    )

    # Global normalization
    vmin = float(monthly_acres.values.min()) if monthly_acres.size else 0.0
    vmax = float(monthly_acres.values.max()) if monthly_acres.size else 1.0
    if vmin == vmax:
        vmin, vmax = 0.0, vmax if vmax > 0 else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # White → base color colormap (same for all states)
    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', base_color])

    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Transparent figure
    fig, axes = plt.subplots(
        nrows=1, ncols=len(states), figsize=figsize,
        subplot_kw=dict(aspect='equal'), dpi=600
    )
    if len(states) == 1:
        axes = np.array([axes])
    fig.patch.set_alpha(0)
    fig.patch.set_facecolor('none')
    for ax in axes:
        ax.set_facecolor('none')

    # Donuts
    for j, st in enumerate(states):
        ax = axes[j]
        values = monthly_acres.loc[st].values if st in monthly_acres.index else np.zeros(12)
        colors = [cmap(norm(v)) for v in values]
    
        ax.pie(
            np.ones(12), labels=month_labels, colors=colors,
            startangle=90, counterclock=False, radius=0.58,
            wedgeprops=dict(width=wedge_width),
            textprops={'fontsize': 9.5, 'fontweight': 'bold'}
        )
    
        # Add state name in the center
        ax.text(0, 0, st,
                ha='center', va='center',
                fontsize=16, fontweight='bold')

    # Shared colorbar (white→base_color)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.01, pad=0.12, aspect=40
    )
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["Low", "High"], fontsize=18)

    # Title above the colorbar, bold
    cbar.set_label(
        "Acres burned in 2017–2019",
        fontsize=18,
        fontweight='bold',
        labelpad=8,           # space between bar and label
        loc='center'
    )
    cbar.ax.xaxis.set_label_position('top')

    # Bold tick labels
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontweight('bold')

    fig.savefig(out_png, bbox_inches='tight', dpi=600,
                transparent=True, facecolor='none', edgecolor='none')
    plt.close(fig)

# ------------------------- Load core spatial data -------------------------
gdf_states_all = gpd.read_file(STATES_SHP)
gdf_SE         = gdf_states_all[gdf_states_all["STUSPS"].isin(SE_ST_ABBR)]
gdf_FL         = gdf_states_all[gdf_states_all["STUSPS"] == "FL"]
gdf_GA         = gdf_states_all[gdf_states_all["STUSPS"] == "GA"]
gdf_SC         = gdf_states_all[gdf_states_all["STUSPS"] == "SC"]

cmaq_lon, cmaq_lat = load_cmaq_grid(metcro2d_filename)
nrows, ncols = cmaq_lat.shape
tree = grid_kdtree(cmaq_lat, cmaq_lon)

mask_se = build_state_mask(cmaq_lon, cmaq_lat, gdf_SE)
mask_fl = build_state_mask(cmaq_lon, cmaq_lat, gdf_FL)
mask_ga = build_state_mask(cmaq_lon, cmaq_lat, gdf_GA)
mask_sc = build_state_mask(cmaq_lon, cmaq_lat, gdf_SC)

# ------------------------- Figure 1a (transparent) -------------------------
def make_fig_1a():
    permits_df = load_year_concat(PERMIT_TEMPLATE, YEARS, parse_dates_col="DATE")
    permits_pct_grid = percent_grid_from_points(
        df=permits_df, lat_col="LATITUDE", lon_col="LONGITUDE",
        acres_col="ACRES", years=YEARS, tree=tree, nrows=nrows, ncols=ncols
    )
    
    colors_a = ['#e6e4e6', '#eed5bb', '#f6c690', '#ee9e6b', '#d55e4d', '#bd1e2f']
    bounds_a = [0, 5, 10, 25, 50, 75, 100]
    cmap_a = LinearSegmentedColormap.from_list('burned_percent', colors_a)
    cmap_a.set_bad((1, 1, 1, 0))  # Transparent for truly masked areas
    norm_a = BoundaryNorm(bounds_a, cmap_a.N)
    
    fig, ax = plt.subplots(
        1, figsize=(7, 5.25), dpi=600,
        subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)},
        facecolor='none'
    )
    
    # Full transparency setup
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    if hasattr(ax, "background_patch"):
        ax.background_patch.set_visible(False)
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)
    
    ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
    ax.axis('off')
    
    # Step 1: Plot gray background for SE region only
    gray_background = np.where(mask_se, 1, np.nan)  # 1 where SE, NaN elsewhere
    ax.pcolormesh(cmaq_lon, cmaq_lat, gray_background,
                  cmap=LinearSegmentedColormap.from_list('gray', ['#e6e4e6', '#e6e4e6']),
                  transform=ccrs.PlateCarree(), zorder=1, alpha=0.7)
    
    # Step 2: Overlay actual data (mask where no data, not where outside SE)
    permits_data_only = np.where((mask_se) & (~np.isnan(permits_pct_grid)), 
                                permits_pct_grid, np.nan)
    
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, permits_data_only,
                       cmap=cmap_a, norm=norm_a, shading='auto',
                       transform=ccrs.PlateCarree(), zorder=2)
    
    # State outlines
    ax.add_geometries(gdf_SE.to_crs(epsg=4326).geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='k', linewidth=1.2, zorder=3)
    
    plt.tight_layout()
    plt.savefig(OUT_1A, bbox_inches='tight', dpi=600, transparent=True,
                facecolor='none', edgecolor='none')
    plt.close(fig)

# ------------------------- Figure 3a (transparent) -------------------------
def make_fig_3a():
    mean_fire_pm25 = multi_year_annual_mean_pm25(YEARS)
    mean_fire_pm25_masked = np.where(mask_se, mean_fire_pm25, np.nan)
    # cmap_pm = cc.cm.CET_L11.copy()
    cmap_pm = cmap_new.copy()
    cmap_pm.set_bad((1,1,1,0))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=600,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
    ax.set_extent([-91, -75, 24, 37])
    ax.axis('off')

    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.2, zorder=3)

    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_fire_pm25_masked, vmin=0, vmax=3.5,
                       cmap=cmap_pm, transform=ccrs.PlateCarree(), zorder=2)

    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    plt.savefig(OUT_3A, bbox_inches='tight', dpi=600, transparent=True)
    plt.close(fig)

# ------------------------- Figure 3b (transparent) -------------------------
def make_fig_3b():
    mean_smoke_day_counts = multi_year_avg_smoke_days(YEARS, SMOKE_THRESHOLD)
    mean_smoke_day_counts_masked = np.where(mask_se, mean_smoke_day_counts, np.nan)

    colors = [
        '#B6B3D6', '#CFCCE3', '#D5D3DE', '#D5D1D1',
        '#F6DFD6', '#F8B2A2', '#F1837A', '#E9687A'
    ]
    bounds_sd = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    cmap_sd = LinearSegmentedColormap.from_list('smoke_days', colors)
    cmap_sd.set_bad((1,1,1,0))
    norm_sd = BoundaryNorm(bounds_sd, cmap_sd.N)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=600,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
    ax.set_extent([-91, -75, 24, 37])
    ax.axis('off')

    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.2, zorder=3)

    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_smoke_day_counts_masked,
                       cmap=cmap_sd, norm=norm_sd, transform=ccrs.PlateCarree(), zorder=2)

    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    plt.savefig(OUT_3B, bbox_inches='tight', dpi=600, transparent=True)
    plt.close(fig)

# ------------------------- Figure 4a (transparent) -------------------------
def make_fig_4a():    
    # --- Aggregate population to CMAQ ---
    lat_vals, lon_vals, pop_data = load_population(POP_TIF)
    pop_agg = aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon)
    pop_masked = np.ma.masked_where((~mask_se) | (pop_agg <= 0), pop_agg)

    # --- Colormap / norm with transparent bad/under ---
    cmap = cc.cm.CET_L17.copy()
    cmap.set_bad((1, 1, 1, 0))
    cmap.set_under((1, 1, 1, 0))
    norm = LogNorm(vmin=1, vmax=100000)

    # Create figure with explicit transparent background
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(3, 2), dpi=600, facecolor='none')
    fig.patch.set_alpha(0)
    
    # Add subplot with projection
    ax = fig.add_subplot(1, 1, 1, 
                         projection=ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33),
                         facecolor='none')
    
    # Remove all backgrounds
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0)
    ax.set_facecolor('none')
    
    # Cartopy backgrounds
    if hasattr(ax, "background_patch"):
        ax.background_patch.set_visible(False)
    if hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(False)

    ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
    ax.axis('off')

    # Plot data
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, pop_masked,
                       cmap=cmap, norm=norm, shading='auto',
                       transform=ccrs.PlateCarree(), zorder=2)

    # State outlines
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(),
                      facecolor='none', edgecolor='k', linewidth=1.2, zorder=3)

    # Save with maximum transparency settings
    fig.savefig(OUT_4A, bbox_inches='tight', pad_inches=0, dpi=600,
                transparent=True, facecolor='none', edgecolor='none')
    plt.close(fig)
    plt.ion()  # Turn interactive mode back on
    
# ------------------------- Figure Donut (transparent) -------------------------
def make_fig_donut():
    """Wrapper so your __main__ section matches other figures."""
    plot_rx_burn_donut_grid(
        csv_template=PERMIT_TEMPLATE,
        years=YEARS,
        states=("FL", "GA", "SC"),
        base_color="#D4D925",
        figsize=(7, 5),
        wedge_width=0.2,
        out_png=OUT_DONUT,
    )
    
# ------------------------- Run all -------------------------
if __name__ == "__main__":
    print("Replotting with transparent backgrounds ...")
    # make_fig_1a()
    # make_fig_3a()
    # make_fig_3b()
    # make_fig_4a()
    make_fig_donut()

    print("Saved:")
    # print(" -", OUT_1A)
    # print(" -", OUT_3A)
    # print(" -", OUT_3B)
    # print(" -", OUT_4A)
    print(" -", OUT_DONUT)
