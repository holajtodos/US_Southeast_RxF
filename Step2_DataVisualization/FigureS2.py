# ag_ba_maps.py
# -*- coding: utf-8 -*-
###############################################################################
# Figures S2a/S2b/S2c
# author: Jingting HUANG
# purpose: Mean annual reported burned area (%) maps for ag fires:
#          - Permits (S2a)
#          - NEI (S2b)
#          - Difference (Permits - NEI) (S2c)
###############################################################################
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
import shapely.prepared as shp_prep
import shapely.ops as shp_ops
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from pyproj import datadir

# ------------------------- USER PATHS / CONSTANTS -------------------------
DIR_FIG = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'
DIR_SCRIPTS = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
PROJ_DIR = "/home/jh94030/.conda/envs/uscensus_ej/share/proj"

PERMIT_TMPL = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/SE_Combined_Permit_lf_3states_agr_{}.csv"
NEI_TMPL    = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/NEI_rxf_inv/SE_Combined_NEI_ag_3states_{}.csv"

MET_DIR = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
MET_FILE = f"{MET_DIR}/METCRO2D_20170101.nc"

SHP_STATE = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'
# (Optional) County shapefile kept for future use
# SHP_COUNTY = '/work/chflab/jthuang/breadcrumbs/us_demo_county_2020/cb_2020_us_county_500k.shp'

YEARS = [2017, 2018, 2019]
CELL_AREA_KM2 = 12 * 12
CELL_AREA_ACRES = CELL_AREA_KM2 * 247.105
SE_STATES_ABBR = ['FL', 'GA', 'SC']
SE_STATEFPS = ['12', '13', '45']  # FL, GA, SC
CITIES = {
    'Atlanta': {'lat': 33.7490, 'lon': -84.3880},
    'Orlando': {'lat': 28.5383, 'lon': -81.3792},
    'Tallahassee': {'lat': 30.4383, 'lon': -84.2807},
    'Columbia': {'lat': 34.0007, 'lon': -81.0348},
    'Jacksonville': {'lat': 30.3322, 'lon': -81.6557},
    'Savannah': {'lat': 32.0809, 'lon': -81.0912},
    'Pensacola': {'lat': 30.4213, 'lon': -87.2169},
    'Tampa': {'lat': 27.9506, 'lon': -82.4572},
    'Miami': {'lat': 25.7617, 'lon': -80.1918},
    'Columbus': {'lat': 32.4600, 'lon': -84.9877},
    'Albany': {'lat': 31.5785, 'lon': -84.1557},
    'Charleston': {'lat': 32.7765, 'lon': -79.9311}
}

# Plot defaults you used
COLORS_ABS = ['#e6e4e6', '#eed5bb', '#f6c690', '#ee9e6b', '#d55e4d', '#bd1e2f']
BOUNDS_ABS = [0, 5, 10, 25, 50, 75, 100]
COLORS_DIFF = ['#746170', '#99879C', '#c2b7c6', '#C5E3F6', '#fee0b6', '#fdb863', '#e08214', '#b35806']
BOUNDS_DIFF = [-45, -20, -5, -1, 0, 1, 5, 20, 45]

# ------------------------- ENV / IMPORTS SETUP -------------------------
os.environ["PROJ_LIB"] = PROJ_DIR
os.environ["PROJ_DATA"] = PROJ_DIR
datadir.set_data_dir(PROJ_DIR)

sys.path.append(os.path.join(DIR_SCRIPTS, 'step3_BurnDataSelection'))
from util import CMAQGrid2D

# Fonts
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# ------------------------- HELPERS -------------------------
def load_cmaq_grid():
    cmaq_info = CMAQGrid2D(MET_FILE)
    lon = np.asarray(cmaq_info['Lon'], dtype=float)
    lat = np.asarray(cmaq_info['Lat'], dtype=float)
    nrows, ncols = lat.shape

    # Extent (preserve your offsets)
    bot_left_lat  = lat[0][0] + 2
    bot_left_lon  = lon[0][0] + 32
    top_right_lat = lat[-1][-1] - 15
    top_right_lon = lon[-1][-1] - 22
    extent = [bot_left_lon, top_right_lon, bot_left_lat, top_right_lat]

    return lon, lat, nrows, ncols, extent

def build_se_mask(cmaq_lon, cmaq_lat, se_gdf):
    # Prepared unary union; covers = contains | touches
    se_union = shp_ops.unary_union(se_gdf.geometry)
    se_prep  = shp_prep.prep(se_union)
    mask = np.zeros(cmaq_lon.shape, dtype=bool)
    for i in range(cmaq_lon.shape[0]):
        for j in range(cmaq_lon.shape[1]):
            if se_prep.covers(Point(cmaq_lon[i, j], cmaq_lat[i, j])):
                mask[i, j] = True
    return mask

def process_data(file_template, years, lat_col, lon_col, acres_col, cmaq_lon, cmaq_lat):
    """Return masked array of mean annual burned percent for given dataset."""
    dfs = []
    for yr in years:
        df = pd.read_csv(file_template.format(yr), parse_dates=['DATE'])
        df["YEAR"] = yr
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Drop invalid coords for KDTree safety
    data = data.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    # KDTree map (lat, lon) consistent with how grid points are stacked
    nrows, ncols = cmaq_lat.shape
    grid_pts = np.column_stack((cmaq_lat.ravel(), cmaq_lon.ravel()))
    tree = cKDTree(grid_pts)
    pts = np.column_stack((data[lat_col].to_numpy(), data[lon_col].to_numpy()))
    _, idx_flat = tree.query(pts, k=1)
    data["ROW"] = idx_flat // ncols
    data["COL"] = idx_flat %  ncols

    grouped = (data.groupby(["YEAR", "ROW", "COL"])[acres_col]
                  .sum()
                  .unstack(level=0)
                  .fillna(0.0))
    annual_mean_acres = grouped.mean(axis=1).to_numpy()
    annual_mean_percent = (annual_mean_acres / CELL_AREA_ACRES) * 100.0

    burned_percent_grid = np.full((nrows, ncols), np.nan, dtype=float)
    for (row, col), percent in zip(grouped.index, annual_mean_percent):
        burned_percent_grid[row, col] = percent

    return burned_percent_grid

def plot_common(ax, gdf_SE, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.axis('off')
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='#e6e4e6',
                      edgecolor='k', linewidth=0, zorder=2)
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.2, zorder=3)

    for (txt, xy) in [("FL", (-84.5, 28.5)), ("GA", (-87.0, 32.5)), ("SC", (-78.5, 34.2))]:
        t = ax.text(xy[0], xy[1], txt, color='k', fontweight="bold", fontsize=16, transform=ccrs.Geodetic())
        t.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

    for city, coord in CITIES.items():
        ax.scatter(coord['lon'], coord['lat'], marker='o', facecolor='none', edgecolor='k', s=10,
                   transform=ccrs.PlateCarree(), zorder=3)
        ax.text(coord['lon'] + 0.15, coord['lat'] - 0.1, city,
                fontsize=7, fontweight='bold', color='black',
                transform=ccrs.PlateCarree(), zorder=3,
                path_effects=[path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

def render_abs_map(out_png, grid_masked, cmaq_lon, cmaq_lat, gdf_SE, extent):
    cmap = LinearSegmentedColormap.from_list('burned_percent', COLORS_ABS)
    norm = BoundaryNorm(BOUNDS_ABS, cmap.N)

    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
    plot_common(ax, gdf_SE, extent)

    im = ax.pcolormesh(cmaq_lon, cmaq_lat, grid_masked.astype(float),
                       cmap=cmap, norm=norm, shading='auto',
                       transform=ccrs.PlateCarree(), zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor='w', edgecolor='k', linewidth=0.5, zorder=2)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.35, pad=0.05)
    cbar.set_label('Mean annual reported burned area (%)', fontsize=9)
    for l in cbar.ax.xaxis.get_ticklabels():
        l.set_fontsize(7)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, out_png), bbox_inches='tight', dpi=600)
    plt.close(fig)

def render_diff_map(out_png, diff_masked, cmaq_lon, cmaq_lat, gdf_SE, extent):
    cmap = LinearSegmentedColormap.from_list('burned_percent', COLORS_DIFF)
    norm = BoundaryNorm(BOUNDS_DIFF, cmap.N)

    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
    plot_common(ax, gdf_SE, extent)

    im = ax.pcolormesh(cmaq_lon, cmaq_lat, diff_masked.astype(float),
                       cmap=cmap, norm=norm, shading='auto',
                       transform=ccrs.PlateCarree(), zorder=2)
    
    ax.add_feature(cfeature.LAKES, facecolor='w', edgecolor='k', linewidth=0.5, zorder=2)


    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.35, pad=0.05)
    cbar.set_ticks(BOUNDS_DIFF)
    cbar.set_ticklabels(BOUNDS_DIFF)
    cbar.set_label('Difference between permits and NEI (%)', fontsize=9)
    for l in cbar.ax.xaxis.get_ticklabels():
        l.set_fontsize(7)

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIG, out_png), bbox_inches='tight', dpi=600)
    plt.close(fig)

# ------------------------- MAIN FLOW -------------------------
def main(mode):
    # Working dir
    os.makedirs(DIR_FIG, exist_ok=True)
    os.chdir(DIR_FIG)

    # Load grid and states
    cmaq_lon, cmaq_lat, nrows, ncols, extent = load_cmaq_grid()
    gdf_states = gpd.read_file(SHP_STATE)
    gdf_SE = gdf_states[gdf_states['STUSPS'].isin(SE_STATES_ABBR)].reset_index(drop=True)

    # Build SE mask once
    se_mask = build_se_mask(cmaq_lon, cmaq_lat, gdf_SE)

    # Prepare masked arrays depending on mode
    permit_masked = None
    nei_masked = None

    if mode in ('permit', 'all', 'diff'):
        permit_grid = process_data(PERMIT_TMPL, YEARS, 'LATITUDE', 'LONGITUDE', 'ACRES', cmaq_lon, cmaq_lat)
        permit_masked = np.ma.masked_where(~se_mask, permit_grid)

    if mode in ('nei', 'all', 'diff'):
        nei_grid = process_data(NEI_TMPL, YEARS, 'latitude', 'longitude', 'ACRESBURNED', cmaq_lon, cmaq_lat)
        nei_masked = np.ma.masked_where(~se_mask, nei_grid)

    # Render requested figures
    if mode in ('permit', 'all'):
        render_abs_map('spatial_ba_permits_agr.png', permit_masked, cmaq_lon, cmaq_lat, gdf_SE, extent)

    if mode in ('nei', 'all'):
        render_abs_map('spatial_ba_NEI_agr.png', nei_masked, cmaq_lon, cmaq_lat, gdf_SE, extent)

    if mode in ('diff', 'all'):
        # Ensure identical mask before subtraction
        union_mask = np.ma.getmaskarray(permit_masked) | np.ma.getmaskarray(nei_masked)
        permit_masked.mask = union_mask
        nei_masked.mask    = union_mask
        diff_masked = np.ma.array(permit_masked - nei_masked, mask=union_mask)

        render_diff_map('spatial_ba_diff_agr.png', diff_masked, cmaq_lon, cmaq_lat, gdf_SE, extent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate S2a/S2b/S2c ag burned-area maps.")
    parser.add_argument("--mode", choices=["permit", "nei", "diff", "all"], default="all",
                        help="Which figure(s) to produce.")
    args = parser.parse_args()
    main(args.mode)