# -*- coding: utf-8 -*-
###############################################################################
# regrid_rxburn_acres.py 
# author: Jingting HUANG
# purpose: Regrid prescribed fire burned area over FL, GA, SC to CMAQ grid
# figures: Figure S5 with federal lands
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
PERMIT_TEMPLATE  = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/SE_Combined_Permit_lf_3states_rx_{}.csv"

# Shapefiles
STATES_SHP   = "/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
COUNTY_SHP   = "/work/chflab/jthuang/breadcrumbs/us_demo_county_2020/cb_2020_us_county_500k.shp"
FEDLANDS_SHP = "/work/chflab/jthuang/breadcrumbs/federal_lands/USA_Federal_Lands.shp"

SE_ST_ABBR = ["FL", "GA", "SC"]
SE_ST_FIPS = ["12", "13", "45"]

# MCIP grid file
met_filedir        = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename  = f"{met_filedir}/METCRO2D_20170101.nc"

# ------------------------- Fonts -------------------------
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

plt.rcParams["font.family"]     = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

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
gdf_federal     = gpd.read_file(FEDLANDS_SHP)

# Match CRS for fedlands selection and plotting
target_ll = gdf_SE[["STUSPS", "geometry"]].to_crs(gdf_federal.crs).copy()
target_ll = target_ll.dissolve(by="STUSPS", as_index=False)
# Keep ONLY polygons strictly within each state boundary
fed_fl_ga_sc = (gpd.sjoin(gdf_federal, target_ll, how="inner", predicate="intersects")
                  .rename(columns={"STUSPS": "STATE"})
                  .drop(columns=["index_right"]))

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

# Federal lands (plot in their native CRS if lon/lat; here enforce lon/lat)
ax.add_geometries(fed_fl_ga_sc.to_crs(epsg=4326).geometry,
                  crs=ccrs.PlateCarree(), facecolor="none",
                  edgecolor="#4F8A8B", linewidth=0.8, zorder=2)

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
plt.savefig(os.path.join(dir_python_local, "spatial_ba_permits_federal_lands.png"), bbox_inches='tight', dpi=600)