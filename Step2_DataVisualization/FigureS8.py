###############################################################################
import os
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib import font_manager

import cartopy.crs as ccrs

# Set plot style and font
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
plt.rcParams['font.family'] = 'Arial'

# Change directory 
print('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'

# Append the location of our function directory
dir_python_scripts = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
sys.path.append(os.path.join(dir_python_scripts, 'step3_BurnDataSelection'))

from util import CMAQGrid2D

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print('cwd is %s ' % (os.getcwd()))
###############################################################################
# Load Southeastern states shapefile
shapefile_path = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'
gdf_states = gpd.read_file(shapefile_path)
gdf_SE = gdf_states[gdf_states['STUSPS'].isin(['FL', 'GA', 'SC'])]
gdf_FL = gdf_states[gdf_states['STUSPS'].isin(['FL'])]
gdf_GA = gdf_states[gdf_states['STUSPS'].isin(['GA'])]
gdf_SC = gdf_states[gdf_states['STUSPS'].isin(['SC'])]

# Load CMAQ grid information
met_filedir = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename = f"{met_filedir}/METCRO2D_20170101.nc"
cmaq_info = CMAQGrid2D(metcro2d_filename)
cmaq_lon, cmaq_lat = cmaq_info['Lon'], cmaq_info['Lat']

# Define mask for SE states
def create_mask(lon_grid, lat_grid, gdf):
    mask = np.zeros(lon_grid.shape, dtype=bool)
    # Note: This is simple but slow; consider vectorized point-in-polygon for large grids.
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            if any(state.geometry.contains(point) or state.geometry.touches(point) for state in gdf.itertuples()):
                mask[i, j] = True
    return mask

mask    = create_mask(cmaq_lon, cmaq_lat, gdf_SE)
mask_fl = create_mask(cmaq_lon, cmaq_lat, gdf_FL)
mask_ga = create_mask(cmaq_lon, cmaq_lat, gdf_GA)
mask_sc = create_mask(cmaq_lon, cmaq_lat, gdf_SC)

def best_text_color(hex_color: str) -> str:
    """Return 'black' or 'white' for best contrast on the given hex background."""
    r, g, b = mcolors.to_rgb(hex_color)  # 0–1 floats
    luminance = 0.299*r + 0.587*g + 0.114*b
    return 'black' if luminance > 0.5 else 'white'
    
# --- Directories ---
cmaq_all_path = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift_old/combined/hr2dy'
cmaq_rmv_path = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'

# ------------ AQI bins (μg/m3) and standard colors ------------
aqi_bins = {
    "Moderate": {"lo": 12.1, "hi": 35.4, "color": "#FFFF00"},  # yellow
    "Unhealthy for Sensitive Groups": {"lo": 35.5, "hi": 55.4, "color": "#FF7E00"},  # orange
    "Unhealthy": {"lo": 55.5, "hi": 150.4, "color": "#FF0000"},  # red
    "Very Unhealthy": {"lo": 150.5, "hi": 200.4, "color": "#8F3F97"},  # purple
    "Hazardous": {"lo": 200.5, "hi": 500.4, "color": "#7E0023"},  # maroon
}

# ------------ Colormap for "total number of smoke days" (shared across panels) ------------
colors = ['#B6B3D6', '#CFCCE3', '#D5D3DE', '#D5D1D1',
          '#F6DFD6', '#F8B2A2', '#F1837A', '#E9687A']
# Adjust bounds upward if counts exceed 80 anywhere across 3 years
bounds = [1, 10, 20, 30, 40, 50, 100, 140, 180]
cmap = LinearSegmentedColormap.from_list('smoke_days', colors)
cmap.set_under('white')  # show < bounds[0] as white
norm = BoundaryNorm(bounds, cmap.N)

# ------------ Accumulate (sum) high smoke days across 2017–2019 per AQI category ------------
cat_to_total = {}  # cat -> (ROW, COL) array of total counts across all years

for year in [2017, 2018, 2019]:
    print(f"Processing year: {year}")

    # Load CMAQ files
    f_all = os.path.join(cmaq_all_path, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
    f_rmv = os.path.join(cmaq_rmv_path, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')

    ds_all = xr.open_dataset(f_all)
    ds_rmv = xr.open_dataset(f_rmv)

    # Total PM2.5 and Rx-fire PM2.5 (surface layer)
    total_pm25 = ds_all['PM25_TOT_AVG'].isel(LAY=0)            # (TSTEP, ROW, COL)
    fire_pm25  = (ds_all['PM25_TOT_AVG'] - ds_rmv['PM25_TOT_AVG']).isel(LAY=0)
    fire_pm25  = fire_pm25.where(fire_pm25 > 0, 0)

    # Smoke-day flag (Rx fire > 3.5 μg/m3)
    smoke_mask = fire_pm25 > 3.5                                # (TSTEP, ROW, COL)

    # For each AQI bin, sum smoke days in this year and add to running total
    for cat, info in aqi_bins.items():
        lo, hi = info["lo"], info["hi"]
        aqi_mask = (total_pm25 >= lo) & (total_pm25 <= hi)      # inclusive
        combo = smoke_mask & aqi_mask

        counts_year = combo.sum(dim='TSTEP').values             # (ROW, COL)

        if cat not in cat_to_total:
            cat_to_total[cat] = counts_year.astype(np.float32)  # init
        else:
            cat_to_total[cat] += counts_year

    # Close datasets
    ds_all.close()
    ds_rmv.close()

# Optional spatial mask (boolean array same shape as a single map)
with np.errstate(invalid='ignore'):
    for cat in cat_to_total:
        cat_to_total[cat] = np.where(mask, cat_to_total[cat], np.nan)

# ------------ Plot: 5 panels, AQI-colored titles, shared colorbar ------------
fig, axes = plt.subplots(
    nrows=1, ncols=5, figsize=(7, 5.5), dpi=600,
    gridspec_kw={'left': 0.05, 'right': 0.99, 'bottom': 0.02, 'top': 0.98, 'wspace': 0.005},
    subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)}
)
axes = axes.ravel()

cats_order = list(aqi_bins.keys())

for i, cat in enumerate(cats_order):
    ax = axes[i]
    ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
    ax.axis('off')

    # State outlines
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.0, zorder=3)

    data = cat_to_total[cat]  # total counts across 2017–2019

    # --- compute min/max over valid (non-NaN) cells ---
    valid = np.isfinite(data)
    if np.any(valid):
        minv = int(np.nanmin(data))
        maxv = int(np.nanmax(data))
    else:
        minv = maxv = None  # nothing to annotate if everything is NaN

    im = ax.pcolormesh(
        cmaq_lon, cmaq_lat, data,
        cmap=cmap, norm=norm, shading='auto',
        transform=ccrs.PlateCarree(), zorder=2
    )

    bg = aqi_bins[cat]["color"]
    fg = best_text_color(bg)

    # --- annotate min/max in lower-right of each map ---
    if minv is not None:
        ax.text(
            0.98, 0.02,
            f"max: {maxv}\nmin: {minv}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
            zorder=4
        )

    ax.set_title(
        cat,
        fontsize=7,
        color=fg,
        pad=2,
        bbox=dict(
            facecolor=bg,
            edgecolor='none',
            boxstyle='round,pad=0.25'
        ),
    )
    
# Shared colorbar
cax = fig.add_axes([0.005, 0.4, 0.015, 0.2])  # [left, bottom, width, height] in figure coords
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.set_ticks(bounds)
cbar.set_ticklabels(bounds)
cbar.set_label('No. High Rx Fire Smoke Days', fontweight='bold', fontsize=6, rotation=90, labelpad=10)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.yaxis.set_ticks_position('right')
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_fontsize(7)
    
outpng = os.path.join(dir_python_local, "RxSmokeDays_by_AQI_2017_2019_TOTAL.png")
plt.savefig(outpng, bbox_inches='tight', dpi=600)
plt.close()
print(f"Saved: {outpng}")