import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

# Change directory 
os.getcwd()
print ('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'

# Append the location of our function directory
dir_python_scripts = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
sys.path.append(os.path.join(dir_python_scripts,'step3_BurnDataSelection'))

from util import CMAQGrid2D

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print ('cwd is %s ' % (os.getcwd()))

# --------------------- Inputs & Paths ---------------------
CMAQ_ALL_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift/combined/hr2dy'
CMAQ_RMV_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'

# Load Southeastern states shapefile
shapefile_path = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'
gdf_states = gpd.read_file(shapefile_path)
gdf_SE = gdf_states[gdf_states['STUSPS'].isin(['FL', 'GA', 'SC'])]
gdf_FL = gdf_states[gdf_states['STUSPS'].isin(['FL'])]
gdf_GA = gdf_states[gdf_states['STUSPS'].isin(['GA'])]
gdf_SC = gdf_states[gdf_states['STUSPS'].isin(['SC'])]

# Load CMAQ Grid Information (for lon/lat)
met_filedir = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
metcro2d_filename = f"{met_filedir}/METCRO2D_20170101.nc"
cmaq_info = CMAQGrid2D(metcro2d_filename)
cmaq_lon, cmaq_lat = cmaq_info['Lon'], cmaq_info['Lat']   # 2D arrays [y, x]

# --------------------- Masks (kept simple; can be vectorized later) ---------------------
def create_mask(lon_grid, lat_grid, gdf_poly):
    mask = np.zeros(lon_grid.shape, dtype=bool)
    # small speed-ups via bounds check first
    gdf_poly = gdf_poly.to_crs("EPSG:4326")
    bounds = gdf_poly.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds
    in_bbox = (lon_grid >= minx) & (lon_grid <= maxx) & (lat_grid >= miny) & (lat_grid <= maxy)
    poly_union = gdf_poly.unary_union
    yi, xi = np.where(in_bbox)
    for y, x in zip(yi, xi):
        pt = Point(lon_grid[y, x], lat_grid[y, x])
        if poly_union.contains(pt) or poly_union.touches(pt):
            mask[y, x] = True
    return mask

mask_all = create_mask(cmaq_lon, cmaq_lat, gdf_SE)
mask_fl  = create_mask(cmaq_lon, cmaq_lat, gdf_FL)
mask_ga  = create_mask(cmaq_lon, cmaq_lat, gdf_GA)
mask_sc  = create_mask(cmaq_lon, cmaq_lat, gdf_SC)

masks = {
    "Grid-cell average over full region": mask_all,
    "Grid-cell average in FL": mask_fl,
    "Grid-cell average in GA": mask_ga,
    "Grid-cell average in SC": mask_sc,
}

# --------------------- Helpers ---------------------
def open_year_ds(year):
    ds_all = xr.open_dataset(os.path.join(
        CMAQ_ALL_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc'))
    ds_rmv = xr.open_dataset(os.path.join(
        CMAQ_RMV_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc'))
    return ds_all, ds_rmv

def iter_season_days(ds_all, ds_rmv, months):
    # TFLAG: yyyyddd (year + day-of-year)
    tflags = ds_all['TFLAG'][:, 0, 0].values
    for idx, tflag in enumerate(tflags):
        dt = datetime.strptime(str(int(tflag)), '%Y%j')
        if dt.month in months:
            pm_all = ds_all['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values
            pm_rmv = ds_rmv['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values
            fire = pm_all - pm_rmv
            # physical guard
            fire = np.where(fire < 0, 0.0, fire)
            yield fire, pm_all

def seasonal_mean_fields(years, months):
    """Return seasonal multi-day mean fields (2D) of fire and total across ALL selected days in 'years'."""
    fire_days = []
    total_days = []
    for yr in years:
        ds_all, ds_rmv = open_year_ds(yr)
        for fire, tot in iter_season_days(ds_all, ds_rmv, months):
            fire_days.append(fire)
            total_days.append(tot)
        ds_all.close(); ds_rmv.close()
    fire_days = np.stack(fire_days, axis=0)  # [days, y, x]
    total_days = np.stack(total_days, axis=0)
    mean_fire_2d  = np.nanmean(fire_days, axis=0)
    mean_total_2d = np.nanmean(total_days, axis=0)
    return mean_fire_2d, mean_total_2d

def seasonal_mean_fields_by_year(year, months):
    """Return seasonal multi-day mean fields (2D) for a single year."""
    ds_all, ds_rmv = open_year_ds(year)
    fire_days, total_days = [], []
    for fire, tot in iter_season_days(ds_all, ds_rmv, months):
        fire_days.append(fire)
        total_days.append(tot)
    ds_all.close(); ds_rmv.close()
    fire_days = np.stack(fire_days, axis=0)
    total_days = np.stack(total_days, axis=0)
    return np.nanmean(fire_days, axis=0), np.nanmean(total_days, axis=0)

def stats_over_mask(mean_fire_2d, mean_total_2d, mask):
    """Compute Mean±SD for fire, % of total (mean±SD), and GM/GSD for fire over masked cells."""
    fire_vals  = mean_fire_2d[mask].astype(np.float64)
    total_vals = mean_total_2d[mask].astype(np.float64)

    # Arithmetic mean ± SD of fire
    mean_fire = np.nanmean(fire_vals)
    sd_fire   = np.nanstd(fire_vals, ddof=1)

    # % contribution: ratio per grid then spatial stats
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.where(total_vals > 0, (fire_vals / total_vals) * 100.0, np.nan)
    mean_pct = np.nanmean(pct)
    sd_pct   = np.nanstd(pct, ddof=1)

    # Geometric mean & GSD of fire (ignore zeros & negatives)
    pos = fire_vals > 0
    if np.any(pos):
        ln = np.log(fire_vals[pos])
        gm  = float(np.exp(np.nanmean(ln)))
        gsd = float(np.exp(np.nanstd(ln, ddof=1)))
    else:
        gm, gsd = np.nan, np.nan

    return {
        "mean_fire": float(mean_fire),
        "sd_fire": float(sd_fire),
        "mean_pct": float(mean_pct),
        "sd_pct": float(sd_pct),
        "gm": gm,
        "gsd": gsd,
    }

def format_pair(s):
    """Format helper: (Mean ± SD (% of total PM2.5)), and 'GM, GSD'."""
    return f"{s['mean_fire']:.4f} +/- {s['sd_fire']:.4f} ({s['mean_pct']:.1f}%)", f"{s['gm']:.4f}, {s['gsd']:.3f}"
    
# --------------------- Build Table S7 ---------------------
rows = []
def add_row(season_label, year_label, mf2d, mt2d):
    # Build columns for each region
    row = {"Burn season": season_label, "Year": year_label}
    for col_name, m in masks.items():
        s = stats_over_mask(mf2d, mt2d, m)
        mean_sd_str, gm_gsd_str = format_pair(s)
        row[f"{col_name} - Mean +/- SD (% of total PM2.5)"] = mean_sd_str
        row[f"{col_name} - GM, GSD"] = gm_gsd_str
    rows.append(row)

years = [2017, 2018, 2019]
season_defs = {
    "All seasons":       list(range(1, 13)),
    "High-burn":         [1, 2, 3, 4],
    "Low-burn":          [5, 6, 7, 8, 9, 10, 11, 12],
}

# Per-season, per-year rows
for season_name, months in season_defs.items():
    for yr in years:
        mf2d, mt2d = seasonal_mean_fields_by_year(yr, months)
        add_row(season_name, str(yr), mf2d, mt2d)
    # 2017–2019 multi-year for that season
    mf2d_all, mt2d_all = seasonal_mean_fields(years, months)
    add_row(season_name, "2017-2019", mf2d_all, mt2d_all)

# Create DataFrame with your exact header buckets
df = pd.DataFrame(rows)

# Optional: order columns nicely
base_cols = ["Burn season", "Year"]
order = base_cols + sum([
    [f"{name} - Mean +/- SD (% of total PM2.5)", f"{name} - GM, GSD"]
    for name in [
        "Grid-cell average over full region",
        "Grid-cell average in FL",
        "Grid-cell average in GA",
        "Grid-cell average in SC",
    ]
], [])
df = df[order]

# Save to CSV for SI
out_csv = "Table_S7_permit_based_rx_PM25_stats.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")