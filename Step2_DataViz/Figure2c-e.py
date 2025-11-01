# -*- coding: utf-8 -*-
# ==========================
# Figure 2c-e
# Author: Jingting HUANG
# ==========================

import os
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from datetime import datetime
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs
import colorcet as cc

# -----------------------------
# Environment (PROJ paths)
# -----------------------------
os.environ["PROJ_LIB"] = "/home/jh94030/.conda/envs/myenv/share/proj"
os.environ["PROJ_DATA"] = "/home/jh94030/.conda/envs/myenv/share/proj"

# -----------------------------
# Paths
# -----------------------------
DIR_SCRIPTS = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
DIR_FIG = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'
SHAPEFILE_STATES = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'
POP_TIF = "/work/chflab/jthuang/breadcrumbs/ciesen_nasa/ciesen_nasa_gpw_v4_population_count_2020.tif"
MET_DIR = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
METCRO2D_FILE = f"{MET_DIR}/METCRO2D_20170101.nc"

CMAQ_ALL_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift/combined/hr2dy'
CMAQ_RMV_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'

# -----------------------------
# Utilities
# -----------------------------
# Append util module path and import CMAQGrid2D
sys.path.append(os.path.join(DIR_SCRIPTS, 'step3_BurnDataSelection'))
from util import CMAQGrid2D  # noqa: E402

def load_cmaq_grid(metcro2d_filename):
    info = CMAQGrid2D(metcro2d_filename)
    return info['Lon'], info['Lat']

def load_states():
    gdf_states = gpd.read_file(SHAPEFILE_STATES)
    gdf_se = gdf_states[gdf_states['STUSPS'].isin(['FL', 'GA', 'SC'])]
    return gdf_states, gdf_se, \
           gdf_states[gdf_states['STUSPS'].isin(['FL'])], \
           gdf_states[gdf_states['STUSPS'].isin(['GA'])], \
           gdf_states[gdf_states['STUSPS'].isin(['SC'])]

def create_mask(lon_grid, lat_grid, gdf):
    mask = np.zeros(lon_grid.shape, dtype=bool)
    # Iterate grid cells; mark True if point is within or touches boundary
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            if any(geom.contains(point) or geom.touches(point) for geom in gdf.geometry):
                mask[i, j] = True
    return mask

def load_population(pop_tif):
    with rasterio.open(pop_tif) as src:
        pop_data = src.read(1).astype(np.float64)
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            pop_data = np.where(pop_data == nodata, 0.0, pop_data)

        width, height = src.width, src.height
        x_coords = np.arange(width) + 0.5
        y_coords = np.arange(height) + 0.5

        lon_vals = transform.c + x_coords * transform.a
        lat_vals = transform.f + y_coords * transform.e

        # Ensure increasing latitude order (south -> north)
        if lat_vals[0] > lat_vals[-1]:
            lat_vals = lat_vals[::-1]
            pop_data = pop_data[::-1, :]

    # Subset bounds used in your original
    lat_mask = (lat_vals >= 20) & (lat_vals <= 57)
    lon_mask = (lon_vals >= -135) & (lon_vals <= -53)
    pop_data = pop_data[np.ix_(lat_mask, lon_mask)]
    lat_vals = lat_vals[lat_mask]
    lon_vals = lon_vals[lon_mask]

    return lat_vals, lon_vals, pop_data

def aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon, radius_deg=0.06):
    # Manual aggregation by summing fine pixels within ~6 km of CMAQ cell centers.
    print("Manual aggregation by finding which high-res pixels belong to each CMAQ cell")

    pop_coords = np.column_stack([np.repeat(lat_vals, len(lon_vals)),
                                  np.tile(lon_vals, len(lat_vals))])
    tree = cKDTree(pop_coords)

    pop_agg = np.zeros(cmaq_lat.shape)
    nlon = len(lon_vals)

    for i in range(cmaq_lat.shape[0]):
        for j in range(cmaq_lat.shape[1]):
            center_lat = cmaq_lat[i, j]
            center_lon = cmaq_lon[i, j]
            indices = tree.query_ball_point([center_lat, center_lon], r=radius_deg)
            if indices:
                lat_idx = [idx // nlon for idx in indices]
                lon_idx = [idx % nlon for idx in indices]
                pop_agg[i, j] = np.sum(pop_data[lat_idx, lon_idx])

    return pop_agg

def compute_exposure(years, months, masks, pop_agg):
    # Replicates your compute_exposure behavior with no changes to logic.
    results = {}
    for year in years:
        ds_all = xr.open_dataset(os.path.join(
            CMAQ_ALL_PATH,
            f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc'
        ))
        ds_rmv = xr.open_dataset(os.path.join(
            CMAQ_RMV_PATH,
            f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc'
        ))
        time_coord = ds_all['TFLAG'][:, 0, 0].values

        for idx, tflag in enumerate(time_coord):
            date = datetime.strptime(str(tflag), '%Y%j')
            if date.month not in months:
                continue

            fire_pm25 = ds_all['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values \
                        - ds_rmv['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values
            fire_pm25[fire_pm25 < 0] = 0

            for state, mask in masks.items():
                weighted_pm25 = np.where(mask, fire_pm25 * pop_agg, np.nan)
                smoke_days = np.where(mask & (fire_pm25 > 3.5), pop_agg, 0)

                results.setdefault(state, {}).setdefault(year, {'weighted_pm25': [], 'smoke_days': []})
                results[state][year]['weighted_pm25'].append(np.nansum(weighted_pm25))
                results[state][year]['smoke_days'].append(np.nansum(smoke_days))

    summary = {}
    for state in results:
        summary[state] = {'multi_year': {}, 'annual': {}}
        total_pop = np.nansum(np.where(masks[state], pop_agg, 0))

        for year in years:
            # Preserve your original days_count logic
            if set(months).issubset(set(range(1, 5))):
                days_count = 119
            elif set(months).issubset(set(range(5, 13))):
                days_count = 246
            else:
                days_count = len(results[state][year]['weighted_pm25'])

            annual_avg_pm25 = sum(results[state][year]['weighted_pm25']) / (total_pop * days_count)
            annual_person_days = sum(results[state][year]['smoke_days'])
            summary[state]['annual'][year] = {'avg_pm25': annual_avg_pm25, 'person_days': annual_person_days}

        summary[state]['multi_year']['avg_pm25'] = np.mean(
            [summary[state]['annual'][yr]['avg_pm25'] for yr in years]
        )
        summary[state]['multi_year']['person_days'] = np.mean(
            [summary[state]['annual'][yr]['person_days'] for yr in years]
        )

    return summary

def plot_lollipop_by_category_h(data_dict, title, xlabel, colors, states, save_path, scale_factor=1.0):
    # Category order: All Seasons (first), High-burn (second), Low-burn (third)
    categories = ['all', 'high', 'low']
    category_labels = [r'$\bf{All\ Seasons}$' + '\n(Year-round)',
                       r'$\bf{High}$' + r'$\bf{-burn}$' + '\n(Jan-Apr)',
                       r'$\bf{Low}$' + r'$\bf{-burn}$' + '\n(May-Dec)']

    y_idx = np.arange(len(categories))

    # Peak per category (max across states)
    cat_peaks = {cat: max(data_dict[cat][st] / scale_factor for st in states) for cat in categories}
    x_lim = max(cat_peaks.values()) * 1.15

    # Fonts
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(figsize=(6.8, 3))

    # Gray stick per category (0 -> category max)
    for i, cat in enumerate(categories):
        ax.hlines(y=y_idx[i], xmin=0, xmax=cat_peaks[cat], color='#AAAAAA', linewidth=4, zorder=0)

    # Three colored circles on each stick (one per state)
    for st in states:
        x_vals = [data_dict[cat][st] / scale_factor for cat in categories]
        ax.scatter(x_vals, y_idx, s=250, label=st, color=colors[st],
                   edgecolors='#F9F9F9', linewidths=1.2, zorder=3)

    # Axes & labels
    ax.set_xlim(0, x_lim)
    ax.set_yticks(y_idx)
    ax.set_yticklabels(category_labels, fontsize=13)
    ax.set_xlabel('')  # hide default

    # Exactly two decimals (preserve your intent)
    if scale_factor == 1.0:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

    xmax = ax.get_xlim()[1] * 1.2
    ax.text(xmax, -0.06, xlabel, transform=ax.get_xaxis_transform(),
            ha='right', va='top', clip_on=False, fontsize=12)

    ax.set_title(title, fontsize=20, fontweight='bold', pad=40,
                 bbox=dict(boxstyle='round4', fc='white', ec='black', lw=1.2))

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.invert_yaxis()

    # Clean look
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Legend on right
    ax.legend(loc='center', bbox_to_anchor=(1.1, 0.5), ncol=1, frameon=True,
              fontsize=15, labelspacing=2, fancybox=False, edgecolor='k', handletextpad=0.2)

    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Ensure output dir
    os.makedirs(DIR_FIG, exist_ok=True)

    # Load core spatial data once
    cmaq_lon, cmaq_lat = load_cmaq_grid(METCRO2D_FILE)
    gdf_states, gdf_SE, gdf_FL, gdf_GA, gdf_SC = load_states()

    # Masks used in Fig.4a and for exposure
    mask_se = create_mask(cmaq_lon, cmaq_lat, gdf_SE)
    mask_fl = create_mask(cmaq_lon, cmaq_lat, gdf_FL)
    mask_ga = create_mask(cmaq_lon, cmaq_lat, gdf_GA)
    mask_sc = create_mask(cmaq_lon, cmaq_lat, gdf_SC)
    masks = {'FL': mask_fl, 'GA': mask_ga, 'SC': mask_sc}

    # ----- Shared population aggregation -----
    lat_vals, lon_vals, pop_data = load_population(POP_TIF)
    pop_aggregated = aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon)

    # ===================
    # Figure 4a
    # ===================
    # Plot settings
    below_threshold_color = '#e6e4e6'
    # Mask SE region
    pop_masked_adjusted = np.ma.masked_where(~mask_se, pop_aggregated)

    # Colormap & normalization
    cmap = cc.cm.CET_L17
    cmap.set_under(below_threshold_color)
    norm = LogNorm(vmin=1, vmax=100000)

    # Font family (as in your plotting code)
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(1, figsize=(6, 5), dpi=600,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})

    # Make the map axes fill the whole canvas (removes outer frame padding)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # or: ax.set_position([0,0,1,1]

    # Give the whole map a gray background so areas outside your state polygons aren’t white
    ax.set_facecolor('#e6e4e6')                           # same as below_threshold_color

    ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
    ax.axis('off')

    # Background color below threshold
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor=below_threshold_color,
                      edgecolor='k', linewidth=0, zorder=1)

    # Plot data
    im = ax.pcolormesh(cmaq_lon, cmaq_lat, pop_masked_adjusted, cmap=cmap, norm=norm, alpha=0.75,
                       shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # State outlines
    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.2, zorder=3)

    # Colorbar with log-spaced ticks
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.4, pad=0.05, extend='both')
    log_ticks = [1, 10, 100, 1000, 10000, 100000]
    cbar.set_ticks(log_ticks)
    cbar.set_label('Per-pixel population count\n(at a 12-km level)', fontsize=7)
    for l in cbar.ax.xaxis.get_ticklabels():
        l.set_fontsize(7)

    # Donut chart (state population share)
    total_fl = 21538187
    total_ga = 10711908
    total_sc = 5118425
    state_pop = [total_ga, total_fl, total_sc]
    state_labels = ['GA', 'FL', 'SC']
    state_colors = ['#B983FF', '#CFF800', '#FCD307']

    inset_ax = fig.add_axes([0.29, 0.3, 0.2, 0.2])
    wedges, texts = inset_ax.pie(
        state_pop,
        labels=state_labels,
        colors=state_colors,
        wedgeprops=dict(width=0.35, edgecolor='w'),
        textprops=dict(fontsize=7),
        startangle=90
    )
    inset_ax.axis('equal')
    inset_ax.text(0, 0, 'Total\n\nPopulation\n\nShare',
                  ha='center', va='center',
                  fontsize=6, fontweight='bold')

    out_png_4a = os.path.join(DIR_FIG, 'reaggregated_pop_12km.png')
    fig.savefig(out_png_4a, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig)

    # Print mass conservation info to match your original prints
    total_orig = np.sum(pop_data)
    total_regrid = np.sum(pop_aggregated)
    print(f"Original total: {total_orig:,.0f}")
    print(f"Regridded total: {total_regrid:,.0f}")
    print(f"Conservation ratio: {total_regrid/total_orig:.4f}")

    # ===================
    # Figure 4b / 4c
    # ===================
    # Exposure metrics (multi-year, 2017–2019)
    years = [2017, 2018, 2019]
    results_all_year = compute_exposure(years, range(1, 13), masks, pop_aggregated)
    results_high = compute_exposure(years, range(1, 5), masks, pop_aggregated)
    results_low = compute_exposure(years, range(5, 13), masks, pop_aggregated)

    # Prep for plotting
    states = ['FL', 'GA', 'SC']
    colors = {'FL': '#CFF800', 'GA': '#B983FF', 'SC': '#FCD307'}

    all_season = {st: results_all_year[st]['multi_year']['avg_pm25'] for st in states}
    high_burn = {st: results_high[st]['multi_year']['avg_pm25'] for st in states}
    low_burn = {st: results_low[st]['multi_year']['avg_pm25'] for st in states}

    all_season_pd = {st: results_all_year[st]['multi_year']['person_days'] for st in states}
    high_burn_pd = {st: results_high[st]['multi_year']['person_days'] for st in states}
    low_burn_pd = {st: results_low[st]['multi_year']['person_days'] for st in states}

    # Figure 4b: PM2.5
    out_png_4b = os.path.join(DIR_FIG, 'lollipop_h_PM25.png')
    plot_lollipop_by_category_h(
        data_dict={'all': all_season, 'high': high_burn, 'low': low_burn},
        title='$E_{\\mathrm{annual}}^{\\mathrm{Rx}}$',
        xlabel='(in µg/m³)',
        colors=colors,
        states=states,
        save_path=out_png_4b,
        scale_factor=1.0
    )

    # Figure 4c: Person-days (millions)
    out_png_4c = os.path.join(DIR_FIG, 'lollipop_h_PersonDays.png')
    plot_lollipop_by_category_h(
        data_dict={'all': all_season_pd, 'high': high_burn_pd, 'low': low_burn_pd},
        title='$PD_{\\mathrm{annual}}^{\\mathrm{Rx}}$',
        xlabel='(in millions)',
        colors=colors,
        states=states,
        save_path=out_png_4c,
        scale_factor=1e6
    )

    print("Saved:")
    print(" -", out_png_4a)
    print(" -", out_png_4b)
    print(" -", out_png_4c)