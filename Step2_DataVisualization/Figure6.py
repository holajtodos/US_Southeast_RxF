# Generates Figure 6a, 6b, 6c
# Author: Jingting HUANG (cleanup/merge assist)

import os
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import geopandas as gpd

from shapely.geometry import Point
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import colorcet as cc
import cartopy.crs as ccrs

from scipy.stats import gaussian_kde

# ------------------------- Paths & Globals -------------------------
BASE_DIR = '/home/jh94030/scripts/python/postdoc_project/rxfire'
FIG_DIR  = f'{BASE_DIR}/figure'
DATA_DIR = f'{BASE_DIR}/data'

# analysis helper path
DIR_PYTHON_SCRIPTS = f'{BASE_DIR}/analysis'
sys.path.append(os.path.join(DIR_PYTHON_SCRIPTS, 'step3_BurnDataSelection'))

from util import CMAQGrid2D  # expects your util.py as before

# Fonts
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
plt.rcParams['font.family'] = 'Arial'

YEARS = [2017, 2018, 2019]

# Shapefile
SHAPEFILE_PATH = '/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp'

# CMAQ met file (for grid)
MET_FILEDIR = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
METCRO2D = f"{MET_FILEDIR}/METCRO2D_20170101.nc"

# CMAQ daily directories
CMAQ_ALL_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift/combined/hr2dy'
CMAQ_RMV_PATH = '/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy'

# High-fire AQS site list
HIGH_FIRE_SITE_DATA = {
    "SiteId": [
        120730012, 130210007, 130210012, 130590002,  130950007, 131530001,
        131850003, 132150001, 132150008, 132150011, 133030001, 450370001
    ],
    "Latitude": [
        30.4397, 32.7775, 32.8053, 33.9181, 31.5776, 32.6056,
        30.8486, 32.4842, 32.5213, 32.4297, 32.9674, 33.7400
    ],
    "Longitude": [
        -84.3464, -83.6410, -83.5435, -83.3444, -84.0998, -83.5978,
        -83.2933, -84.9789, -84.9446, -84.9316, -82.8069, -81.8536
    ]
}
HIGH_FIRE_SITES = pd.DataFrame(HIGH_FIRE_SITE_DATA)

# ------------------------- Helpers -------------------------
def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)

def load_states():
    gdf_states = gpd.read_file(SHAPEFILE_PATH)
    # ensure geographic CRS for PlateCarree
    if gdf_states.crs is not None and str(gdf_states.crs).lower() not in ('epsg:4326', 'epsg:4269'):
        gdf_states = gdf_states.to_crs(epsg=4326)
    gdf_SE = gdf_states[gdf_states['STUSPS'].isin(['FL', 'GA', 'SC'])]
    gdf_FL = gdf_states[gdf_states['STUSPS'].isin(['FL'])]
    gdf_GA = gdf_states[gdf_states['STUSPS'].isin(['GA'])]
    gdf_SC = gdf_states[gdf_states['STUSPS'].isin(['SC'])]
    return gdf_SE, gdf_FL, gdf_GA, gdf_SC

def create_mask(lon_grid, lat_grid, gdf):
    mask = np.zeros(lon_grid.shape, dtype=bool)
    geoms = list(gdf.geometry)
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            for poly in geoms:
                if poly.contains(point) or poly.touches(point):
                    mask[i, j] = True
                    break
    return mask

def remove_nan_values_default(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]

def metrics_dict_default():
    from scipy.stats import spearmanr
    def n_pairs(p, o): return np.sum(np.isfinite(p) & np.isfinite(o))
    def mb(p, o): return float(np.mean(p - o))
    def rmse(p, o): return float(np.sqrt(np.mean((p - o)**2)))
    def rs(p, o):
        if len(p) < 2:
            return np.nan
        r = spearmanr(p, o, nan_policy='omit').correlation
        return float(r) if np.isfinite(r) else np.nan
    return {
        "# Pairs": {"func": n_pairs},
        "MB": {"func": mb},
        "RMSE": {"func": rmse},
        "Spearman R": {"func": rs},
    }

# Allow overrides if defined elsewhere
remove_nan_values = globals().get('remove_nan_values', remove_nan_values_default)
metrics_dict = globals().get('metrics_dict', metrics_dict_default())

# ------------------------- Figure 6a -------------------------
def generate_figure6a(gdf_SE, gdf_FL, gdf_GA, gdf_SC):
    # Load CMAQ grid
    cmaq_info = CMAQGrid2D(METCRO2D)
    cmaq_lon, cmaq_lat = cmaq_info['Lon'], cmaq_info['Lat']

    # Build masks
    mask     = create_mask(cmaq_lon, cmaq_lat, gdf_SE)
    mask_fl  = create_mask(cmaq_lon, cmaq_lat, gdf_FL)
    mask_ga  = create_mask(cmaq_lon, cmaq_lat, gdf_GA)
    mask_sc  = create_mask(cmaq_lon, cmaq_lat, gdf_SC)

    # Collect daily percent for Jan–Apr
    all_daily_percent = []
    for year in YEARS:
        print(f"[6a] Processing high-burn season of {year}")
        f_all = os.path.join(CMAQ_ALL_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')
        f_rmv = os.path.join(CMAQ_RMV_PATH, f'dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_{year}01-{year}12.nc')

        ds_all = xr.open_dataset(f_all)
        ds_rmv = xr.open_dataset(f_rmv)

        time_coord = ds_all['TFLAG'][:, 0, 0].values
        for idx, tflag in enumerate(time_coord):
            date = datetime.strptime(str(int(tflag)), '%Y%j')
            if date.month > 4:
                continue  # only Jan–Apr

            pm25_all = ds_all['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values
            pm25_rmv = ds_rmv['PM25_TOT_AVG'].isel(TSTEP=idx, LAY=0).values
            fire_pm25 = pm25_all - pm25_rmv
            fire_pm25[fire_pm25 < 0] = 0

            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pct = 100 * fire_pm25 / pm25_all
                daily_pct = np.where(pm25_all == 0, 0, daily_pct)
                daily_pct = np.where(mask, daily_pct, np.nan)

            all_daily_percent.append(daily_pct)

    all_daily_percent_arr = np.stack(all_daily_percent)
    with np.errstate(invalid='ignore'):
        mean_percent = np.nanmean(all_daily_percent_arr, axis=0)
        mean_percent = np.where(mask, mean_percent, np.nan)

    # Quintiles for colorbar bounds
    valid_vals = mean_percent[~np.isnan(mean_percent)]
    if valid_vals.size == 0:
        raise ValueError("[6a] No valid values found for mean_percent within mask.")

    q1, q2, q3, q4 = np.percentile(valid_vals, [20, 40, 60, 80])
    vmin = float(np.min(valid_vals))
    vmax = float(np.max(valid_vals))
    bounds = [vmin, q1, q2, q3, q4, vmax]
    bounds = [round(b, 1) for b in bounds]
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i-1]:
            bounds[i] = np.nextafter(bounds[i-1], np.inf)

    colors = ['#4198AC', '#7BC0CD', '#DBCB92', '#ECB66C', '#ED8D5A']
    cmap = LinearSegmentedColormap.from_list('quintile_rxf_pm25', colors, N=len(colors))
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    # === Map (original Figure 6a) ===
    aqs_fp = os.path.join(DATA_DIR, 'collocated_mod_obs', f'aq_SE_{YEARS[0]}_RXF',
                          f'AQS_Daily_aq_SE_{YEARS[0]}_RXF_with_smoke_day.csv')
    df_result = pd.read_csv(aqs_fp)
    site_info = df_result[['SiteId', 'Latitude', 'Longitude']].drop_duplicates()
    lat = site_info['Latitude'].values
    lon = site_info['Longitude'].values

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300,
                           subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})
    ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
    ax.axis('off')

    ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(), facecolor='none',
                      edgecolor='k', linewidth=1.2, zorder=3)

    im = ax.pcolormesh(cmaq_lon, cmaq_lat, mean_percent,
                       cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=2)

    ax.scatter(lon, lat, facecolor='none', edgecolor='#393E46', linewidth=0.5, s=25,
               marker='^', transform=ccrs.PlateCarree(), zorder=3)

    ax.scatter(HIGH_FIRE_SITES['Longitude'], HIGH_FIRE_SITES['Latitude'], facecolor='w',
               edgecolor='#393E46', linewidth=0.5, s=25, marker='^',
               transform=ccrs.PlateCarree(), zorder=3)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.5, pad=0.05, boundaries=bounds, ticks=bounds)
    cbar.set_ticklabels([f"{b:.1f}" for b in bounds])
    cbar.set_label('Multi-year mean daily Rx fire contribution (%)', fontsize=7, fontweight='bold')
    for l in cbar.ax.xaxis.get_ticklabels():
        l.set_fontsize(7)

    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='$\mathrm{PM}_{2.5}$ sites',
               markerfacecolor='w', markeredgecolor='#393E46', markersize=6, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='small', frameon=True)

    out_map = os.path.join(FIG_DIR, "pm25_fire_percent_2017_2019_highburn.png")
    plt.savefig(out_map, bbox_inches='tight', dpi=600)
    plt.close(fig)
    print(f"[6a] Saved: {out_map}")

    # === NEW: KDE joyplot by state using normalized quintiles ===
    mean_percent_fl = np.where(mask_fl, mean_percent, np.nan).flatten()
    mean_percent_ga = np.where(mask_ga, mean_percent, np.nan).flatten()
    mean_percent_sc = np.where(mask_sc, mean_percent, np.nan).flatten()

    def get_clean_data(data, threshold=0.01):
        return data[~np.isnan(data) & (data > threshold)]

    data_fl = get_clean_data(mean_percent_fl)
    data_ga = get_clean_data(mean_percent_ga)
    data_sc = get_clean_data(mean_percent_sc)

    df_long = pd.concat([
        pd.DataFrame({'State': 'FL', 'PM25_%': data_fl}),
        pd.DataFrame({'State': 'GA', 'PM25_%': data_ga}),
        pd.DataFrame({'State': 'SC', 'PM25_%': data_sc}),
    ], ignore_index=True)

    state_list = ['FL', 'GA', 'SC']
    normed_kde_results = {}
    normed_x_grid = np.linspace(0, 5, 500)  # uniform axis for 5 quintiles

    for state in state_list:
        values = df_long[df_long['State'] == state]['PM25_%'].values
        if values.size == 0:
            # Avoid KDE error on empty
            normed_kde_results[state] = np.zeros_like(normed_x_grid)
            continue
        # Map original values onto [0..5] using colorbar bounds
        x_normed = np.interp(values, bounds, range(len(bounds)))
        # Guard for single-point degenerate cases
        if np.allclose(x_normed, x_normed[0]):
            kde_vals = np.zeros_like(normed_x_grid)
            kde_vals[np.argmin(np.abs(normed_x_grid - x_normed[0]))] = 1.0
            normed_kde_results[state] = kde_vals
        else:
            kde = gaussian_kde(x_normed, bw_method=0.3)
            normed_kde_results[state] = kde(normed_x_grid)

    fig, axarr = plt.subplots(len(state_list), 1, figsize=(6, 4), dpi=600, sharex=True)
    for i, (ax, state) in enumerate(zip(axarr, state_list)):
        y_vals = normed_kde_results[state]
        offset = i * 1.0
        y_offset = y_vals + offset

        for j in range(len(normed_x_grid) - 1):
            x_seg = normed_x_grid[j:j+2]
            y_seg = y_offset[j:j+2]
            # Color segments by original value scale via bounds/norm
            orig_val = np.interp(np.mean(x_seg), range(len(bounds)), bounds)
            color = cmap(norm(orig_val))
            ax.fill_between(x_seg, y_seg, offset, color=color, linewidth=0)

        ax.plot(normed_x_grid, y_offset, color='black', linewidth=2)

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.set_yticks([])
        ax.set_ylabel(state, fontsize=14, fontweight='semibold', rotation=0, labelpad=16, va='center')
        ax.set_xlim(0, 5)
        ax.set_ylim(offset, offset + 0.8)

        if state == state_list[-1]:
            ax.tick_params(axis='x', labelsize=10, direction='out', bottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', bottom=False)

    # Normalized x-axis ticks correspond to the 6 bounds points
    xticks = np.arange(6)
    xtick_labels = [f"{b:.1f}" for b in bounds]
    axarr[-1].set_xticks(xticks)
    axarr[-1].set_xticklabels(xtick_labels, fontsize=10)
    axarr[-1].tick_params(axis='x', bottom=True, direction='out')
    axarr[-1].set_xlabel("Multi-year mean daily Rx fire contribution (%)", fontsize=12)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.18, hspace=0.1)
    out_joy = os.path.join(FIG_DIR, "joyplot_normalized_quintiles_rx_fire_percent_highburn.png")
    plt.savefig(out_joy, dpi=600)
    plt.close(fig)
    print(f"[6a] Saved: {out_joy}")

# ------------------------- Figure 6b -------------------------
def generate_figure6b(gdf_SE):
    selected_metrics = ["# Pairs", "MB", "RMSE", "Spearman R"]
    colorbar_limits = {
        "# Pairs": (0, 100),
        "MB": (-15, 15),
        "RMSE": (0, 25),
        "Spearman R": (0, 1)
    }
    # colorcet → safe fallback for the 4th
    colormaps = [cc.cm.CET_L12, cc.cm.CET_CBD1, cc.cm.CET_L17, cc.cm.dimgray_r]

    # Load and filter per-year CSVs
    all_years_df_season = []
    for year in YEARS:
        file_path = os.path.join(
            DATA_DIR, 'collocated_mod_obs', f'aq_SE_{year}_RXF',
            f'AQS_Daily_aq_SE_{year}_RXF_with_smoke_day.csv'
        )
        df_result = pd.read_csv(file_path)
        df_result['date'] = pd.to_datetime(dict(year=df_result['SYYYY'], month=df_result['SMM'], day=df_result['SDD']))
        df_result['season'] = df_result['date'].dt.month.map(lambda m: 'High-burn' if m <= 4 else 'Low-burn')
        df_season = df_result[
            (df_result['season'].str.lower() == 'high-burn') &
            (df_result['smoke_missing_date'] == False) &
            (df_result['smokePM'] > 1e-3)
        ].copy()
        df_season = df_season.merge(HIGH_FIRE_SITES, on=["SiteId", "Latitude", "Longitude"], how="inner")
        df_season['PM_TOT_mod_rxf'] = df_season['PM_TOT_mod_rxf'].clip(lower=1e-6)
        df_season['smokePM'] = df_season['smokePM'].clip(lower=1e-6)
        df_season['Year'] = year
        all_years_df_season.append(df_season)

    df_all = pd.concat(all_years_df_season, ignore_index=True)

    # Compute site metrics
    site_metrics = []
    for site, group in df_all.groupby(['SiteId', 'Latitude', 'Longitude']):
        site_id, lat, lon = site
        pred_valid, obs_valid = remove_nan_values(group['PM_TOT_mod_rxf'].values, group['smokePM'].values)
        if len(pred_valid) < 10:
            continue
        if np.all(pred_valid == pred_valid[0]) or np.all(obs_valid == obs_valid[0]):
            continue
        metrics = {'SiteId': site_id, 'Latitude': lat, 'Longitude': lon}
        metrics['State'] = group['State'].iloc[0] if 'State' in group.columns else 'Unknown'
        for metric_name, metric_info in metrics_dict.items():
            try:
                metrics[metric_name] = metric_info['func'](pred_valid, obs_valid)
            except Exception:
                metrics[metric_name] = np.nan
        site_metrics.append(metrics)

    metrics_df = pd.DataFrame(site_metrics)
    criteria = lambda row: not (row['Spearman R'] >= 0.4)
    metrics_df['highlight'] = metrics_df.apply(criteria, axis=1)
    df_highlight = df_all.merge(metrics_df[metrics_df['highlight']],
                                on=['SiteId', 'Latitude', 'Longitude'], how='inner')

    # Save tables
    metrics_out = os.path.join(DATA_DIR, 'collocated_mod_obs', 'multi_year_site_metrics_highburn_only.csv')
    hl_out = os.path.join(DATA_DIR, 'collocated_mod_obs', 'multi_year_highlighted_points_highburn_only.csv')
    metrics_df.to_csv(metrics_out, index=False)
    df_highlight.to_csv(hl_out, index=False)
    print(f"[6b] Saved: {metrics_out}")
    print(f"[6b] Saved: {hl_out}")

    # Plot maps per metric
    for k, metric in enumerate(selected_metrics):
        fig, ax = plt.subplots(1, figsize=(7, 6), dpi=300,
                               subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)})

        ax.set_extent([-91, -75, 24, 37], crs=ccrs.PlateCarree())
        ax.axis('off')

        ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(),
                          facecolor='#e6e4e6', edgecolor='k', alpha=0.5, linewidth=0, zorder=2)
        ax.add_geometries(gdf_SE.geometry, crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='k', linewidth=1.2, zorder=3)

        limits = colorbar_limits.get(metric)
        vmin, vmax = limits if limits else (None, None)

        in_criteria = metrics_df[~metrics_df['highlight']]
        out_criteria = metrics_df[metrics_df['highlight']]

        sc1 = ax.scatter(in_criteria['Longitude'], in_criteria['Latitude'], c=in_criteria[metric], cmap=colormaps[k],
                         edgecolor='#C70039', linewidth=0.5, s=60, marker='^', transform=ccrs.PlateCarree(), alpha=0.9,
                         vmin=vmin, vmax=vmax, zorder=3)

        ax.scatter(out_criteria['Longitude'], out_criteria['Latitude'], c=out_criteria[metric], cmap=colormaps[k],
                   edgecolor='#393E46', linewidth=1.5, s=25, marker='^', transform=ccrs.PlateCarree(), alpha=0.9,
                   vmin=vmin, vmax=vmax, zorder=4)

        if metric == '# Pairs':
            cbar = plt.colorbar(sc1, ax=ax, orientation='horizontal', extend='max', shrink=0.6, pad=0.01)
        elif metric == 'MB':
            cbar = plt.colorbar(sc1, ax=ax, orientation='horizontal', extend='both', shrink=0.6, pad=0.01)
        elif metric == 'Spearman R':
            cbar = plt.colorbar(sc1, ax=ax, orientation='horizontal', extend='min', shrink=0.6, pad=0.01)
        else:
            cbar = plt.colorbar(sc1, ax=ax, orientation='horizontal', shrink=0.4, pad=0.01)

        cbar.set_label(metric if metric != 'Spearman R' else 'r', size=15)
        cbar.ax.tick_params(labelsize=12)

        legend_elements = [
            Line2D([0], [0], marker='^', color='w', label='r > 0.4',
                   markerfacecolor='#e6e4e6', markeredgecolor='#C70039', markersize=9, linestyle='None'),
            Line2D([0], [0], marker='^', color='w', label='r ≤ 0.4',
                   markerfacecolor='#e6e4e6', markeredgecolor='#393E46', markersize=6, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize='small', frameon=True)

        plt.title(f"Multi-year Average ({YEARS[0]}–{YEARS[-1]}) - {metric}", fontsize=11)
        plt.tight_layout()
        out_png = os.path.join(FIG_DIR, f'multi-year_-{metric}.png')
        plt.savefig(out_png, bbox_inches='tight', dpi=600)
        plt.close(fig)
        print(f"[6b] Saved: {out_png}")

# ------------------------- Figure 6c -------------------------
def generate_figure6c():
    from sklearn.linear_model import LinearRegression

    years = YEARS
    all_years_subset = None

    # ---------------- Data prep ----------------
    for year in years:
        file_path = os.path.join(
            DATA_DIR, 'collocated_mod_obs', f'aq_SE_{year}_RXF',
            f'AQS_Daily_aq_SE_{year}_RXF_with_smoke_day.csv'
        )
        df_result = pd.read_csv(file_path)

        subset = df_result[(df_result['smoke_missing_date'] == False) &
                           (df_result['smokePM'] > 1e-3)].copy()
        subset['PM_TOT_mod_rxf'] = subset['PM_TOT_mod_rxf'].clip(lower=1e-3)
        subset['smokePM'] = subset['smokePM'].clip(lower=1e-3)

        subset['log_PM_TOT_mod_rxf'] = np.log10(subset['PM_TOT_mod_rxf'])
        subset['log_smokePM'] = np.log10(subset['smokePM'])

        if 'season' not in subset.columns:
            subset['date'] = pd.to_datetime(dict(year=subset['SYYYY'],
                                                 month=subset['SMM'],
                                                 day=subset['SDD']))
            subset['season'] = subset['date'].dt.month.map(
                lambda m: 'High-burn' if m <= 4 else 'Low-burn'
            )

        subset = subset[subset['season'].str.lower() == 'high-burn']
        subset = subset.merge(HIGH_FIRE_SITES,
                              on=["SiteId", "Latitude", "Longitude"],
                              how="inner")

        all_years_subset = subset.copy() if all_years_subset is None \
            else pd.concat([all_years_subset, subset], ignore_index=True)

    if all_years_subset is None or all_years_subset.empty:
        raise ValueError("[6c] No data available for the specified filters.")

    # ---------------- Regression ----------------
    X = all_years_subset['log_PM_TOT_mod_rxf'].values.reshape(-1, 1)
    y = all_years_subset['log_smokePM'].values
    reg = LinearRegression().fit(X, y)
    slope, intercept = reg.coef_[0], reg.intercept_

    # ---------------- Plot ----------------
    g0 = sns.jointplot(
        data=all_years_subset, x="log_PM_TOT_mod_rxf", y="log_smokePM",
        kind="reg", color='k',
        height=5, ratio=8, space=0,
        marginal_kws=dict(binwidth=0.1,
                          binrange=(-2, np.log10(60)),
                          linewidth=0, fill=False),
        marginal_ticks=False
    )
    g0.ax_joint.cla()

    # Heatmap + colorbar
    fig = plt.gcf()
    cbar_ax = fig.add_axes([0.95, 0.27, 0.02, 0.2])
    sns.histplot(
        data=all_years_subset, x="log_PM_TOT_mod_rxf", y="log_smokePM",
        stat='count', cmap=cc.cm.CET_L19, cbar=True, cbar_ax=cbar_ax,
        cbar_kws=dict(shrink=.85, label='Count'),
        ax=g0.ax_joint, binwidth=0.1
    )
    cbar = g0.ax_joint.collections[-1].colorbar
    cbar.set_label('Count', fontsize=13, fontweight='semibold')

    g0.set_axis_labels(
        'CMAQ-predicted Rx fire ' + r'$\mathbf{PM}_{2.5}$',
        'Observed Smoke ' + r'$\mathbf{PM}_{2.5}$',
        fontsize=16, labelpad=10
    )
    g0.ax_joint.xaxis.label.set_fontweight('semibold')
    g0.ax_joint.yaxis.label.set_fontweight('semibold')

    # Regression + 1:1 line
    x_vals = np.linspace(-2, np.log10(60), 100)
    y_vals = slope * x_vals + intercept
    g0.ax_joint.plot(x_vals, y_vals, color='k', lw=4, linestyle='-')
    g0.ax_joint.text(
        0.05, 0.62, f'y = {slope:.2f}x + {intercept:.2f}',
        transform=g0.ax_joint.transAxes, fontsize=15, fontweight='semibold',
        color='k', ha='left'
    )
    g0.ax_joint.plot(x_vals, x_vals, linestyle='--', color='black', alpha=0.7)

    g0.ax_joint.set_xlim(-2, np.log10(60))
    g0.ax_joint.set_ylim(-2, np.log10(60))

    # Percentile lines + text
    x_percentiles = np.percentile(all_years_subset['PM_TOT_mod_rxf'], [25, 50, 75])
    log_x_percentiles = np.percentile(all_years_subset['log_PM_TOT_mod_rxf'], [25, 50, 75])
    y_percentiles = np.percentile(all_years_subset['smokePM'], [25, 50, 75])
    log_y_percentiles = np.percentile(all_years_subset['log_smokePM'], [25, 50, 75])

    for i, percentile in enumerate(log_x_percentiles):
        yloc = g0.ax_marg_x.get_ylim()[1] * (0.9 if i == 1 else 1.2)
        g0.ax_marg_x.axvline(x=percentile, ymin=0,
                             ymax=(0.8 if i == 1 else 1.1),
                             color='k', linestyle='-', alpha=0.7)
        g0.ax_marg_x.text(percentile, yloc, f'{x_percentiles[i]:.3f}',
                          fontsize=12, color='k',
                          ha='center', va='bottom')

    for i, percentile in enumerate(log_y_percentiles):
        g0.ax_marg_y.axhline(y=percentile, color='k',
                             linestyle='-', alpha=0.7)
        g0.ax_marg_y.text(g0.ax_marg_y.get_xlim()[1] * 1.05, percentile,
                          f'{y_percentiles[i]:.3f}', fontsize=12,
                          color='k', ha='left', va='center')

    # Ticks
    original_ticks = [1e-2, 1, 5, 10, 60]
    log_ticks = np.log10(original_ticks)
    tick_labels = [str(v) for v in original_ticks]

    for ax in [g0.ax_joint, g0.ax_marg_x]:
        ax.set_xticks(log_ticks)
        ax.set_xticklabels(tick_labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(15)

    for ax in [g0.ax_joint, g0.ax_marg_y]:
        ax.set_yticks(log_ticks)
        ax.set_yticklabels(tick_labels)
        for label in ax.get_yticklabels():
            label.set_fontsize(15)

    # Save
    out_png = os.path.join(FIG_DIR, 'Rx_fire_pm25_comp.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=600)
    plt.close(plt.gcf())
    print(f"[6c] Saved: {out_png}")

# ------------------------- main -------------------------
def main():
    ensure_dirs()
    print(f"cwd is {os.getcwd()}")
    os.chdir(FIG_DIR)
    print(f"now in {os.getcwd()}")

    gdf_SE, gdf_FL, gdf_GA, gdf_SC = load_states()

    generate_figure6a(gdf_SE, gdf_FL, gdf_GA, gdf_SC)
    generate_figure6b(gdf_SE)
    generate_figure6c()

if __name__ == "__main__":
    main()