# -*- coding: utf-8 -*-
"""
Figure 1: Rx fire burned area and PM2.5 emissions over FL, GA, and SC.
Author: Jingting HUANG

Purpose
-------
This script uses 2017–2019 permit-based and NEI Rx fire data, maps point
burned area to the 12-km CMAQ grid, summarizes annual totals by state and inventory,
and plots the spatial and lollipop panels used in Figure 1.

Outputs
-------
- f1a_spatial_ba_permits.png
- f1b_spatial_ba_diff.png
- f1c_annual_ba_diff.png
- f1d_<species>_<inventory>_spatial_mean_annual.png
- f1e_annual_pm25_diff.png
- summary CSV files for burned area and PM2.5 emissions
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union

import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager, ticker
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


# =============================================================================
# Settings
# =============================================================================

DIR_FIG = "/home/jh94030/scripts/python/postdoc_project/rxfire/figure"
DIR_ANALYSIS = "/home/jh94030/scripts/python/postdoc_project/rxfire/analysis"
sys.path.append(os.path.join(DIR_ANALYSIS, "step3_BurnDataSelection"))
from util import CMAQGrid2D

os.chdir(DIR_FIG)

YEARS = [2017, 2018, 2019]
SE_ST_ABBR = ["FL", "GA", "SC"]
BOTTOM_TO_TOP_STATES = ["SC", "GA", "FL"]
INVENTORIES = ["Permits", "NEI"]

CELL_AREA_KM2 = 12 * 12
CELL_AREA_ACRES = CELL_AREA_KM2 * 247.105

METCRO2D_FILE = (
    "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/"
    "mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc"
)
STATES_SHP = (
    "/work/chflab/jthuang/breadcrumbs/mapping_state/"
    "cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
)

NEI_BA_TPL = (
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/"
    "NEI_rxf_inv/SE_Combined_NEI_rx_3states_{}.csv"
)
PERMIT_BA_TPL = (
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data/"
    "SE_permit_data_2010-2020/update_criteria/"
    "SE_Combined_Permit_lf_3states_rx_{}.csv"
)

GRIDDED_EMIS_DIR = os.path.join(
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data", "gridded_CMAQ_12US1"
)
PERMIT_PM25_TPL = (
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data/"
    "SE_permit_data_2010-2020/output_emis/SE_{yr}_bluesky_rx_emis.csv"
)
NEI_PM25_TPL = (
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/"
    "NEI_rxf_inv/SE_Combined_NEI_rx_3states_{yr}.csv"
)

font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

BA_PROJ = ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)
EMIS_PROJ = ccrs.AlbersEqualArea(central_longitude=-84, central_latitude=30)

STATE_LABELS = [("FL", (-84.5, 28.5)), ("GA", (-87.0, 32.5)), ("SC", (-78.5, 34.2))]
CITIES = {
    "Atlanta": (33.7490, -84.3880), "Orlando": (28.5383, -81.3792),
    "Tallahassee": (30.4383, -84.2807), "Columbia": (34.0007, -81.0348),
    "Jacksonville": (30.3322, -81.6557), "Savannah": (32.0809, -81.0912),
    "Pensacola": (30.4213, -87.2169), "Tampa": (27.9506, -82.4572),
    "Miami": (25.7617, -80.1918), "Columbus": (32.4600, -84.9877),
    "Albany": (31.5785, -84.1557), "Charleston": (32.7765, -79.9311),
}

BA_CMAP = LinearSegmentedColormap.from_list(
    "burned_percent", ["#e6e4e6", "#eed5bb", "#f6c690", "#ee9e6b", "#d55e4d", "#bd1e2f"]
)
BA_BOUNDS = [0, 5, 10, 25, 50, 75, 100]
BA_NORM = BoundaryNorm(BA_BOUNDS, BA_CMAP.N)

DIFF_CMAP = LinearSegmentedColormap.from_list(
    "burned_percent_diff",
    ["#403E4B", "#746170", "#99879C", "#c2b7c6", "#fee0b6", "#e08214", "#7f3b08"],
)
DIFF_BOUNDS = [-72, -48, -24, -12, 0, 12, 24, 48]
DIFF_NORM = BoundaryNorm(DIFF_BOUNDS, DIFF_CMAP.N)

BA_YEAR_COLORS = {2017: "#C9E8ED", 2018: "#84B7D6", 2019: "#508CB6"}
PM25_YEAR_COLORS = {2017: "#DDEED9", 2018: "#B3D7AE", 2019: "#7DBA7F"}

SPECIES = ["PM25", "CO", "CO2", "NOx", "NH3", "SO2"]
VARMAP = {
    "PM25": {"Permits": "PM2.5", "NEI": "PM2_5"},
    "CO": {"Permits": "CO", "NEI": "CO"},
    "CO2": {"Permits": "CO2", "NEI": "CO2"},
    "NOx": {"Permits": "NOx", "NEI": "NOX"},
    "NH3": {"Permits": "NH3", "NEI": "NH3"},
    "SO2": {"Permits": "SO2", "NEI": "SO2"},
}
SPECIES_DISPLAY = {
    "PM25": r"PM$_{\mathregular{2.5}}$", "PM10": r"PM$_{\mathregular{10}}$",
    "CO": "CO", "CO2": r"CO$_{\mathregular{2}}$", "NOx": r"NO$_{\mathregular{x}}$",
    "NH3": r"NH$_{\mathregular{3}}$", "SO2": r"SO$_{\mathregular{2}}$",
    "OC": "OC", "BC": "BC",
}


# =============================================================================
# Shared helpers
# =============================================================================

def read_ba_grid(metcro_file):
    info = CMAQGrid2D(metcro_file)
    lon, lat = info["Lon"], info["Lat"]
    nrows, ncols = lat.shape
    extent = (lon[0, 0] + 32, lon[-1, -1] - 22, lat[0, 0] + 2, lat[-1, -1] - 15)
    return lon, lat, nrows, ncols, extent


def read_emis_grid(metcro_file):
    with nc.Dataset(metcro_file) as ds:
        p_alp, p_bet = float(ds.P_ALP), float(ds.P_BET)
        xcent, ycent = float(ds.XCENT), float(ds.YCENT)
        xorig, yorig = float(ds.XORIG), float(ds.YORIG)
        xcell, ycell = float(ds.XCELL), float(ds.YCELL)
        ncols, nrows = int(ds.NCOLS), int(ds.NROWS)

    proj4 = (
        f"+proj=lcc +a=6370000.0 +b=6370000.0 +lat_1={p_alp} +lat_2={p_bet} "
        f"+lat_0={ycent} +lon_0={xcent} +x_0=0 +y_0=0 +units=m +no_defs"
    )
    transformer = Transformer.from_proj(proj4, "epsg:4326", always_xy=True)

    x = np.linspace(xorig + xcell / 2, xorig + xcell / 2 + xcell * (ncols - 1), ncols)
    y = np.linspace(yorig + ycell / 2, yorig + ycell / 2 + ycell * (nrows - 1), nrows)
    x2d, y2d = np.meshgrid(x, y)
    lon, lat = transformer.transform(x2d, y2d)
    extent = (lon[0, 0] + 32, lon[-1, -1] - 22, lat[0, 0] + 2, lat[-1, -1] - 15)

    return lon, lat, xcell * ycell, extent


def load_states():
    states = gpd.read_file(STATES_SHP)
    return states[states["STUSPS"].isin(SE_ST_ABBR)].copy()


def load_annual_csvs(template, years, parse_dates_col="DATE"):
    frames = []
    for yr in years:
        df = pd.read_csv(template.format(yr), parse_dates=[parse_dates_col])
        df["YEAR"] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_state_mask(lon, lat, states_gdf):
    geom = unary_union(states_gdf.to_crs(epsg=4326).geometry.values)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    mask = np.fromiter(
        (geom.contains(Point(xy)) or geom.touches(Point(xy)) for xy in points),
        dtype=bool,
        count=points.shape[0],
    )
    return mask.reshape(lon.shape)


def format_sci_xaxis(ax, scilimits):
    ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.xaxis.get_major_formatter().set_useMathText(True)


def savefig(path):
    plt.savefig(path, bbox_inches="tight", dpi=600)
    plt.close()


def add_southeast_basemap(ax, states_gdf, extent, facecolor="#e6e4e6", city_fontsize=7):
    ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
    ax.axis("off")

    geom = states_gdf.to_crs(epsg=4326).geometry
    ax.add_geometries(geom, crs=ccrs.PlateCarree(), facecolor=facecolor,
                      edgecolor="k", linewidth=0, zorder=2)
    ax.add_geometries(geom, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor="k", linewidth=1.2, zorder=3)

    for label, xy in STATE_LABELS:
        text = ax.text(xy[0], xy[1], label, color="k", fontweight="bold",
                       fontsize=16, transform=ccrs.Geodetic())
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground="white"),
                               path_effects.Normal()])

    for name, (lat, lon) in CITIES.items():
        ax.scatter(lon, lat, marker="o", facecolor="none", edgecolor="k", s=10,
                   transform=ccrs.PlateCarree(), zorder=3)
        ax.text(lon + 0.15, lat - 0.1, name, fontsize=city_fontsize,
                fontweight="bold", color="black", transform=ccrs.PlateCarree(),
                zorder=3, path_effects=[path_effects.Stroke(linewidth=1, foreground="white"),
                                         path_effects.Normal()])

    ax.add_feature(cfeature.LAKES, facecolor="w", edgecolor="k", linewidth=0.5, zorder=2)


def make_y_positions(states=BOTTOM_TO_TOP_STATES):
    y_positions, ytick_values, ytick_labels = {}, [], []
    for i, state in enumerate(states):
        for inventory, offset in [("Permits", 0.3), ("NEI", -0.3)]:
            y_positions[(state, inventory)] = i * 2 + offset
            ytick_values.append(i * 2 + offset)
            ytick_labels.append(f"{state} ({inventory})")
    return y_positions, ytick_values, ytick_labels


# =============================================================================
# Burned-area figures: Figure 1a–1c
# =============================================================================

def point_ba_to_pct_grid(df, lat_col, lon_col, acres_col, years, tree, nrows, ncols):
    df = df.dropna(subset=[lat_col, lon_col, acres_col]).copy()
    _, idx = tree.query(np.column_stack((df[lat_col].values, df[lon_col].values)), k=1)
    df["ROW"], df["COL"] = idx // ncols, idx % ncols

    grouped = (
        df.groupby(["YEAR", "ROW", "COL"], observed=True)[acres_col]
        .sum().unstack(level=0).reindex(columns=years).fillna(0.0)
    )
    pct = grouped.mean(axis=1).values / CELL_AREA_ACRES * 100.0

    grid = np.full((nrows, ncols), np.nan)
    for (row, col), value in zip(grouped.index, pct):
        grid[row, col] = value
    return grid


def plot_ba_map(lon, lat, data, states_gdf, extent, cmap, norm, label, output, is_diff=False):
    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600, subplot_kw={"projection": BA_PROJ})
    add_southeast_basemap(ax, states_gdf, extent, facecolor="w" if is_diff else "#e6e4e6")

    im = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading="auto",
                       transform=ccrs.PlateCarree(), zorder=2)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.35, pad=0.05)
    if is_diff:
        cbar.set_ticks(DIFF_BOUNDS)
        cbar.set_ticklabels(DIFF_BOUNDS)
    cbar.set_label(label, fontsize=9)
    for tick in cbar.ax.xaxis.get_ticklabels():
        tick.set_fontsize(7)

    plt.tight_layout()
    savefig(os.path.join(DIR_FIG, output))


def prepare_ba_summaries(permit_df, nei_df):
    permit = (
        permit_df.groupby(["STATE", "YEAR"], observed=True)["ACRES"]
        .sum().reset_index()
    )
    nei = (
        nei_df.groupby(["STATE", "YEAR"], observed=True)["ACRESBURNED"]
        .sum().reset_index()
        .rename(columns={"ACRESBURNED": "ACRES"})
    )

    permit["Inventory"] = "Permits"
    nei["Inventory"] = "NEI"
    annual = pd.concat([permit, nei], ignore_index=True)
    annual = annual[annual["STATE"].isin(SE_ST_ABBR)].copy()
    annual["STATE"] = pd.Categorical(annual["STATE"], categories=["FL", "GA", "SC"], ordered=True)

    annual_long = (
        annual.groupby(["Inventory", "STATE", "YEAR"], observed=True)["ACRES"]
        .sum().reset_index().sort_values(["Inventory", "STATE", "YEAR"])
        .rename(columns={"STATE": "State", "YEAR": "Year", "ACRES": "Acres"})
    )
    summary = (
        annual_long.pivot_table(index=["Inventory", "State"], columns="Year",
                                values="Acres", aggfunc="sum")
        .reindex(columns=YEARS)
    )
    summary["Mean_2017_2019"] = summary[YEARS].mean(axis=1)
    summary["Std_2017_2019"] = summary[YEARS].std(axis=1, ddof=1)
    summary["Total_2017_2019"] = summary[YEARS].sum(axis=1)
    summary = summary.reset_index()

    annual_long.to_csv(os.path.join(DIR_FIG, "annual_burned_acres_by_state_2017_2019.csv"), index=False)
    summary.to_csv(os.path.join(DIR_FIG, "burned_acres_state_stats_2017_2019.csv"), index=False)

    with pd.option_context("display.max_rows", None, "display.float_format", "{:,.0f}".format):
        print("\nAnnual burned acres by state & inventory (2017–2019):")
        print(annual_long.to_string(index=False))
        print("\nPer-state mean, std, and totals across 2017–2019 (by inventory):")
        print(summary.to_string(index=False))

    return annual.rename(columns={"ACRES": "Value"})


def draw_lollipop_panel(df, value_col, output, xlabel, year_colors, marker, marker_sizes,
                        break_low=None, break_high=None, single_if_small_gap=False,
                        low_xlim_min=None, low_xlim_max_scale=1.05,
                        scilimits=(5, 5), y_fontsize=12,
                        legend_loc="upper left", legend_bbox=(0, 1.05),
                        single_legend_loc=None, single_legend_bbox=None,
                        special_size=None):
    y_positions, ytick_values, ytick_labels = make_y_positions()
    all_values = df.groupby(["STATE", "Inventory", "YEAR"], observed=True)[value_col].sum().values

    if break_low is None or break_high is None:
        sorted_values = np.sort(all_values)
        diffs = np.diff(sorted_values)
        if len(diffs) > 0:
            gap_idx = np.argmax(diffs)
            break_low = sorted_values[gap_idx] * 1.05
            break_high = sorted_values[gap_idx + 1] * 0.95
        else:
            break_low, break_high = 5.0e4, 7.0e4
        if break_low >= break_high:
            break_low, break_high = 5.0e4, 7.0e4

    use_broken = True
    if single_if_small_gap:
        sorted_values = np.sort(all_values)
        use_broken = (break_high - break_low) > 0.15 * (sorted_values[-1] - sorted_values[0])

    if use_broken:
        fig = plt.figure(figsize=(7, 2.8))
        gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.02)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(figsize=(7, 2.8))
        ax2, axes = None, [ax1]

    def draw_points(ax_low, ax_high=None):
        for inventory in INVENTORIES:
            inv_df = df[df["Inventory"] == inventory]
            for state in BOTTOM_TO_TOP_STATES:
                sub = inv_df[inv_df["STATE"] == state].sort_values("YEAR")
                if sub.empty:
                    continue

                y = y_positions[(state, inventory)]
                size = marker_sizes[inventory]
                mean_value = sub[value_col].mean()

                if ax_high is None:
                    ax_low.plot(sub[value_col], np.full(len(sub), y), color="grey", lw=3, zorder=2)
                    ax_low.plot([mean_value, mean_value], [y - 0.25, y + 0.25],
                                color="#DB5127", lw=1.8, zorder=6)
                else:
                    lo = sub[sub[value_col] <= break_low]
                    hi = sub[sub[value_col] >= break_high]
                    if not lo.empty:
                        ax_low.plot(lo[value_col], np.full(len(lo), y), color="grey", lw=3, zorder=2)
                    if not hi.empty:
                        ax_high.plot(hi[value_col], np.full(len(hi), y), color="grey", lw=3, zorder=2)
                    if mean_value <= break_low:
                        ax_low.plot([mean_value, mean_value], [y - 0.25, y + 0.25],
                                    color="#DB5127", lw=1.8, zorder=6)
                    if mean_value >= break_high:
                        ax_high.plot([mean_value, mean_value], [y - 0.25, y + 0.25],
                                     color="#DB5127", lw=1.8, zorder=6)

                for z_offset, year in enumerate(reversed(YEARS)):
                    row = sub[sub["YEAR"] == year]
                    if row.empty:
                        continue
                    value = row[value_col].values[0]
                    plot_size = special_size(state, inventory, year, size) if special_size else size

                    if ax_high is None or value <= break_low:
                        ax_low.scatter(value, y, s=plot_size, c=year_colors[year], marker=marker,
                                       edgecolor="#312738", zorder=3 + z_offset)
                    if ax_high is not None and value >= break_high:
                        ax_high.scatter(value, y, s=plot_size, c=year_colors[year], marker=marker,
                                        edgecolor="#312738", zorder=3 + z_offset)

    draw_points(ax1, ax2)

    if ax2 is None:
        sorted_values = np.sort(all_values)
        ax1.set_xlim(sorted_values[0] * 0.4, sorted_values[-1] * 1.05)
        ax1.set_xlabel(xlabel, fontsize=14, labelpad=20)
    else:
        lo_values = [v for v in all_values if v <= break_low]
        hi_values = [v for v in all_values if v >= break_high]
        ax1.set_xlim((low_xlim_min if low_xlim_min is not None else min(lo_values) * 0.3),
                     break_low * low_xlim_max_scale)
        ax2.set_xlim(break_high, max(hi_values) * 1.05)
        ax2.set_xlabel(xlabel, fontsize=14, labelpad=20)

        d = 0.01
        ax1.plot((1 - d * 2, 1 + d * 2), (-d * 2, +d * 2),
                 transform=ax1.transAxes, color="k", clip_on=False)
        ax2.plot((-d, +d), (-d * 2, +d * 2),
                 transform=ax2.transAxes, color="k", clip_on=False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax1.yaxis.tick_left()
        ax2.tick_params(axis="y", which="both", length=0)

    for ax in axes:
        ax.set_yticks(ytick_values)
        ax.set_yticklabels(ytick_labels if ax is ax1 else [], fontsize=y_fontsize)
        ax.tick_params(axis="x", labelsize=12)
        format_sci_xaxis(ax, scilimits)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", linestyle=":", linewidth=1, zorder=1)
        ax.set_ylim(-1, len(BOTTOM_TO_TOP_STATES) * 2)

    handles = [
        Line2D([0], [0], marker=marker, color="w", label=str(year),
               markerfacecolor=year_colors[year], markeredgecolor="#312738", markersize=10)
        for year in YEARS
    ]
    handles.append(Line2D([0], [0], color="#DB5127", lw=1.8, label="Multi-year mean"))
    legend_ax = ax2 if ax2 is not None else ax1
    actual_legend_loc = legend_loc if ax2 is not None or single_legend_loc is None else single_legend_loc
    actual_legend_bbox = legend_bbox if ax2 is not None else single_legend_bbox
    legend_kw = dict(handles=handles, loc=actual_legend_loc, ncol=4, frameon=False, fontsize=12)
    if actual_legend_bbox is not None:
        legend_kw["bbox_to_anchor"] = actual_legend_bbox
    legend_ax.legend(**legend_kw)

    savefig(os.path.join(DIR_FIG, output))


def run_burned_area_figures(lon, lat, nrows, ncols, extent, states_gdf):
    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))
    state_mask = make_state_mask(lon, lat, states_gdf)

    permit_df = load_annual_csvs(PERMIT_BA_TPL, YEARS, parse_dates_col="DATE")
    nei_df = load_annual_csvs(NEI_BA_TPL, YEARS, parse_dates_col="DATE")

    permit_grid = point_ba_to_pct_grid(permit_df, "LATITUDE", "LONGITUDE", "ACRES", YEARS, tree, nrows, ncols)
    nei_grid = point_ba_to_pct_grid(nei_df, "latitude", "longitude", "ACRESBURNED", YEARS, tree, nrows, ncols)

    permit_masked = np.ma.masked_where(~state_mask, permit_grid)
    nei_masked = np.ma.masked_where(~state_mask, nei_grid)
    diff_masked = permit_masked - nei_masked

    plot_ba_map(lon, lat, permit_masked, states_gdf, extent, BA_CMAP, BA_NORM,
                "Mean annual reported burned area (%)", "f1a_spatial_ba_permits.png", is_diff=False)
    print("Max % Burned:", np.nanmax(permit_masked.compressed()))
    print("Min % Burned:", np.nanmin(permit_masked.compressed()))

    plot_ba_map(lon, lat, diff_masked, states_gdf, extent, DIFF_CMAP, DIFF_NORM,
                "Difference between permits and NEI (%)", "f1b_spatial_ba_diff.png", is_diff=True)
    print("Max difference:", np.nanmax(diff_masked.compressed()))
    print("Min difference:", np.nanmin(diff_masked.compressed()))

    ba_summary = prepare_ba_summaries(permit_df, nei_df)
    draw_lollipop_panel(
        ba_summary, "Value", "f1c_annual_ba_diff.png",
        xlabel="Reported acres burned per year", year_colors=BA_YEAR_COLORS,
        marker="o", marker_sizes={"Permits": 200, "NEI": 50},
        break_low=4.4e5, break_high=0.9e6, low_xlim_min=3.49e5,
        low_xlim_max_scale=1.0, scilimits=(5, 5), y_fontsize=12,
        legend_loc="upper left", legend_bbox=(0, 1.05),
    )


# =============================================================================
# Emissions figures: Figure 1d–1e
# =============================================================================

def jan_dec_dates(years):
    dates = []
    for yr in years:
        dates.extend(pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D").tolist())
    return pd.DatetimeIndex(dates)


def load_gridded_inventory(inv_dir, prefix, dates, species_map):
    data = {sp: [] for sp in species_map}
    valid_dates = []

    for date in dates:
        path = os.path.join(inv_dir, f"{prefix}_CMAQ12US1_{date:%Y%m%d}.nc")
        if not os.path.isfile(path):
            continue

        with nc.Dataset(path) as ds:
            found = False
            day = {}
            for sp, varname in species_map.items():
                if varname in ds.variables:
                    day[sp] = np.array(ds.variables[varname][:].squeeze())
                    found = True
                else:
                    day[sp] = None

        if found:
            valid_dates.append(date)
            for sp in species_map:
                data[sp].append(day[sp])

    for sp in species_map:
        arrays = [arr for arr in data[sp] if arr is not None]
        data[sp] = np.stack(arrays, axis=0) if arrays else None

    return data, valid_dates


def annual_emis_sum(data_3d):
    return (data_3d * 86400.0 * 1e6 * 1e-3).sum(axis=0)


def zero_to_nan(arr):
    out = arr.copy().astype(float)
    out[out == 0] = np.nan
    return out


def plot_emis_map(arr, lon, lat, cell_area_m2, states_gdf, extent, inventory, species,
                  vmin, vmax, ticks, output):
    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600, subplot_kw={"projection": EMIS_PROJ})
    add_southeast_basemap(ax, states_gdf, extent, facecolor="#e6e4e6", city_fontsize=6)

    im = ax.pcolormesh(lon, lat, zero_to_nan(arr), cmap=cmocean.cm.matter,
                       vmin=vmin, vmax=vmax, shading="auto",
                       transform=ccrs.PlateCarree(), zorder=2)

    total_tg = np.nansum(arr * (cell_area_m2 / 1e6)) / 1e6
    ax.set_title(f"{inventory}\n\nTotal: {total_tg:.4f} Tg", fontweight="bold", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.35, pad=0.05, extend="max")
    unit = r"Mg $\mathregular{km^{-2}}$ $\mathregular{yr^{-1}}$"
    cbar.set_label(f"{SPECIES_DISPLAY.get(species, species)} Emissions ({unit})", size=11)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{tick:1g}" for tick in ticks])
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.tick_params(which="major", length=6, width=1.5)
    cbar.ax.tick_params(which="minor", length=3, width=1)
    cbar.ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))

    savefig(os.path.join(DIR_FIG, output))
    print(f"    Saved {output}")


def run_gridded_emissions_figures(states_gdf):
    print("\nReading CMAQ grid …")
    lon, lat, cell_area_m2, extent = read_emis_grid(METCRO2D_FILE)
    print(f"  Grid: {lon.shape[1]} x {lon.shape[0]},  cell area: {cell_area_m2 / 1e6:.2f} km²")
    print(f"  Map extent: {extent}")
    print(f"  Loaded {len(states_gdf)} SE states: {list(states_gdf['STUSPS'])}")

    dates = jan_dec_dates(YEARS)
    inv_dirs = {
        "Permits": (os.path.join(GRIDDED_EMIS_DIR, "Permit"), "Permit"),
        "NEI": (os.path.join(GRIDDED_EMIS_DIR, "NEI"), "NEI"),
    }

    all_data, all_dates = {}, {}
    for inventory in INVENTORIES:
        sp_map = {sp: VARMAP[sp][inventory] for sp in SPECIES if inventory in VARMAP[sp]}
        inv_dir, prefix = inv_dirs[inventory]
        print(f"\nLoading {inventory} ({len(dates)} dates, {len(sp_map)} species) …")
        all_data[inventory], all_dates[inventory] = load_gridded_inventory(inv_dir, prefix, dates, sp_map)
        n_years = len({d.year for d in all_dates[inventory]}) if all_dates[inventory] else 0
        print(f"  {len(all_dates[inventory])} valid days, {n_years} year(s)")

    for species in SPECIES:
        print(f"\n====== {species} ======")
        annual = {}

        for inventory in INVENTORIES:
            data_3d = all_data[inventory].get(species)
            valid_dates = all_dates[inventory]
            if data_3d is None or len(valid_dates) == 0:
                annual[inventory] = None
                continue

            n_years = len({d.year for d in valid_dates})
            arr = annual_emis_sum(data_3d)
            annual[inventory] = arr / n_years if n_years > 1 else arr

        valid = [arr for arr in annual.values() if arr is not None and np.any(arr > 0)]
        data_max = float(max(np.nanmax(arr) for arr in valid)) if valid else 100.0
        ticks = ticker.MaxNLocator(nbins=4, integer=True, min_n_ticks=5).tick_values(0, data_max)[:5]
        vmin, vmax = ticks[0], ticks[-1]
        print(f"  Colorbar ticks: {ticks}  (vmin={vmin}, vmax={vmax})")

        for inventory in INVENTORIES:
            if annual[inventory] is None:
                print(f"  {inventory}: N/A for {species}")
                continue

            print(f"  Plotting {inventory} – {species} …")
            plot_emis_map(
                annual[inventory], lon, lat, cell_area_m2, states_gdf, extent,
                inventory, species, vmin, vmax, ticks,
                f"f1d_{species}_{inventory}_spatial_mean_annual.png",
            )

    print("\n=== Figure 1d done ===")


def load_permits_pm25(years=YEARS):
    """Load permit-based BlueSky PM2.5 emissions and summarize by state and year."""
    frames = []
    for year in years:
        df = pd.read_csv(PERMIT_PM25_TPL.format(yr=year))
        df = df.rename(columns={"state": "STATE"})
        df["YEAR"] = year
        frames.append(df[["STATE", "YEAR", "PM2.5"]])

    out = pd.concat(frames, ignore_index=True)
    out["STATE"] = out["STATE"].astype(str).str.upper()
    annual = (
        out.groupby(["STATE", "YEAR"], observed=True)["PM2.5"]
        .sum()
        .reset_index()
        .rename(columns={"PM2.5": "PM25_Mg"})
    )
    annual["PM25_Mg"] *= 0.907185  # short-ton -> Mg
    annual["Inventory"] = "Permits"
    return annual


def load_nei_pm25(years=YEARS):
    """Load NEI PM2.5 emissions and summarize by state and year."""
    frames = []
    for year in years:
        df = pd.read_csv(NEI_PM25_TPL.format(yr=year))
        df["YEAR"] = year
        frames.append(df[["STATE", "YEAR", "PM2_5"]])

    out = pd.concat(frames, ignore_index=True)
    out["STATE"] = out["STATE"].astype(str).str.upper()
    annual = (
        out.groupby(["STATE", "YEAR"], observed=True)["PM2_5"]
        .sum()
        .reset_index()
        .rename(columns={"PM2_5": "PM25_Mg"})
    )
    annual["PM25_Mg"] *= 0.907185  # short-ton -> Mg
    annual["Inventory"] = "NEI"
    return annual


def print_and_save_pm25_summary(df):
    """Save the Figure 1e data table and the 2017-2019 summary statistics."""
    summary = (
        df.pivot_table(index=["Inventory", "STATE"], columns="YEAR",
                       values="PM25_Mg", aggfunc="sum")
        .reindex(columns=YEARS)
    )
    summary["Mean_2017_2019"] = summary[YEARS].mean(axis=1)
    summary["Std_2017_2019"] = summary[YEARS].std(axis=1, ddof=1)
    summary["Total_2017_2019"] = summary[YEARS].sum(axis=1)
    summary = summary.reset_index()

    csv_out = os.path.join(DIR_FIG, "annual_rx_pm25_emis_by_state_2017_2019.csv")
    stats_out = os.path.join(DIR_FIG, "rx_pm25_emis_state_stats_2017_2019.csv")
    df.to_csv(csv_out, index=False)
    summary.to_csv(stats_out, index=False)

    with pd.option_context("display.max_rows", None, "display.float_format", "{:,.0f}".format):
        print("\n  Annual PM2.5 emissions (Mg) by state & inventory:")
        print(df.to_string(index=False))
        print("\nPer-state mean, std, and totals across 2017–2019 (by inventory):")
        print(summary.to_string(index=False))
    print(f"  Wrote -> {csv_out}")
    print(f"  Wrote -> {stats_out}")


def plot_pm25_single_axis(df, ax, y_positions):
    """Original Figure 1e point/mean-line drawing logic, without a broken x-axis."""
    marker_sizes = {"Permits": 200, "NEI": 50}
    mean_line_half_h = 0.25

    for inventory in INVENTORIES:
        inv_df = df[df["Inventory"] == inventory]
        size = marker_sizes[inventory]

        for state in BOTTOM_TO_TOP_STATES:
            sub = inv_df[inv_df["STATE"] == state].sort_values("YEAR")
            if sub.empty:
                continue

            y = y_positions[(state, inventory)]
            ax.plot(sub["PM25_Mg"], np.full(len(sub), y),
                    color="grey", lw=3, zorder=2)

            mean_value = sub["PM25_Mg"].mean()
            ax.plot([mean_value, mean_value],
                    [y - mean_line_half_h, y + mean_line_half_h],
                    color="#DB5127", lw=1.8, zorder=6)

            for z_offset, year in enumerate(reversed(YEARS)):
                row = sub[sub["YEAR"] == year]
                if row.empty:
                    continue

                value = row["PM25_Mg"].values[0]
                plot_size = size * 0.4 if (
                    state == "SC" and inventory == "Permits" and year == 2017
                ) else size

                ax.scatter(value, y, s=plot_size, c=PM25_YEAR_COLORS[year],
                           marker="^", edgecolor="#312738", zorder=3 + z_offset)


def add_pm25_legend(ax, loc="upper right", bbox=None):
    """Year-coloured triangle legend plus multi-year mean line."""
    handles = [
        Line2D([0], [0], marker="^", color="w", label=str(year),
               markerfacecolor=PM25_YEAR_COLORS[year], markeredgecolor="#312738",
               markersize=10)
        for year in YEARS
    ]
    handles.append(Line2D([0], [0], color="#DB5127", lw=1.8,
                          label="Multi-year mean"))

    legend_kw = dict(handles=handles, loc=loc, ncol=4, frameon=False, fontsize=12)
    if bbox is not None:
        legend_kw["bbox_to_anchor"] = bbox
    ax.legend(**legend_kw)


def run_pm25_lollipop():
    """
    Figure 1e: annual total PM2.5 emissions by state and inventory.

    This version intentionally uses a single x-axis. It keeps the original
    Figure 1e data loading, markers, sizes, colors, y-positioning, labels, and
    output file name, but removes the broken-axis branch so no points can be
    hidden by an omitted x-axis interval.
    """
    plt.rcParams.update({"font.size": 12})

    df = pd.concat([load_permits_pm25(), load_nei_pm25()], ignore_index=True)
    df = df[df["STATE"].isin(BOTTOM_TO_TOP_STATES)].copy()
    df["STATE"] = pd.Categorical(df["STATE"], categories=["FL", "GA", "SC"], ordered=True)

    print_and_save_pm25_summary(df)

    y_positions = {}
    for i, state in enumerate(BOTTOM_TO_TOP_STATES):
        y_positions[(state, "Permits")] = i * 2 + 0.3
        y_positions[(state, "NEI")] = i * 2 - 0.3

    ytick_values = [y_positions[(state, inventory)]
                    for state in BOTTOM_TO_TOP_STATES for inventory in INVENTORIES]
    ytick_labels = [f"{state} ({inventory})"
                    for state in BOTTOM_TO_TOP_STATES for inventory in INVENTORIES]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    plot_pm25_single_axis(df, ax, y_positions)

    values = df.groupby(["STATE", "Inventory", "YEAR"], observed=True)["PM25_Mg"].sum().values
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No PM2.5 emissions values were loaded for Figure 1e.")

    xmin = values.min() * 0.4 if values.min() > 0 else 0
    xmax = values.max() * 1.05
    if xmin == xmax:
        xmax = xmin + 1

    ax.set_xlim(xmin, xmax)
    ax.set_yticks(ytick_values)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.tick_params(axis="x", labelsize=12)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.xaxis.get_major_formatter().set_useMathText(True)

    ax.set_xlabel(r"$\mathrm{PM_{2.5}}$ emissions in Mg per year", fontsize=14, labelpad=20)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", linestyle=":", linewidth=1, zorder=1)
    ax.set_ylim(-1, len(BOTTOM_TO_TOP_STATES) * 2)

    add_pm25_legend(ax, loc="upper right")

    out = os.path.join(DIR_FIG, "f1e_annual_pm25_diff.png")
    plt.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  Saved figure -> {out}")


# =============================================================================
# Main workflow
# =============================================================================

def main():
    states_gdf = load_states()
    ba_lon, ba_lat, nrows, ncols, ba_extent = read_ba_grid(METCRO2D_FILE)

    run_burned_area_figures(ba_lon, ba_lat, nrows, ncols, ba_extent, states_gdf)
    run_gridded_emissions_figures(states_gdf)
    run_pm25_lollipop()

    print("\n=== All Figure 1 outputs complete ===")


if __name__ == "__main__":
    main()