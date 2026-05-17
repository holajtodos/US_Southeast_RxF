# -*- coding: utf-8 -*-
"""
Figure S6: High-burn-season burned-area comparison across fire inventories.

This script maps permit, NEI, and FINN point records to the 12-km CMAQ grid, 
loads the precomputed SEFM gridded product:

    fig_a_spatial_ba_permits_JanApr.png
    fig_b_spatial_diff_permits_NEI_JanApr.png
    fig_c_spatial_diff_permits_FINN_JanApr.png
    fig_d_spatial_diff_permits_SEFM_JanApr.png
    fig_e_annual_ba_all_inventories_JanApr.png
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = "/home/jh94030/scripts/python/postdoc_project/rxfire"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figure")

DIR_SCRIPTS = os.path.join(BASE_DIR, "analysis")
sys.path.append(os.path.join(DIR_SCRIPTS, "step3_BurnDataSelection"))
from util import CMAQGrid2D

YEARS = [2017, 2018, 2019]
SE_ST_ABBR = ["FL", "GA", "SC"]
CELL_AREA_ACRES = 12 * 12 * 247.105

PERMIT_TEMPLATE = os.path.join(
    DATA_DIR, "SE_permit_data_2010-2020/update_criteria",
    "SE_Combined_Permit_lf_3states_rx_{}.csv",
)
NEI_TEMPLATE = os.path.join(
    DATA_DIR, "oth_fire_inv/NEI_rxf_inv", "SE_Combined_NEI_rx_3states_{}.csv",
)
FINN_TEMPLATE = os.path.join(
    DATA_DIR, "oth_fire_inv/FINN_rxf_inv", "SE_Combined_FINN_rx_wf_{}_Jan-Apr.csv",
)
SEFM_PCT_NPY = os.path.join(
    DATA_DIR, "oth_fire_inv/SEFM_gridded_daily", "sefm_pct_grid.npy",
)
SEFM_STATE_CSV = os.path.join(
    DATA_DIR, "oth_fire_inv/SEFM_gridded_daily",
    "sefm_annual_acres_by_state_JanApr.csv",
)
STATES_SHP = (
    "/work/chflab/jthuang/breadcrumbs/mapping_state/"
    "cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
)
METCRO2D_FILE = (
    "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/"
    "mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc"
)

COLORS_ABS = ["#e6e4e6", "#eed5bb", "#f6c690", "#ee9e6b", "#d55e4d", "#bd1e2f"]
BOUNDS_ABS = [0, 5, 10, 25, 50, 75, 100]
COLORS_DIFF = [
    "#222831", "#393E46", "#746170", "#99879C", "#c2b7c6",
    "#fee0b6", "#fdb863", "#e08214", "#b35806", "#7f3b08",
]
BOUNDS_DIFF = [-100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100]
COLORS_YEAR = {2017: "#C9E8ED", 2018: "#84B7D6", 2019: "#508CB6"}

CITIES = {
    "Atlanta": (33.7490, -84.3880), "Orlando": (28.5383, -81.3792),
    "Tallahassee": (30.4383, -84.2807), "Columbia": (34.0007, -81.0348),
    "Jacksonville": (30.3322, -81.6557), "Savannah": (32.0809, -81.0912),
    "Pensacola": (30.4213, -87.2169), "Tampa": (27.9506, -82.4572),
    "Miami": (25.7617, -80.1918), "Columbus": (32.4600, -84.9877),
    "Albany": (31.5785, -84.1557), "Charleston": (32.7765, -79.9311),
}
STATE_LABELS = [("FL", (-84.5, 28.5)), ("GA", (-87.0, 32.5)), ("SC", (-78.5, 34.2))]


# =============================================================================
# Setup, loading, and regridding
# =============================================================================

def setup():
    os.makedirs(FIG_DIR, exist_ok=True)
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]


def load_grid():
    info = CMAQGrid2D(METCRO2D_FILE)
    lon, lat = info["Lon"], info["Lat"]
    nrows, ncols = lat.shape
    extent = (lon[0, 0] + 32, lon[-1, -1] - 22, lat[0, 0] + 2, lat[-1, -1] - 15)
    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))
    return lon, lat, nrows, ncols, extent, tree


def load_state_boundaries_and_mask(lon, lat):
    states = gpd.read_file(STATES_SHP)
    gdf_se = states[states["STUSPS"].isin(SE_ST_ABBR)].copy()
    union_geom = unary_union(gdf_se.to_crs(epsg=4326).geometry.values)
    points = np.column_stack((lon.ravel(), lat.ravel()))
    mask_flat = np.fromiter(
        (union_geom.contains(Point(xy)) or union_geom.touches(Point(xy)) for xy in points),
        dtype=bool,
        count=points.shape[0],
    )
    return gdf_se, mask_flat.reshape(lon.shape)


def load_inventory(template, date_col, label, months=None):
    frames = []
    for year in YEARS:
        fpath = template.format(year)
        if not os.path.isfile(fpath):
            print(f"  WARNING: {label} file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[date_col])
        df["YEAR"] = year
        if months is not None:
            df = df[df[date_col].dt.month.between(months[0], months[1])]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No {label} files found.")

    out = pd.concat(frames, ignore_index=True)
    print(f"  {label}: {len(out):,} records, {out['YEAR'].nunique()} years")
    return out


def points_to_percent_grid(df, lat_col, lon_col, acres_col, tree, nrows, ncols):
    df = df.copy().dropna(subset=[lat_col, lon_col, acres_col])
    _, idx_flat = tree.query(np.column_stack((df[lat_col].values, df[lon_col].values)), k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols

    annual = (
        df.groupby(["YEAR", "ROW", "COL"], observed=True)[acres_col]
        .sum().unstack(level=0).reindex(columns=YEARS).fillna(0.0)
    )
    grid = np.full((nrows, ncols), np.nan)
    mean_pct = annual.mean(axis=1).values / CELL_AREA_ACRES * 100.0
    for (row, col), value in zip(annual.index, mean_pct):
        grid[row, col] = value
    return grid


def state_totals(df, value_col):
    return df.groupby(["STATE", "YEAR"], observed=True)[value_col].sum().reset_index()


def load_sefm_precomputed():
    if not os.path.isfile(SEFM_PCT_NPY):
        raise FileNotFoundError(f"Missing SEFM grid file: {SEFM_PCT_NPY}")
    if not os.path.isfile(SEFM_STATE_CSV):
        raise FileNotFoundError(f"Missing SEFM state summary file: {SEFM_STATE_CSV}")
    print("  Loaded precomputed SEFM grid and annual state totals.")
    return np.load(SEFM_PCT_NPY), pd.read_csv(SEFM_STATE_CSV)


# =============================================================================
# Shared map styling
# =============================================================================

def draw_labels_and_cities(ax):
    for label, xy in STATE_LABELS:
        text = ax.text(xy[0], xy[1], label, color="k", fontweight="bold", fontsize=16,
                       transform=ccrs.Geodetic())
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground="white"),
                               path_effects.Normal()])

    for name, (lat, lon) in CITIES.items():
        ax.scatter(lon, lat, marker="o", facecolor="none", edgecolor="k", s=10,
                   transform=ccrs.PlateCarree(), zorder=3)
        ax.text(lon + 0.15, lat - 0.1, name, fontsize=7, fontweight="bold",
                color="black", transform=ccrs.PlateCarree(), zorder=3,
                path_effects=[path_effects.Stroke(linewidth=1, foreground="white"),
                              path_effects.Normal()])


def add_basemap(ax, gdf_se, extent, fill_color):
    ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
    ax.axis("off")
    geom = gdf_se.to_crs(epsg=4326).geometry
    ax.add_geometries(geom, crs=ccrs.PlateCarree(), facecolor=fill_color,
                      edgecolor="k", linewidth=0, zorder=1)
    ax.add_geometries(geom, crs=ccrs.PlateCarree(), facecolor="none",
                      edgecolor="k", linewidth=1.2, zorder=3)
    draw_labels_and_cities(ax)


# =============================================================================
# Spatial panels
# =============================================================================

def plot_permits_map(lon, lat, grid, state_mask, gdf_se, extent):
    data = np.ma.masked_where(~state_mask, grid)
    cmap = LinearSegmentedColormap.from_list("burned_pct", COLORS_ABS)
    norm = BoundaryNorm(BOUNDS_ABS, cmap.N)

    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600,
                           subplot_kw={"projection": ccrs.AlbersEqualArea(
                               central_longitude=-88, central_latitude=33)})
    add_basemap(ax, gdf_se, extent, fill_color="#e6e4e6")
    im = ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, shading="auto",
                       transform=ccrs.PlateCarree(), zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="w", edgecolor="k", linewidth=0.5, zorder=2)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.35, pad=0.05)
    cbar.set_label("Mean annual (Jan-Apr) reported burned area (%)", fontsize=9)
    for label in cbar.ax.xaxis.get_ticklabels():
        label.set_fontsize(7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_a_spatial_ba_permits_JanApr.png")
    plt.savefig(out, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"  Saved figure (a) -> {out}")
    print(f"    Max % burned: {np.nanmax(data.compressed()):.2f}")
    print(f"    Min % burned: {np.nanmin(data.compressed()):.2f}")


def plot_difference_map(lon, lat, permits_grid, other_grid, state_mask, gdf_se, extent, label, outfile):
    permits = np.ma.masked_where(~state_mask, permits_grid)
    other = np.ma.masked_where(~state_mask, other_grid)
    diff = permits - other

    cmap = LinearSegmentedColormap.from_list("burned_diff", COLORS_DIFF)
    norm = BoundaryNorm(BOUNDS_DIFF, cmap.N)

    fig, ax = plt.subplots(1, figsize=(7, 5.25), dpi=600,
                           subplot_kw={"projection": ccrs.AlbersEqualArea(
                               central_longitude=-88, central_latitude=33)})
    add_basemap(ax, gdf_se, extent, fill_color="w")
    im = ax.pcolormesh(lon, lat, diff, cmap=cmap, norm=norm, shading="auto",
                       transform=ccrs.PlateCarree(), zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="w", edgecolor="k", linewidth=0.5, zorder=2)

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.35, pad=0.05)
    cbar.set_ticks(BOUNDS_DIFF)
    cbar.set_ticklabels(BOUNDS_DIFF)
    cbar.set_label(f"Difference between permits and {label} (%)", fontsize=9)
    for tick in cbar.ax.xaxis.get_ticklabels():
        tick.set_fontsize(7)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, outfile)
    plt.savefig(out, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"  Saved -> {out}")
    print(f"    Max difference: {np.nanmax(diff.compressed()):.2f}")
    print(f"    Min difference: {np.nanmin(diff.compressed()):.2f}")


# =============================================================================
# Panel e: annual burned area by state and inventory
# =============================================================================

def combined_annual_table(df_permit, df_nei, df_finn, df_sefm):
    frames = [
        df_permit.rename(columns={"ACRES": "ACRES"}).assign(Inventory="Permits"),
        df_nei.rename(columns={"ACRESBURNED": "ACRES"}).assign(Inventory="NEI"),
        df_finn.rename(columns={"AREA": "ACRES"}).assign(Inventory="FINN"),
        df_sefm.rename(columns={"ACRES": "ACRES"}).assign(Inventory="SEFM"),
    ]
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["STATE"].isin(["FL", "GA", "SC"])].copy()
    df_all["STATE"] = pd.Categorical(df_all["STATE"], categories=["FL", "GA", "SC"], ordered=True)

    annual = (df_all.groupby(["Inventory", "STATE", "YEAR"], observed=True)["ACRES"]
              .sum().reset_index().sort_values(["Inventory", "STATE", "YEAR"])
              .rename(columns={"STATE": "State", "YEAR": "Year"}))
    wide = annual.pivot_table(index=["Inventory", "State"], columns="Year",
                              values="ACRES", aggfunc="sum").reindex(columns=YEARS)
    wide["Mean"] = wide[YEARS].mean(axis=1)
    wide["Std"] = wide[YEARS].std(axis=1, ddof=1)

    with pd.option_context("display.max_rows", None, "display.float_format", "{:,.0f}".format):
        print("\n  Annual burned acres by state & inventory\n(High-burn season only):")
        print(annual.to_string(index=False))
        print("\n  Summary:")
        print(wide.reset_index().to_string(index=False))

    csv_out = os.path.join(FIG_DIR, "annual_burned_acres_all_inventories_JanApr.csv")
    annual.to_csv(csv_out, index=False)
    print(f"  Wrote -> {csv_out}")
    return df_all


def plot_inventory_points(df, value_col, inventory, size, states, y_pos, break_low, break_high, ax_low, ax_high):
    mean_half_height = 0.25
    for state in states:
        sub = df[df["STATE"] == state].sort_values("YEAR")
        if sub.empty:
            continue
        y = y_pos[(state, inventory)]

        low = sub[sub[value_col] <= break_low]
        high = sub[sub[value_col] >= break_high]
        if not low.empty:
            ax_low.plot(low[value_col], np.full(len(low), y), color="grey", lw=3, zorder=2)
        if not high.empty:
            ax_high.plot(high[value_col], np.full(len(high), y), color="grey", lw=3, zorder=2)

        mean_val = sub[value_col].mean()
        mean_style = dict(color="#DB5127", lw=1.8, zorder=6)
        if mean_val <= break_low:
            ax_low.plot([mean_val, mean_val], [y - mean_half_height, y + mean_half_height], **mean_style)
        if mean_val >= break_high:
            ax_high.plot([mean_val, mean_val], [y - mean_half_height, y + mean_half_height], **mean_style)

        for z_offset, year in enumerate(reversed(YEARS)):
            row = sub[sub["YEAR"] == year]
            if row.empty:
                continue
            value = row[value_col].values[0]
            marker_size = size * 0.65 if (state == "SC" and inventory == "Permits" and year == 2017) else size
            if value <= break_low:
                ax_low.scatter(value, y, s=marker_size, c=COLORS_YEAR[year], edgecolor="#312738", zorder=3 + z_offset)
            if value >= break_high:
                ax_high.scatter(value, y, s=marker_size, c=COLORS_YEAR[year], edgecolor="#312738", zorder=3 + z_offset)


def plot_annual_acres_broken(df_permit, df_nei, df_finn, df_sefm):
    states = ["SC", "GA", "FL"]
    inventories = ["Permits", "NEI", "FINN", "SEFM"]
    df_all = combined_annual_table(df_permit, df_nei, df_finn, df_sefm)

    values = df_all.groupby(["STATE", "Inventory", "YEAR"], observed=True)["ACRES"].sum().values
    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)
    if len(gaps) > 0:
        gap_idx = np.argmax(gaps)
        break_low, break_high = sorted_values[gap_idx] * 1.05, sorted_values[gap_idx + 1] * 0.95
    else:
        break_low, break_high = 5.0e5, 0.7e6
    if break_low >= break_high:
        break_low, break_high = 5.0e5, 0.7e6

    y_pos = {}
    spacing = len(inventories)
    for i, state in enumerate(states):
        for j, inv in enumerate(inventories):
            y_pos[(state, inv)] = i * (spacing + 1) + (spacing - 1) / 2 - j * 0.6
    ytick_vals = [y_pos[(state, inv)] for state in states for inv in inventories]
    ytick_labs = [f"{state} ({inv})" for state in states for inv in inventories]

    fig = plt.figure(figsize=(7, 5.5))
    gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.02)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    plot_inventory_points(df_permit, "ACRES", "Permits", 200, states, y_pos, break_low, break_high, ax1, ax2)
    plot_inventory_points(df_nei, "ACRESBURNED", "NEI", 100, states, y_pos, break_low, break_high, ax1, ax2)
    plot_inventory_points(df_finn, "AREA", "FINN", 70, states, y_pos, break_low, break_high, ax1, ax2)
    plot_inventory_points(df_sefm, "ACRES", "SEFM", 50, states, y_pos, break_low, break_high, ax1, ax2)

    lo_min = min(v for v in values if v <= break_low) if any(v <= break_low for v in values) else 0
    hi_max = max(v for v in values if v >= break_high) if any(v >= break_high for v in values) else break_high * 1.5
    ax1.set_xlim(lo_min * 0.4, break_low)
    ax2.set_xlim(break_high, hi_max * 1.05)

    ax1.set_yticks(ytick_vals)
    ax1.set_yticklabels(ytick_labs, fontsize=10)
    ax2.set_yticks(ytick_vals)
    ax2.set_yticklabels([])
    for ax in [ax1, ax2]:
        ax.tick_params(axis="x", labelsize=12)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
        if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
            ax.xaxis.get_major_formatter().set_useMathText(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(axis="y", linestyle=":", linewidth=1, zorder=1)
        ax.set_ylim(min(ytick_vals) - 1, max(ytick_vals) + 1)

    fig.text(0.5, -0.02, "Acres burned per year\n(High-burn season only)",
             ha="center", va="top", fontsize=14, fontweight="bold")
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.tick_params(axis="y", which="both", length=0)

    d = 0.01
    ax1.plot((1 - d * 2, 1 + d * 2), (-d * 2, +d * 2), transform=ax1.transAxes, color="k", clip_on=False)
    ax2.plot((-d, +d), (-d * 2, +d * 2), transform=ax2.transAxes, color="k", clip_on=False)

    handles = [Line2D([0], [0], marker="o", color="w", label=str(year),
                      markerfacecolor=COLORS_YEAR[year], markeredgecolor="#312738", markersize=10)
               for year in YEARS]
    handles.append(Line2D([0], [0], color="#DB5127", lw=1.8, label="Multi-year mean"))
    ax2.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.5, 1.05),
               ncol=4, frameon=False, fontsize=12)

    out = os.path.join(FIG_DIR, "fig_e_annual_ba_all_inventories_JanApr.png")
    plt.savefig(out, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"  Saved figure (e) -> {out}")


# =============================================================================
# Main workflow
# =============================================================================

def main():
    setup()

    print("=" * 60)
    print("Loading CMAQ grid and state boundaries ...")
    lon, lat, nrows, ncols, extent, tree = load_grid()
    gdf_se, state_mask = load_state_boundaries_and_mask(lon, lat)

    print("\n" + "=" * 60)
    print("Loading and gridding point inventories ...")
    permits = load_inventory(PERMIT_TEMPLATE, "DATE", "Permits (Jan-Apr)", months=(1, 4))
    nei = load_inventory(NEI_TEMPLATE, "DATE", "NEI (Jan-Apr)", months=(1, 4))
    finn = load_inventory(FINN_TEMPLATE, "DAY", "FINN")

    permits_grid = points_to_percent_grid(permits, "LATITUDE", "LONGITUDE", "ACRES", tree, nrows, ncols)
    nei_grid = points_to_percent_grid(nei, "latitude", "longitude", "ACRESBURNED", tree, nrows, ncols)
    finn_grid = points_to_percent_grid(finn, "LATI", "LONGI", "AREA", tree, nrows, ncols)

    print("\n" + "=" * 60)
    print("Loading precomputed SEFM outputs ...")
    sefm_grid, sefm_state = load_sefm_precomputed()

    print("\n" + "=" * 60)
    print("Preparing annual state totals ...")
    permit_state = state_totals(permits, "ACRES")
    nei_state = state_totals(nei, "ACRESBURNED")
    finn_state = state_totals(finn, "AREA")

    print("\n" + "=" * 60)
    print("Plotting spatial panels ...")
    plot_permits_map(lon, lat, permits_grid, state_mask, gdf_se, extent)
    plot_difference_map(lon, lat, permits_grid, nei_grid, state_mask, gdf_se, extent,
                        "NEI", "fig_b_spatial_diff_permits_NEI_JanApr.png")
    plot_difference_map(lon, lat, permits_grid, finn_grid, state_mask, gdf_se, extent,
                        "FINN", "fig_c_spatial_diff_permits_FINN_JanApr.png")
    plot_difference_map(lon, lat, permits_grid, sefm_grid, state_mask, gdf_se, extent,
                        "SEFM", "fig_d_spatial_diff_permits_SEFM_JanApr.png")

    print("\n" + "=" * 60)
    print("Plotting annual state comparison ...")
    plot_annual_acres_broken(permit_state, nei_state, finn_state, sefm_state)

    print("\n" + "=" * 60)
    print("=== All figures saved ===")


if __name__ == "__main__":
    main()