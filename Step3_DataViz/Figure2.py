# -*- coding: utf-8 -*-
###############################################################################
# Figure 2: Rx-fire PM2.5, smoke days, population, and exposure metrics
# Author: Jingting HUANG
#
# Purpose
# -------
# This script shows where Rx fires contributed higher PM2.5 and more 
# smoke-impacted days, then uses regridded population data to estimate 
# state-level population-weighted PM2.5 exposure and smoke-exposure 
# person-days by season.
#
# Outputs
# -------
# Figure 2a:
#   pm25_fire_2017_2019.png
#   joyplot_gradient_rx_fire_pm25.png
#
# Figure 2b:
#   pm25_fire_smoke_days_2017_2019.png
#   joyplot_gradient_rx_fire_smoke_days.png
#
# Figure 2c:
#   reaggregated_pop_12km.png
#
# Figure 2d:
#   lollipop_h_PM25.png
#
# Figure 2e:
#   lollipop_h_PersonDays.png
###############################################################################

import os
import sys
from datetime import datetime

import colorcet as cc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import cartopy.crs as ccrs

from matplotlib import font_manager, ticker
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    LogNorm,
    Normalize,
)
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from shapely.geometry import Point
from shapely.prepared import prep


# =============================================================================
# Environment, paths, and fixed settings
# =============================================================================

os.environ["PROJ_LIB"] = "/home/jh94030/.conda/envs/myenv/share/proj"
os.environ["PROJ_DATA"] = "/home/jh94030/.conda/envs/myenv/share/proj"

DIR_SCRIPTS = "/home/jh94030/scripts/python/postdoc_project/rxfire/analysis"
DIR_FIG = "/home/jh94030/scripts/python/postdoc_project/rxfire/figure"

CMAQ_ALL_PATH = "/scratch/jh94030/CMAQ-output/EQUATES/w+_rxf/no_bs_shift/combined/hr2dy"
CMAQ_RMV_PATH = "/scratch/jh94030/CMAQ-output/EQUATES/wo_rxf/combined/hr2dy"

SHAPEFILE_STATES = "/work/chflab/jthuang/breadcrumbs/mapping_state/cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
SELECTED_GA_COUNTIES_SHP = os.path.join(DIR_FIG, "nonattainment_GA_counties_obs.shp")
SELECTED_FL_COUNTIES_SHP = os.path.join(DIR_FIG, "nonattainment_FL_counties_obs.shp")

POP_TIF = "/work/chflab/jthuang/breadcrumbs/ciesen_nasa/ciesen_nasa_gpw_v4_population_count_2020.tif"

MET_DIR = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
METCRO2D_FILE = f"{MET_DIR}/METCRO2D_20170101.nc"

YEARS = [2017, 2018, 2019]
SE_STATES = ["FL", "GA", "SC"]
SMOKE_THRESHOLD = 3.5  # µg/m3

MAP_EXTENT = [-91, -75, 24, 37]
MAP_PROJ = ccrs.AlbersEqualArea(central_longitude=-88, central_latitude=33)

FONT_ARIAL = "/home/jh94030/fonts/Arial.ttf"
FONT_ARIAL_BOLD = "/home/jh94030/fonts/Arial Bold.ttf"

sys.path.append(os.path.join(DIR_SCRIPTS, "step3_BurnDataSelection"))
from util import CMAQGrid2D  # noqa: E402


# =============================================================================
# Style and shared loading helpers
# =============================================================================

def configure_matplotlib():
    """Load Arial fonts and apply the same font settings used in the originals."""
    font_manager.fontManager.addfont(FONT_ARIAL)
    font_manager.fontManager.addfont(FONT_ARIAL_BOLD)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]


def load_cmaq_grid(metcro2d_file):
    """Load 2-D CMAQ longitude and latitude arrays from METCRO2D."""
    cmaq_info = CMAQGrid2D(metcro2d_file)
    return cmaq_info["Lon"], cmaq_info["Lat"]


def load_state_geometries():
    """Load Southeast state geometries and individual state subsets."""
    gdf_states = gpd.read_file(SHAPEFILE_STATES)
    gdf_se = gdf_states[gdf_states["STUSPS"].isin(SE_STATES)].copy()
    gdf_fl = gdf_states[gdf_states["STUSPS"].isin(["FL"])].copy()
    gdf_ga = gdf_states[gdf_states["STUSPS"].isin(["GA"])].copy()
    gdf_sc = gdf_states[gdf_states["STUSPS"].isin(["SC"])].copy()
    return gdf_states, gdf_se, gdf_fl, gdf_ga, gdf_sc


def load_nonattainment_counties():
    """Load county outlines used only for the Figure 2a PM2.5 map overlay."""
    gdf_ga_counties = gpd.read_file(SELECTED_GA_COUNTIES_SHP)
    gdf_fl_counties = gpd.read_file(SELECTED_FL_COUNTIES_SHP)
    return gdf_ga_counties, gdf_fl_counties


def make_mask(lon_grid, lat_grid, gdf):
    """
    Build a boolean mask for grid cells that fall within or touch a geometry.

    The prepared-geometry approach keeps the same point-in-polygon logic as the
    original scripts, while avoiding repeated geometry checks in later steps.
    """
    union_geom = gdf.geometry.unary_union
    prepared = prep(union_geom)

    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    mask_flat = np.fromiter(
        (
            prepared.contains(Point(lon, lat)) or prepared.touches(Point(lon, lat))
            for lon, lat in zip(lon_flat, lat_flat)
        ),
        dtype=bool,
        count=lon_grid.size,
    )
    return mask_flat.reshape(lon_grid.shape)


def build_masks(cmaq_lon, cmaq_lat, gdf_se, gdf_fl, gdf_ga, gdf_sc):
    """Create Southeast and state-specific masks once and reuse them."""
    return {
        "SE": make_mask(cmaq_lon, cmaq_lat, gdf_se),
        "FL": make_mask(cmaq_lon, cmaq_lat, gdf_fl),
        "GA": make_mask(cmaq_lon, cmaq_lat, gdf_ga),
        "SC": make_mask(cmaq_lon, cmaq_lat, gdf_sc),
    }


def cmaq_file_path(base_dir, year):
    """Construct the annual daily-average CMAQ file path used in both scripts."""
    filename = (
        "dailyavg_o3_pm25_v55_cb6r5_ae7_aq_WR413_MYR_gcc_12US1_"
        f"{year}01-{year}12.nc"
    )
    return os.path.join(base_dir, filename)


def open_cmaq_pair(year):
    """Open paired all-emissions and no-Rx-fire CMAQ outputs for one year."""
    ds_all = xr.open_dataset(cmaq_file_path(CMAQ_ALL_PATH, year))
    ds_rmv = xr.open_dataset(cmaq_file_path(CMAQ_RMV_PATH, year))
    return ds_all, ds_rmv


def rx_fire_pm25(ds_all, ds_rmv):
    """
    Compute daily surface-layer Rx-fire PM2.5 as the nonnegative difference
    between the all-emissions and no-Rx-fire simulations.
    """
    fire_pm25 = (ds_all["PM25_TOT_AVG"] - ds_rmv["PM25_TOT_AVG"]).isel(LAY=0)
    return fire_pm25.where(fire_pm25 > 0, 0)


def clean_flat(data, threshold=0.01):
    """Flatten data, remove NaNs, and keep values above the original threshold."""
    values = data.flatten()
    return values[~np.isnan(values) & (values > threshold)]


# =============================================================================
# Figure 2a-b: Rx-fire PM2.5 and smoke-day spatial/distribution plots
# =============================================================================

def make_pm25_colormap():
    """Return the modified CET_L11 colormap used for Figure 2a."""
    cmap_orig = cc.cm.CET_L11
    return LinearSegmentedColormap.from_list(
        "modifiedCET_L11",
        [
            (0.0, cmap_orig(0.0)),
            (0.25, cmap_orig(0.25)),
            (0.5, cmap_orig(0.5)),
            (1.0, "darkorange"),
        ],
    )


def make_smoke_day_colormap():
    """Return the smoke-day colormap and boundary normalization used for Figure 2b."""
    colors = [
        "#B6B3D6",
        "#CFCCE3",
        "#D5D3DE",
        "#D5D1D1",
        "#F6DFD6",
        "#F8B2A2",
        "#F1837A",
        "#E9687A",
    ]
    bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    cmap = LinearSegmentedColormap.from_list("smoke_days", colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def compute_multi_year_annual_mean_pm25(years):
    """
    Compute the 2017-2019 mean of annual-average Rx-fire PM2.5.

    For each year, daily Rx-fire PM2.5 is averaged over TSTEP. The resulting
    annual mean grids are then averaged across the requested years.
    """
    annual_means = []

    for year in years:
        print(f"[PM2.5] Processing year: {year}")
        ds_all, ds_rmv = open_cmaq_pair(year)
        fire_pm25 = rx_fire_pm25(ds_all, ds_rmv)
        annual_means.append(fire_pm25.mean(dim="TSTEP").values)
        ds_all.close()
        ds_rmv.close()

    return np.nanmean(np.stack(annual_means, axis=0), axis=0)


def compute_multi_year_avg_smoke_days(years, threshold):
    """
    Compute the annual-average number of smoke-affected days.

    A smoke-affected day is counted when daily Rx-fire PM2.5 exceeds the original
    threshold of 3.5 µg/m3 at a grid cell.
    """
    per_year_counts = []

    for year in years:
        print(f"[SmokeDays] Processing year: {year}")
        ds_all, ds_rmv = open_cmaq_pair(year)
        fire_pm25 = rx_fire_pm25(ds_all, ds_rmv)
        counts = (fire_pm25 > threshold).sum(dim="TSTEP").values
        per_year_counts.append(counts.astype(float))
        ds_all.close()
        ds_rmv.close()

    return np.mean(np.stack(per_year_counts, axis=0), axis=0)


def plot_spatial_pm25(cmaq_lon, cmaq_lat, gdf_se, gdf_ga_counties, gdf_fl_counties,
                      mean_fire_pm25_masked, cmap_pm25):
    """Plot Figure 2a spatial map with the original colorbar and county overlays."""
    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=600,
        subplot_kw={"projection": MAP_PROJ},
    )
    ax.set_extent(MAP_EXTENT)
    ax.axis("off")

    ax.add_geometries(
        gdf_se.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="k",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_geometries(
        gdf_ga_counties.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#8C0909",
        linewidth=1,
        zorder=3,
    )
    ax.add_geometries(
        gdf_fl_counties.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#8C0909",
        linewidth=1,
        zorder=3,
    )

    im = ax.pcolormesh(
        cmaq_lon,
        cmaq_lat,
        mean_fire_pm25_masked,
        vmin=0,
        vmax=4.5,
        cmap=cmap_pm25,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    bounds_pm = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5, pad=0.05)
    cbar.set_ticks(bounds_pm)
    cbar.set_ticklabels(bounds_pm)
    cbar.set_label(
        "Annual average Rx fire $\\mathrm{PM}_{2.5}$ ($\\mu g/m^3$)",
        fontsize=7,
        fontweight="bold",
    )
    for label in cbar.ax.xaxis.get_ticklabels():
        label.set_fontsize(7)

    out_png = os.path.join(DIR_FIG, "pm25_fire_2017_2019.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_spatial_smoke_days(cmaq_lon, cmaq_lat, gdf_se, smoke_days_masked,
                            cmap_sd, norm_sd, bounds_sd):
    """Plot Figure 2b spatial smoke-day map with original settings."""
    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=600,
        subplot_kw={"projection": MAP_PROJ},
    )
    ax.set_extent(MAP_EXTENT)
    ax.axis("off")

    ax.add_geometries(
        gdf_se.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="k",
        linewidth=1.2,
        zorder=3,
    )

    im = ax.pcolormesh(
        cmaq_lon,
        cmaq_lat,
        smoke_days_masked,
        cmap=cmap_sd,
        norm=norm_sd,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.5, pad=0.05)
    cbar.set_ticks(bounds_sd)
    cbar.set_ticklabels(bounds_sd)
    cbar.set_label(
        "Annual average Rx fire smoke-affected days",
        fontsize=7,
        fontweight="bold",
    )
    for label in cbar.ax.xaxis.get_ticklabels():
        label.set_fontsize(7)

    out_png = os.path.join(DIR_FIG, "pm25_fire_smoke_days_2017_2019.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out_png}")


def state_distribution_dataframe(data_grid, masks, value_name):
    """Convert state-masked grid values into a long DataFrame for KDE joyplots."""
    frames = []
    for state in SE_STATES:
        values = clean_flat(np.where(masks[state], data_grid, np.nan))
        print(f"{state}  min={min(values)}, max={max(values)}")
        frames.append(pd.DataFrame({"State": state, value_name: values}))
    return pd.concat(frames, ignore_index=True)


def plot_gradient_joyplot(df_long, value_col, x_grid, x_lim, xticks, cmap, norm,
                          xlabel, save_name, y_upper):
    """
    Plot the gradient KDE joyplot used for both PM2.5 and smoke days.

    This preserves the original state order, figure size, DPI, linewidths, tick
    handling, label sizes, and layout choices.
    """
    state_list = ["FL", "GA", "SC"]
    kde_results = {}

    for state in state_list:
        values = df_long[df_long["State"] == state][value_col].values
        kde = gaussian_kde(values, bw_method=0.3)
        kde_results[state] = kde(x_grid)

    fig, axarr = plt.subplots(
        len(state_list),
        1,
        figsize=(6, 4),
        dpi=600,
        sharex=True,
    )

    for i, (ax, state) in enumerate(zip(axarr, state_list)):
        y_vals = kde_results[state]
        offset = i * 1.0
        y_offset = y_vals + offset

        for j in range(len(x_grid) - 1):
            x_seg = x_grid[j:j + 2]
            y_seg = y_offset[j:j + 2]
            color = cmap(norm(np.mean(x_seg)))
            ax.fill_between(x_seg, y_seg, offset, color=color, linewidth=0)

        ax.plot(x_grid, y_offset, color="black", linewidth=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.set_yticks([])
        ax.set_ylabel(
            state,
            fontsize=20,
            fontweight="semibold",
            rotation=0,
            labelpad=20,
            va="center",
        )

        if state == "SC":
            ax.tick_params(axis="x", labelsize=18, direction="out", bottom=True)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", bottom=False)

        ax.set_xlim(*x_lim)
        ax.set_ylim(offset, offset + y_upper)

    xtick_labels = [f"{v:.2g}" for v in xticks]
    axarr[-1].set_xticks(xticks)
    axarr[-1].set_xticklabels(xtick_labels, fontsize=10)
    axarr[-1].tick_params(axis="x", bottom=True, direction="out")
    axarr[-1].set_xlabel(xlabel, fontsize=15, fontweight="bold")

    plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.15, hspace=0.1)

    out_png = os.path.join(DIR_FIG, save_name)
    plt.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f"Saved: {out_png}")


def run_figure_2ab(cmaq_lon, cmaq_lat, gdf_se, masks):
    """Run Figure 2a-b map and joyplot generation."""
    gdf_ga_counties, gdf_fl_counties = load_nonattainment_counties()

    cmap_pm25 = make_pm25_colormap()
    cmap_sd, norm_sd, bounds_sd = make_smoke_day_colormap()

    mean_fire_pm25 = compute_multi_year_annual_mean_pm25(YEARS)
    mean_fire_pm25_masked = np.where(masks["SE"], mean_fire_pm25, np.nan)

    plot_spatial_pm25(
        cmaq_lon,
        cmaq_lat,
        gdf_se,
        gdf_ga_counties,
        gdf_fl_counties,
        mean_fire_pm25_masked,
        cmap_pm25,
    )

    df_pm25 = state_distribution_dataframe(mean_fire_pm25, masks, "PM25")
    plot_gradient_joyplot(
        df_long=df_pm25,
        value_col="PM25",
        x_grid=np.linspace(0, 10, 500),
        x_lim=(0, 4.5),
        xticks=np.linspace(0, 4.5, 10),
        cmap=cmap_pm25,
        norm=Normalize(vmin=0, vmax=4.5),
        xlabel="Annual average Rx fire $\\mathrm{PM}_{2.5}$ ($\\mu g/m^3$)",
        save_name="joyplot_gradient_rx_fire_pm25.png",
        y_upper=2,
    )

    mean_smoke_day_counts = compute_multi_year_avg_smoke_days(YEARS, SMOKE_THRESHOLD)
    mean_smoke_day_counts_masked = np.where(masks["SE"], mean_smoke_day_counts, np.nan)

    plot_spatial_smoke_days(
        cmaq_lon,
        cmaq_lat,
        gdf_se,
        mean_smoke_day_counts_masked,
        cmap_sd,
        norm_sd,
        bounds_sd,
    )

    df_smoke_days = state_distribution_dataframe(mean_smoke_day_counts_masked, masks, "Smoke_Days")
    plot_gradient_joyplot(
        df_long=df_smoke_days,
        value_col="Smoke_Days",
        x_grid=np.linspace(0, 80, 500),
        x_lim=(0, 80),
        xticks=np.linspace(0, 80, 9),
        cmap=cmap_sd,
        norm=norm_sd,
        xlabel="Annual average Rx fire smoke-affected days",
        save_name="joyplot_gradient_rx_fire_smoke_days.png",
        y_upper=0.06,
    )


# =============================================================================
# Figure 2c-e: population aggregation and exposure lollipop plots
# =============================================================================

def load_population(pop_tif):
    """Load GPWv4 population and subset to the original broad U.S. bounds."""
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

        if lat_vals[0] > lat_vals[-1]:
            lat_vals = lat_vals[::-1]
            pop_data = pop_data[::-1, :]

    lat_mask = (lat_vals >= 20) & (lat_vals <= 57)
    lon_mask = (lon_vals >= -135) & (lon_vals <= -53)

    pop_data = pop_data[np.ix_(lat_mask, lon_mask)]
    lat_vals = lat_vals[lat_mask]
    lon_vals = lon_vals[lon_mask]

    return lat_vals, lon_vals, pop_data


def aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon, radius_deg=0.06):
    """
    Aggregate fine population pixels to CMAQ grid cells using the original
    nearest-radius method and radius.
    """
    print("Manual aggregation by finding which high-res pixels belong to each CMAQ cell")

    pop_coords = np.column_stack([
        np.repeat(lat_vals, len(lon_vals)),
        np.tile(lon_vals, len(lat_vals)),
    ])
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
    """
    Compute population-weighted Rx-fire PM2.5 and smoke person-days.

    This preserves the original formulae and the original high-/low-burn
    days_count handling.
    """
    results = {}

    for year in years:
        ds_all, ds_rmv = open_cmaq_pair(year)
        time_coord = ds_all["TFLAG"][:, 0, 0].values

        for idx, tflag in enumerate(time_coord):
            date = datetime.strptime(str(tflag), "%Y%j")
            if date.month not in months:
                continue

            fire_pm25 = (
                ds_all["PM25_TOT_AVG"].isel(TSTEP=idx, LAY=0).values
                - ds_rmv["PM25_TOT_AVG"].isel(TSTEP=idx, LAY=0).values
            )
            fire_pm25[fire_pm25 < 0] = 0

            for state, mask in masks.items():
                weighted_pm25 = np.where(mask, fire_pm25 * pop_agg, np.nan)
                smoke_days = np.where(mask & (fire_pm25 > SMOKE_THRESHOLD), pop_agg, 0)

                results.setdefault(state, {}).setdefault(
                    year,
                    {"weighted_pm25": [], "smoke_days": []},
                )
                results[state][year]["weighted_pm25"].append(np.nansum(weighted_pm25))
                results[state][year]["smoke_days"].append(np.nansum(smoke_days))

        ds_all.close()
        ds_rmv.close()

    summary = {}

    for state in results:
        summary[state] = {"multi_year": {}, "annual": {}}
        total_pop = np.nansum(np.where(masks[state], pop_agg, 0))

        for year in years:
            if set(months).issubset(set(range(1, 5))):
                days_count = 119
            elif set(months).issubset(set(range(5, 13))):
                days_count = 246
            else:
                days_count = len(results[state][year]["weighted_pm25"])

            annual_avg_pm25 = (
                sum(results[state][year]["weighted_pm25"]) / (total_pop * days_count)
            )
            annual_person_days = sum(results[state][year]["smoke_days"])

            summary[state]["annual"][year] = {
                "avg_pm25": annual_avg_pm25,
                "person_days": annual_person_days,
            }

        summary[state]["multi_year"]["avg_pm25"] = np.mean([
            summary[state]["annual"][year]["avg_pm25"] for year in years
        ])
        summary[state]["multi_year"]["person_days"] = np.mean([
            summary[state]["annual"][year]["person_days"] for year in years
        ])

    return summary


def plot_population_map(cmaq_lon, cmaq_lat, gdf_se, pop_aggregated, mask_se):
    """Plot Figure 2c population map and inset donut chart with original settings."""
    below_threshold_color = "#e6e4e6"
    pop_masked_adjusted = np.ma.masked_where(~mask_se, pop_aggregated)

    cmap = cc.cm.CET_L17
    cmap.set_under(below_threshold_color)
    norm = LogNorm(vmin=1, vmax=100000)

    fig, ax = plt.subplots(
        1,
        figsize=(6, 5),
        dpi=600,
        subplot_kw={"projection": MAP_PROJ},
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_facecolor("#e6e4e6")
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.axis("off")

    ax.add_geometries(
        gdf_se.geometry,
        crs=ccrs.PlateCarree(),
        facecolor=below_threshold_color,
        edgecolor="k",
        linewidth=0,
        zorder=1,
    )

    im = ax.pcolormesh(
        cmaq_lon,
        cmaq_lat,
        pop_masked_adjusted,
        cmap=cmap,
        norm=norm,
        alpha=0.75,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    ax.add_geometries(
        gdf_se.geometry,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="k",
        linewidth=1.2,
        zorder=3,
    )

    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.4, pad=0.05, extend="both")
    log_ticks = [1, 10, 100, 1000, 10000, 100000]
    cbar.set_ticks(log_ticks)
    cbar.set_label("Per-pixel population count\n(at a 12-km level)", fontsize=7)
    for label in cbar.ax.xaxis.get_ticklabels():
        label.set_fontsize(7)

    total_fl = 21538187
    total_ga = 10711908
    total_sc = 5118425
    state_pop = [total_ga, total_fl, total_sc]
    state_labels = ["GA", "FL", "SC"]
    state_colors = ["#B983FF", "#CFF800", "#FCD307"]

    inset_ax = fig.add_axes([0.29, 0.3, 0.2, 0.2])
    inset_ax.pie(
        state_pop,
        labels=state_labels,
        colors=state_colors,
        wedgeprops=dict(width=0.35, edgecolor="w"),
        textprops=dict(fontsize=7),
        startangle=90,
    )
    inset_ax.axis("equal")
    inset_ax.text(
        0,
        0,
        "Total\n\nPopulation\n\nShare",
        ha="center",
        va="center",
        fontsize=6,
        fontweight="bold",
    )

    out_png = os.path.join(DIR_FIG, "reaggregated_pop_12km.png")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(fig)
    print(f"Saved: {out_png}")


def plot_lollipop_by_category_h(data_dict, title, xlabel, colors, states, save_path, scale_factor=1.0, adjust_sc_low=False):
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

    for st in states:
        x_vals = [data_dict[cat][st] / scale_factor for cat in categories]
    
        for i, cat in enumerate(categories):
            size = 250
            if adjust_sc_low and cat == 'low' and st == 'SC':
                size = 50
    
            ax.scatter(
                x_vals[i],
                y_idx[i],
                s=size,
                label=st if i == 0 else None,
                color=colors[st],
                edgecolors='#F9F9F9',
                linewidths=1.2,
                zorder=3
            )

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


def run_figure_2cde(cmaq_lon, cmaq_lat, gdf_se, masks):
    """Run Figure 2c-e population and exposure plotting."""
    lat_vals, lon_vals, pop_data = load_population(POP_TIF)
    pop_aggregated = aggregate_population(lat_vals, lon_vals, pop_data, cmaq_lat, cmaq_lon)

    plot_population_map(cmaq_lon, cmaq_lat, gdf_se, pop_aggregated, masks["SE"])

    total_orig = np.sum(pop_data)
    total_regrid = np.sum(pop_aggregated)
    print(f"Original total: {total_orig:,.0f}")
    print(f"Regridded total: {total_regrid:,.0f}")
    print(f"Conservation ratio: {total_regrid / total_orig:.4f}")

    state_masks = {state: masks[state] for state in SE_STATES}

    results_all_year = compute_exposure(YEARS, range(1, 13), state_masks, pop_aggregated)
    results_high = compute_exposure(YEARS, range(1, 5), state_masks, pop_aggregated)
    results_low = compute_exposure(YEARS, range(5, 13), state_masks, pop_aggregated)

    colors = {"FL": "#CFF800", "GA": "#B983FF", "SC": "#FCD307"}
    states = ["FL", "GA", "SC"]

    all_season = {st: results_all_year[st]["multi_year"]["avg_pm25"] for st in states}
    high_burn = {st: results_high[st]["multi_year"]["avg_pm25"] for st in states}
    low_burn = {st: results_low[st]["multi_year"]["avg_pm25"] for st in states}

    all_season_pd = {st: results_all_year[st]["multi_year"]["person_days"] for st in states}
    high_burn_pd = {st: results_high[st]["multi_year"]["person_days"] for st in states}
    low_burn_pd = {st: results_low[st]["multi_year"]["person_days"] for st in states}

    plot_lollipop_by_category_h(
        data_dict={"all": all_season, "high": high_burn, "low": low_burn},
        title="$E_{\\mathrm{annual}}^{\\mathrm{Rx}}$",
        xlabel="(in µg/m³)",
        colors=colors,
        states=states,
        save_path=os.path.join(DIR_FIG, "lollipop_h_PM25.png"),
        scale_factor=1.0,
        adjust_sc_low=True
    )

    plot_lollipop_by_category_h(
        data_dict={"all": all_season_pd, "high": high_burn_pd, "low": low_burn_pd},
        title="$PD_{\\mathrm{annual}}^{\\mathrm{Rx}}$",
        xlabel="(in millions)",
        colors=colors,
        states=states,
        save_path=os.path.join(DIR_FIG, "lollipop_h_PersonDays.png"),
        scale_factor=1e6,
    )


# =============================================================================
# Main workflow
# =============================================================================

def main():
    """Generate all Figure 2 panels and supporting distribution plots."""
    os.makedirs(DIR_FIG, exist_ok=True)
    os.chdir(DIR_FIG)

    configure_matplotlib()

    print(f"cwd is {os.getcwd()}")

    cmaq_lon, cmaq_lat = load_cmaq_grid(METCRO2D_FILE)
    _, gdf_se, gdf_fl, gdf_ga, gdf_sc = load_state_geometries()
    masks = build_masks(cmaq_lon, cmaq_lat, gdf_se, gdf_fl, gdf_ga, gdf_sc)

    run_figure_2ab(cmaq_lon, cmaq_lat, gdf_se, masks)
    run_figure_2cde(cmaq_lon, cmaq_lat, gdf_se, masks)

    print("All Figure 2 panels and distribution plots are complete.")


if __name__ == "__main__":
    main()