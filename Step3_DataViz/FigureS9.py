# -*- coding: utf-8 -*-
###############################################################################
# FigureS9.py
#
# Author: Jingting HUANG
#
# Purpose
# -------
# This script generates the Figure S9 inventory-comparison products for Rx-fire
# emissions over Florida (FL), Georgia (GA), and South Carolina (SC) on the CMAQ
# 12-km grid.
#
# The workflow combines two related plotting tasks:
#
#   1. Spatial inventory comparison maps for Jan-Apr emissions:
#        - GEUFE:   Jan-Apr 2020
#        - FINN:    Jan-Apr 2017-2019
#        - Permits: Jan-Apr 2017-2019
#        - NEI:     Jan-Apr 2017-2019
#
#      For each species and inventory, the script computes mean annual
#      emissions in Mg km-2 yr-1 and saves one spatial map per species-inventory
#      pair.
#
#   2. Monthly-mean daily emission time series:
#        - GEUFE:   Aug-Dec 2019
#        - FINN:    Jan-Apr 2017-2019
#        - Permits: Jan-Dec 2017-2019
#        - NEI:     Jan-Dec 2017-2019
#
#      For each species and region, the script computes monthly-mean daily
#      emissions in Mg day-1 and saves one time-series plot.
###############################################################################

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as path_effects
from matplotlib import font_manager, ticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

from pyproj import Transformer
from shapely.geometry import Point


# =============================================================================
# PATHS AND CORE SETTINGS
# =============================================================================

METCRO2D_FILE = os.path.join(
    "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/"
    "mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc"
)

DATA_ROOT = os.path.join(
    "/home/jh94030/scripts/python/postdoc_project/rxfire/data",
    "gridded_CMAQ_12US1",
)

GEUFE_DIR = os.path.join(DATA_ROOT, "GEUFE")
FINN_DIR = os.path.join(DATA_ROOT, "FINN")
PERMIT_DIR = os.path.join(DATA_ROOT, "Permit")
NEI_DIR = os.path.join(DATA_ROOT, "NEI")

PLOT_DIR = r"/home/jh94030/scripts/python/postdoc_project/rxfire/figure"

STATES_SHP = os.path.join(
    "/work/chflab/jthuang/breadcrumbs/mapping_state/"
    "cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
)

YEARS = [2017, 2018, 2019]
GEUFE_SPATIAL_YEARS = [2020]

SE_ST_ABBR = ["FL", "GA", "SC"]
INVENTORY_LIST = ["GEUFE", "FINN", "Permits", "NEI"]
SPECIES_LIST = ["PM25", "CO", "CO2", "NOx", "NH3", "SO2"]

VARMAP = {
    "PM25": {"GEUFE": "PM25", "FINN": "PM25", "Permits": "PM2.5", "NEI": "PM2_5"},
    "CO":   {"GEUFE": "CO",   "FINN": "CO",      "Permits": "CO",    "NEI": "CO"},
    "CO2":  {"GEUFE": "CO2",  "FINN": "CO2",     "Permits": "CO2",   "NEI": "CO2"},
    "NOx":  {"GEUFE": "NOx",  "FINN": "NOXasNO", "Permits": "NOx",   "NEI": "NOX"},
    "NH3":  {"GEUFE": "NH3",  "FINN": "NH3",     "Permits": "NH3",   "NEI": "NH3"},
    "SO2":  {"GEUFE": "SO2",  "FINN": "SO2",     "Permits": "SO2",   "NEI": "SO2"},
}

SPECIES_DISPLAY = {
    "PM25": r"PM$_{\mathregular{2.5}}$",
    "PM10": r"PM$_{\mathregular{10}}$",
    "CO": "CO",
    "CO2": r"CO$_{\mathregular{2}}$",
    "NOx": r"NO$_{\mathregular{x}}$",
    "NH3": r"NH$_{\mathregular{3}}$",
    "SO2": r"SO$_{\mathregular{2}}$",
    "OC": "OC",
    "BC": "BC",
}

DIR_PREFIX = {
    "GEUFE": (GEUFE_DIR, "GEUFE_CMAQ12US1"),
    "FINN": (FINN_DIR, "FINN_CMAQ12US1"),
    "Permits": (PERMIT_DIR, "Permit_CMAQ12US1"),
    "NEI": (NEI_DIR, "NEI_CMAQ12US1"),
}

REGIONS = {
    "FL": ["FL"],
    "GA": ["GA"],
    "SC": ["SC"],
    "FL+GA+SC": ["FL", "GA", "SC"],
}

REGION_FULLNAME = {
    "FL": "FL",
    "GA": "GA",
    "SC": "SC",
    "FL+GA+SC": "Full Region",
}

COLOR_DICT = {
    "GEUFE": "darkorange",
    "FINN": "forestgreen",
    "Permits": "firebrick",
    "NEI": "navy",
}

CITIES = {
    "Atlanta": (33.7490, -84.3880),
    "Orlando": (28.5383, -81.3792),
    "Tallahassee": (30.4383, -84.2807),
    "Columbia": (34.0007, -81.0348),
    "Jacksonville": (30.3322, -81.6557),
    "Savannah": (32.0809, -81.0912),
    "Pensacola": (30.4213, -87.2169),
    "Tampa": (27.9506, -82.4572),
    "Miami": (25.7617, -80.1918),
    "Columbus": (32.4600, -84.9877),
    "Albany": (31.5785, -84.1557),
    "Charleston": (32.7765, -79.9311),
}

STATE_LABELS = [
    ("FL", (-84.5, 28.5)),
    ("GA", (-87.0, 32.5)),
    ("SC", (-78.5, 34.2)),
]

CMAP_MATTER = cmocean.cm.matter


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def setup_fonts():
    """Use Arial fonts, matching the original plotting scripts."""
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]


def read_cmaq_grid(metcro_path):
    """
    Read 2-D CMAQ grid-cell centers and cell area from METCRO2D.

    Returns
    -------
    lon2d, lat2d : 2-D arrays
    cell_area_m2 : float
    extent : tuple
        Map extent derived using the same offsets as the original spatial script.
    """
    ds = nc.Dataset(metcro_path)
    p_alp = float(ds.getncattr("P_ALP"))
    p_bet = float(ds.getncattr("P_BET"))
    xcent = float(ds.getncattr("XCENT"))
    ycent = float(ds.getncattr("YCENT"))
    xorig = float(ds.getncattr("XORIG"))
    yorig = float(ds.getncattr("YORIG"))
    xcell = float(ds.getncattr("XCELL"))
    ycell = float(ds.getncattr("YCELL"))
    ncols = int(ds.getncattr("NCOLS"))
    nrows = int(ds.getncattr("NROWS"))
    ds.close()

    proj4_lcc = (
        f"+proj=lcc +a=6370000.0 +b=6370000.0 "
        f"+lat_1={p_alp} +lat_2={p_bet} "
        f"+lat_0={ycent} +lon_0={xcent} "
        f"+x_0=0 +y_0=0 +units=m +no_defs"
    )

    transformer = Transformer.from_proj(proj4_lcc, "epsg:4326", always_xy=True)

    x_centers = np.linspace(
        xorig + xcell / 2,
        xorig + xcell / 2 + xcell * (ncols - 1),
        ncols,
    )
    y_centers = np.linspace(
        yorig + ycell / 2,
        yorig + ycell / 2 + ycell * (nrows - 1),
        nrows,
    )

    x2d, y2d = np.meshgrid(x_centers, y_centers)
    lon2d, lat2d = transformer.transform(x2d, y2d)

    cell_area_m2 = xcell * ycell

    bot_left_lat = lat2d[0, 0] + 2
    bot_left_lon = lon2d[0, 0] + 32
    top_right_lat = lat2d[-1, -1] - 15
    top_right_lon = lon2d[-1, -1] - 22
    extent = (bot_left_lon, top_right_lon, bot_left_lat, top_right_lat)

    return lon2d, lat2d, cell_area_m2, extent


def read_se_states():
    """Load state boundaries and return the full and three-state GeoDataFrames."""
    all_states = gpd.read_file(STATES_SHP)
    gdf_se = all_states[all_states["STUSPS"].isin(SE_ST_ABBR)].copy()
    return all_states, gdf_se


def dates_by_month_range(years, start_month, end_month):
    """Build a daily DatetimeIndex for the same month range in each year."""
    dates = []
    for year in years:
        start = f"{year}-{start_month:02d}-01"
        end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        dates.extend(pd.date_range(start, end, freq="D").tolist())
    return pd.DatetimeIndex(dates)


def jan_apr_dates(years):
    """Daily dates for January-April."""
    return dates_by_month_range(years, 1, 4)


def jan_dec_dates(years):
    """Daily dates for January-December."""
    return dates_by_month_range(years, 1, 12)


def geufe_timeseries_dates():
    """Daily dates for GEUFE monthly time series, matching the original script."""
    return pd.date_range("2019-08-01", "2019-12-31", freq="D")


def netcdf_path(inv_dir, prefix, date):
    """Daily CMAQ-gridded emission file path."""
    return os.path.join(inv_dir, f"{prefix}_{date.strftime('%Y%m%d')}.nc")


# =============================================================================
# SPATIAL INVENTORY COMPARISON: FIGURE S9A
# =============================================================================

def load_inventory_arrays(inv_dir, prefix, dates, species_map):
    """
    Load daily gridded arrays for all requested species in one file pass.

    Returns
    -------
    result : dict
        {species: ndarray(time, row, col) or None}
    valid_dates : list[pd.Timestamp]
        Dates with at least one requested species available.
    """
    result = {species: [] for species in species_map}
    valid_dates = []

    for date in dates:
        fpath = netcdf_path(inv_dir, prefix, date)
        if not os.path.isfile(fpath):
            continue

        ds = nc.Dataset(fpath, "r")
        found_any = False
        day_data = {}

        for species, varname in species_map.items():
            if varname in ds.variables:
                day_data[species] = np.array(ds.variables[varname][:].squeeze())
                found_any = True
            else:
                day_data[species] = None

        ds.close()

        if found_any:
            valid_dates.append(date)
            for species in species_map:
                result[species].append(day_data[species])

    for species in species_map:
        arrays = [arr for arr in result[species] if arr is not None]
        result[species] = np.stack(arrays, axis=0) if arrays else None

    return result, valid_dates


def compute_spatial_annual_sum(data_3d):
    """
    Convert kg m-2 s-1 to Mg km-2 day-1 and sum over available days.

    The multi-year mean division is handled outside this function, matching the
    original Figure S9a workflow.
    """
    sec2day = 86400.0
    m2_to_km2 = 1e6
    kg_to_mg = 1e-3

    daily = data_3d * sec2day * m2_to_km2 * kg_to_mg
    return daily.sum(axis=0)


def mask_zeros(arr):
    """Mask exact zeroes as NaN for spatial plotting."""
    out = arr.copy().astype(float)
    out[out == 0] = np.nan
    return out


def add_basemap(ax, gdf_se, extent, add_cities=True):
    """Draw the shared Figure S9 spatial basemap."""
    bl_lon, tr_lon, bl_lat, tr_lat = extent
    ax.set_extent([bl_lon, tr_lon, bl_lat, tr_lat], crs=ccrs.PlateCarree())
    ax.axis("off")

    se_geom = gdf_se.to_crs(epsg=4326).geometry
    ax.add_geometries(
        se_geom,
        crs=ccrs.PlateCarree(),
        facecolor="#e6e4e6",
        edgecolor="k",
        linewidth=0,
        zorder=2,
    )
    ax.add_geometries(
        se_geom,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="k",
        linewidth=1.2,
        zorder=3,
    )

    for txt, xy in STATE_LABELS:
        label = ax.text(
            xy[0], xy[1], txt,
            color="k",
            fontweight="bold",
            fontsize=16,
            transform=ccrs.Geodetic(),
        )
        label.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground="white"),
            path_effects.Normal(),
        ])

    if add_cities:
        for name, (lat, lon) in CITIES.items():
            ax.scatter(
                lon, lat,
                marker="o",
                facecolor="none",
                edgecolor="k",
                s=10,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            ax.text(
                lon + 0.15,
                lat - 0.1,
                name,
                fontsize=7,
                fontweight="bold",
                color="black",
                transform=ccrs.PlateCarree(),
                zorder=3,
                path_effects=[
                    path_effects.Stroke(linewidth=1, foreground="white"),
                    path_effects.Normal(),
                ],
            )


def plot_spatial_inventory_map(
    arr,
    lon2d,
    lat2d,
    cell_area_m2,
    inv_name,
    species,
    extent,
    gdf_se,
    vmin,
    vmax,
    cbar_ticks,
    savename,
):
    """Plot one Figure S9a spatial map, preserving the original style."""
    projection = ccrs.AlbersEqualArea(central_longitude=-84, central_latitude=30)

    fig, ax = plt.subplots(
        1,
        figsize=(7, 5.25),
        dpi=600,
        subplot_kw={"projection": projection},
    )

    add_basemap(ax, gdf_se, extent, add_cities=True)

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        mask_zeros(arr),
        cmap=CMAP_MATTER,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    ax.add_feature(
        cfeature.LAKES,
        facecolor="w",
        edgecolor="k",
        linewidth=0.5,
        zorder=2,
    )

    cell_area_km2 = cell_area_m2 / 1e6
    total_mg = np.nansum(arr * cell_area_km2)
    total_tg = total_mg / 1e6

    ax.set_title(
        f"{inv_name}\n\nTotal: {total_tg:.4f} Tg",
        fontweight="bold",
        fontsize=14,
    )

    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        shrink=0.6,
        pad=0.02,
        extend="max",
    )
    species_label = SPECIES_DISPLAY.get(species, species)
    unit = r"Mg $\mathregular{km^{-2}}$ $\mathregular{yr^{-1}}$"
    cbar.set_label(label=f"{species_label} Emissions ({unit})", size=11)

    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_yticklabels([f"{tick:1g}" for tick in cbar_ticks])

    cbar.ax.tick_params(labelsize=9)
    cbar.ax.tick_params(which="major", length=6, width=1.5)
    cbar.ax.tick_params(which="minor", length=3, width=1)
    cbar.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(3))

    plt.savefig(
        os.path.join(PLOT_DIR, savename),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close("all")
    print(f"    Saved {savename}")


def run_spatial_comparison(lon2d, lat2d, cell_area_m2, extent, gdf_se):
    """Generate the Figure S9a spatial inventory-comparison maps."""
    print("\n" + "=" * 70)
    print("Figure S9a: spatial inventory-comparison maps")

    spatial_dates = {
        "GEUFE": jan_apr_dates(GEUFE_SPATIAL_YEARS),
        "FINN": jan_apr_dates(YEARS),
        "Permits": jan_apr_dates(YEARS),
        "NEI": jan_apr_dates(YEARS),
    }

    all_data = {}
    all_valid_dates = {}

    for inv in INVENTORY_LIST:
        species_map = {
            species: VARMAP[species][inv]
            for species in SPECIES_LIST
            if VARMAP[species].get(inv) is not None
        }
        inv_dir, prefix = DIR_PREFIX[inv]
        dates = spatial_dates[inv]

        print(f"\nLoading {inv} ({len(dates)} dates, {len(species_map)} species) ...")
        data, valid_dates = load_inventory_arrays(inv_dir, prefix, dates, species_map)

        all_data[inv] = data
        all_valid_dates[inv] = valid_dates

        n_years = len(set(date.year for date in valid_dates)) if valid_dates else 0
        print(f"  {len(valid_dates)} valid days, {n_years} year(s)")

    for species in SPECIES_LIST:
        print(f"\n====== {species} spatial maps ======")
        annual_sums = {}

        for inv in INVENTORY_LIST:
            data_3d = all_data[inv].get(species)
            valid_dates = all_valid_dates[inv]

            if data_3d is None or len(valid_dates) == 0:
                annual_sums[inv] = None
                continue

            n_years = len(set(date.year for date in valid_dates))
            annual_sum = compute_spatial_annual_sum(data_3d)

            if n_years > 1:
                annual_sum = annual_sum / n_years

            annual_sums[inv] = annual_sum

        valid_arrays = [
            arr for arr in annual_sums.values()
            if arr is not None and np.any(arr > 0)
        ]
        data_max = float(max(np.nanmax(arr) for arr in valid_arrays)) if valid_arrays else 100.0

        loc = ticker.MaxNLocator(nbins=4, integer=True, min_n_ticks=5)
        ticks = loc.tick_values(0, data_max)[:5]
        vmin_uni = ticks[0]
        vmax_uni = ticks[-1]

        print(f"  Colorbar ticks: {ticks}  (vmin={vmin_uni}, vmax={vmax_uni})")

        for inv in INVENTORY_LIST:
            arr = annual_sums.get(inv)
            if arr is None:
                print(f"  {inv}: N/A for {species}")
                continue

            print(f"  Plotting {inv} - {species} ...")
            plot_spatial_inventory_map(
                arr=arr,
                lon2d=lon2d,
                lat2d=lat2d,
                cell_area_m2=cell_area_m2,
                inv_name=inv,
                species=species,
                extent=extent,
                gdf_se=gdf_se,
                vmin=vmin_uni,
                vmax=vmax_uni,
                cbar_ticks=ticks,
                savename=f"{species}_{inv}_AnnualSum.png",
            )


# =============================================================================
# MONTHLY TIME SERIES: FIGURE S9B
# =============================================================================

def build_state_masks(lon2d, lat2d, all_states, state_abbrs):
    """
    Create one boolean mask per state using CMAQ grid-cell centers.

    This preserves the original point-in-polygon masking strategy used for the
    monthly region-total time series.
    """
    nrows, ncols = lon2d.shape
    lons_flat = lon2d.ravel()
    lats_flat = lat2d.ravel()

    print("  Building grid point GeoDataFrame ...")
    points = gpd.GeoDataFrame(
        {"idx": np.arange(lons_flat.size)},
        geometry=[Point(lon, lat) for lon, lat in zip(lons_flat, lats_flat)],
        crs="EPSG:4326",
    )

    masks = {}
    for state in state_abbrs:
        print(f"  Masking {state} ...")
        state_geom = all_states[all_states["STUSPS"] == state].to_crs(epsg=4326)
        inside = gpd.sjoin(points, state_geom, predicate="within")

        mask = np.zeros(lons_flat.size, dtype=bool)
        mask[inside["idx"].values] = True
        masks[state] = mask.reshape(nrows, ncols)

        print(f"    {masks[state].sum()} cells inside {state}")

    return masks


def build_region_masks(state_masks):
    """Create individual-state masks and the combined three-state mask."""
    region_masks = {state: state_masks[state] for state in SE_ST_ABBR}
    region_masks["FL+GA+SC"] = (
        state_masks["FL"] | state_masks["GA"] | state_masks["SC"]
    )
    print(f'  FL+GA+SC combined: {region_masks["FL+GA+SC"].sum()} cells')
    return region_masks


def load_daily_region_totals(inv_dir, prefix, dates, species_map, cell_area_m2, region_masks):
    """
    Read daily emissions once per file and compute region totals.

    Input units are kg m-2 s-1. Output daily totals are kg day-1 for each
    species and region.
    """
    sec2day = 86400.0

    result = {
        species: {region: [] for region in region_masks}
        for species in species_map
    }
    valid_dates = []

    for date in dates:
        fpath = netcdf_path(inv_dir, prefix, date)
        if not os.path.isfile(fpath):
            continue

        ds = nc.Dataset(fpath, "r")
        found_any = False
        day_data = {}

        for species, varname in species_map.items():
            if varname in ds.variables:
                arr = np.array(ds.variables[varname][:].squeeze())
                day_data[species] = arr * sec2day * cell_area_m2
                found_any = True
            else:
                day_data[species] = None

        ds.close()

        if found_any:
            valid_dates.append(date)
            for species in species_map:
                arr = day_data.get(species)
                for region, mask in region_masks.items():
                    if arr is None:
                        result[species][region].append(0.0)
                    else:
                        result[species][region].append(float(np.nansum(arr[mask])))

    return result, valid_dates


def compute_monthly_mean(daily_values, valid_dates):
    """
    Convert daily totals in kg day-1 to monthly-mean daily rates in Mg day-1.
    """
    if len(daily_values) == 0 or len(valid_dates) == 0:
        return [], np.array([])

    df = pd.DataFrame({"date": valid_dates, "val": daily_values})
    df["ym"] = df["date"].dt.to_period("M")

    grouped = df.groupby("ym")["val"]
    monthly_sum = grouped.sum()
    monthly_ndays = grouped.count()

    monthly_mean_rate = (monthly_sum / monthly_ndays) / 1e3
    months = [period.to_timestamp() for period in monthly_mean_rate.index]

    return months, monthly_mean_rate.values


def split_finn_segments(months, values):
    """
    Split FINN Jan-Apr values by year so lines do not connect across missing
    May-Dec periods.
    """
    segments_m = []
    segments_v = []

    if len(months) == 0:
        return segments_m, segments_v

    cur_m = [months[0]]
    cur_v = [values[0]]

    for idx in range(1, len(months)):
        gap_days = (months[idx] - months[idx - 1]).days
        if gap_days > 35:
            segments_m.append(np.array(cur_m))
            segments_v.append(np.array(cur_v))
            cur_m = []
            cur_v = []

        cur_m.append(months[idx])
        cur_v.append(values[idx])

    segments_m.append(np.array(cur_m))
    segments_v.append(np.array(cur_v))

    return segments_m, segments_v


def plot_monthly_timeseries(monthly_data, species, region_key, savename):
    """Plot one Figure S9b monthly time-series panel."""
    fig, ax = plt.subplots(figsize=(7, 1), facecolor="white")

    for inv in INVENTORY_LIST:
        if inv not in monthly_data:
            continue

        months, values = monthly_data[inv]
        months = np.array(months)
        values = np.array(values, dtype=float)
        color = COLOR_DICT[inv]

        values[values <= 0] = 0.001

        if inv == "FINN":
            segment_months, segment_values = split_finn_segments(months, values)
            for idx, (seg_m, seg_v) in enumerate(zip(segment_months, segment_values)):
                ax.plot(
                    seg_m,
                    seg_v,
                    marker="o",
                    color=color,
                    label=inv if idx == 0 else "_nolegend_",
                    markersize=4,
                    linewidth=1.5,
                    zorder=10,
                )
        else:
            ax.plot(
                months,
                values,
                marker="o",
                color=color,
                label=inv,
                markersize=4,
                linewidth=1.5,
                zorder=10,
            )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlim(pd.Timestamp("2017-01-01"), pd.Timestamp("2019-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    fig.canvas.draw()
    new_labels = [label.get_text()[0] if label.get_text() else "" for label in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels)
    ax.tick_params(axis="x", which="minor", bottom=False)

    for year in YEARS:
        jan = pd.Timestamp(f"{year}-01-01")
        ax.annotate(
            str(year),
            xy=(jan, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    species_label = SPECIES_DISPLAY.get(species, species)
    region_label = REGION_FULLNAME.get(region_key, region_key)
    unit = r"Mg $\mathregular{day^{-1}}$"

    ax.set_ylabel(
        f"{species_label} Emissions ({unit})",
        fontsize=5,
        fontweight="bold",
    )
    ax.set_xlabel("Month", fontsize=10)
    ax.text(
        0.02,
        0.08,
        region_label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="y", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Permits", "NEI", "FINN", "GEUFE"]
    order_map = {label: (handle, label) for handle, label in zip(handles, labels)}
    ordered = [order_map[key] for key in desired_order if key in order_map]

    if ordered:
        handles, labels = zip(*ordered)
        ax.legend(
            handles,
            labels,
            ncol=len(handles),
            edgecolor="k",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.5),
            prop={"size": 9},
            columnspacing=0.8,
        )

    plt.savefig(
        os.path.join(PLOT_DIR, savename),
        facecolor="white",
        bbox_inches="tight",
        dpi=400,
    )
    plt.close("all")
    print(f"    Saved {savename}")


def run_monthly_timeseries(lon2d, lat2d, cell_area_m2, all_states):
    """Generate the Figure S9b monthly time-series plots."""
    print("\n" + "=" * 70)
    print("Figure S9b: monthly-mean daily emission time series")

    print("Building state and region masks ...")
    state_masks = build_state_masks(lon2d, lat2d, all_states, SE_ST_ABBR)
    region_masks = build_region_masks(state_masks)

    inv_config = {
        "GEUFE": (*DIR_PREFIX["GEUFE"], geufe_timeseries_dates()),
        "FINN": (*DIR_PREFIX["FINN"], jan_apr_dates(YEARS)),
        "Permits": (*DIR_PREFIX["Permits"], jan_dec_dates(YEARS)),
        "NEI": (*DIR_PREFIX["NEI"], jan_dec_dates(YEARS)),
    }

    all_daily = {}
    all_valid_dates = {}

    for inv in INVENTORY_LIST:
        species_map = {species: VARMAP[species][inv] for species in SPECIES_LIST}
        inv_dir, prefix, dates = inv_config[inv]

        print(
            f"\nLoading {inv} ({len(dates)} candidate dates, "
            f"{len(species_map)} species, {len(region_masks)} regions) ..."
        )

        data, valid_dates = load_daily_region_totals(
            inv_dir=inv_dir,
            prefix=prefix,
            dates=dates,
            species_map=species_map,
            cell_area_m2=cell_area_m2,
            region_masks=region_masks,
        )

        all_daily[inv] = data
        all_valid_dates[inv] = valid_dates

        n_years = len(set(date.year for date in valid_dates)) if valid_dates else 0
        print(f"  {len(valid_dates)} valid days, {n_years} year(s)")

    for species in SPECIES_LIST:
        print(f"\n====== {species} monthly time series ======")

        for region_key in REGIONS:
            monthly_data = {}

            for inv in INVENTORY_LIST:
                daily_values = all_daily[inv][species][region_key]
                valid_dates = all_valid_dates[inv]
                months, monthly_mean = compute_monthly_mean(daily_values, valid_dates)
                monthly_data[inv] = (months, monthly_mean)

            plot_monthly_timeseries(
                monthly_data=monthly_data,
                species=species,
                region_key=region_key,
                savename=f"{species}_{region_key}_MonthlyTimeseries.png",
            )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    setup_fonts()
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Reading CMAQ grid ...")
    lon2d, lat2d, cell_area_m2, map_extent = read_cmaq_grid(METCRO2D_FILE)
    cell_area_km2 = cell_area_m2 / 1e6
    nrows, ncols = lon2d.shape

    print(f"  Grid: {ncols} x {nrows},  cell area: {cell_area_km2:.2f} km²")
    print(f"  Map extent: {map_extent}")

    print("Loading state boundaries ...")
    all_states, gdf_se = read_se_states()
    print(f'  Loaded {len(gdf_se)} SE states: {list(gdf_se["STUSPS"])}')

    run_spatial_comparison(lon2d, lat2d, cell_area_m2, map_extent, gdf_se)
    run_monthly_timeseries(lon2d, lat2d, cell_area_m2, all_states)

    print("\n=== Figure S9 workflow complete ===")


if __name__ == "__main__":
    main()