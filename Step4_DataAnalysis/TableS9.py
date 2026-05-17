# -*- coding: utf-8 -*-
"""
TableS9_regression_stats_emissions.py
====================================
Compute regression statistics comparing Permits emissions against NEI
and FINN for each species (PM2.5, CO, CO2, NOx, NH3, SO2) at both
yearly and daily grid-cell resolution.

Comparisons
-----------
1. Permits vs NEI  (Jan-Apr)  : yearly + daily,  all 6 species
2. Permits vs FINN (Jan-Apr)  : yearly + daily,  all 6 species
3. Permits vs NEI  (Full Year): yearly + daily,  all 6 species

Model: Permits (y) = slope * Other (x), OLS through origin.
Output: printed table + CSV file.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================================
# PATHS
# ===========================================================================
BASE_DIR = "/home/jh94030/scripts/python/postdoc_project/rxfire"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figure")

DIR_SCRIPTS = os.path.join(BASE_DIR, "analysis")
sys.path.append(os.path.join(DIR_SCRIPTS, "step3_BurnDataSelection"))
from util import CMAQGrid2D  # noqa: E402

PERMIT_EMIS_TEMPLATE = os.path.join(
    DATA_DIR, "SE_permit_data_2010-2020/output_emis",
    "SE_Combined_Permit_lf_3states_rx_{}.csv",
)
NEI_TEMPLATE = os.path.join(
    DATA_DIR, "oth_fire_inv/NEI_rxf_inv", "SE_Combined_NEI_rx_3states_{}.csv",
)
FINN_TEMPLATE = os.path.join(
    DATA_DIR, "oth_fire_inv/FINN_rxf_inv", "SE_Combined_FINN_rx_wf_{}_Jan-Apr.csv",
)

SEFM_GDB_PATH  = os.path.join("/work/chflab/jthuang/breadcrumbs", "SEFM", "SEFM_L_ABA_1994_2024_polys.gdb")

STATES_SHP = (
    "/work/chflab/jthuang/breadcrumbs/mapping_state/"
    "cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
)
METCRO2D_FILE = (
    "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/"
    "mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc"
)

# ===========================================================================
# CONFIG
# ===========================================================================
YEARS      = [2017, 2018, 2019]
SE_ST_ABBR = ["FL", "GA", "SC"]

# Species mapping: common key -> (Permits col, NEI col, FINN col)
SPECIES_MAP = {
    "PM25": ("PM2.5",  "PM2_5",   "PM25"),
    "CO":   ("CO",     "CO",      "CO"),
    "CO2":  ("CO2",    "CO2",     "CO2"),
    "NOx":  ("NOx",    "NOX",     "NOXasNO"),
    "NH3":  ("NH3",    "NH3",     "NH3"),
    "SO2":  ("SO2",    "SO2",     "SO2"),
}

# Column name references per inventory
# Permits (bluesky): latitude, longitude, date
# NEI:               latitude, longitude, DATE
# FINN:              LATI, LONGI, DAY


# ===========================================================================
# GRID HELPERS  (same as Table_regression_stats.py)
# ===========================================================================
def load_cmaq_grid(metcro_file):
    info = CMAQGrid2D(metcro_file)
    lon, lat = info["Lon"], info["Lat"]
    nrows, ncols = lat.shape
    return lon, lat, nrows, ncols


def grid_kdtree(lat_grid, lon_grid):
    grid_pts = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    return cKDTree(grid_pts)


def build_state_mask(lon_grid, lat_grid, states_gdf):
    states_ll = states_gdf.to_crs(epsg=4326)
    union_geom = unary_union(states_ll.geometry.values)
    pts = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    mask_flat = np.fromiter(
        (union_geom.contains(Point(xy)) or union_geom.touches(Point(xy))
         for xy in pts),
        dtype=bool, count=pts.shape[0])
    return mask_flat.reshape(lon_grid.shape)


# ===========================================================================
# REGRID HELPER  (KDTree snap to nearest cell)
# ===========================================================================
def yearly_emission_grids(df, lat_col, lon_col, emis_col,
                          years, tree, nrows, ncols):
    """Sum emissions per grid cell per year -> dict {year: 2D array}."""
    df = df.copy().dropna(subset=[lat_col, lon_col, emis_col])
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols
    grids = {}
    for yr in years:
        grid = np.zeros((nrows, ncols))
        sub = df[df["YEAR"] == yr]
        if not sub.empty:
            grouped = sub.groupby(["ROW", "COL"], observed=True)[emis_col].sum()
            for (r, c), val in grouped.items():
                grid[r, c] = val
        grids[yr] = grid
    return grids


def daily_emission_grids(df, lat_col, lon_col, emis_col, date_col,
                         tree, nrows, ncols):
    """Sum emissions per grid cell per day -> dict {date: 2D array}."""
    df = df.copy().dropna(subset=[lat_col, lon_col, emis_col])
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols
    df["_date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    grids = {}
    for dt, sub in df.groupby("_date"):
        grid = np.zeros((nrows, ncols))
        grouped = sub.groupby(["ROW", "COL"], observed=True)[emis_col].sum()
        for (r, c), val in grouped.items():
            grid[r, c] = val
        grids[dt] = grid
    return grids


# ===========================================================================
# DATA LOADERS
# ===========================================================================
def load_permits_emis(template, years, jan_apr_only=False):
    """Load bluesky Permits emission CSVs."""
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=["date"])
        df["YEAR"] = yr
        if jan_apr_only:
            df = df[df["date"].dt.month.between(1, 4)]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    tag = "Jan-Apr" if jan_apr_only else "Full Year"
    print(f"  Permits ({tag}): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_nei(template, years, jan_apr_only=False):
    """Load NEI CSVs."""
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=["DATE"])
        df["YEAR"] = yr
        if jan_apr_only:
            df = df[df["DATE"].dt.month.between(1, 4)]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    tag = "Jan-Apr" if jan_apr_only else "Full Year"
    print(f"  NEI ({tag}): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_finn(template, years):
    """Load FINN CSVs (already Jan-Apr only)."""
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=["DAY"])
        df["YEAR"] = yr
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  FINN (Jan-Apr): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


# ===========================================================================
# REGRESSION STATISTICS  (same model as Table_regression_stats.py)
# ===========================================================================
def yearly_regression_stats(permit_grids, other_grids, state_mask, years,
                            label, species, period="Jan-Apr"):
    """
    OLS regression (no intercept) for yearly grid-cell comparison.
    Permits (y) vs other inventory (x).
    Only cells where BOTH inventories have > 0 are included.
    """
    permit_vals, other_vals = [], []
    for yr in years:
        p = permit_grids[yr][state_mask]
        o = other_grids[yr][state_mask]
        mask = (p > 0) & (o > 0)
        permit_vals.append(p[mask])
        other_vals.append(o[mask])

    permit_all = np.concatenate(permit_vals)
    other_all  = np.concatenate(other_vals)

    if len(permit_all) == 0:
        print(f"  WARNING: no matching cells for {species} "
              f"Permits vs {label} ({period}, yearly)")
        return None

    X = other_all.reshape(-1, 1)
    y = permit_all
    model   = sm.OLS(y, X)
    results = model.fit()
    cf      = results.conf_int(alpha=0.05)
    slope   = results.params[0]
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    r2  = reg.score(X, y)

    return dict(
        Comparison=f"Permits vs {label}",
        Species=species,
        Period=period,
        Temporal="Yearly",
        N=len(permit_all),
        Slope=round(slope, 4),
        Slope_CI_lo=round(cf[0, 0], 4),
        Slope_CI_hi=round(cf[0, 1], 4),
        R2=round(r2, 4),
    )


def daily_regression_stats(permit_daily, other_daily, state_mask,
                           label, species, period="Jan-Apr"):
    """
    OLS regression (no intercept) for daily grid-cell comparison.
    Permits (y) vs other inventory (x).
    Only cells where BOTH inventories have > 0 on the same day are included.
    """
    permit_dates = set(permit_daily.keys())
    other_dates  = set(other_daily.keys())
    common_dates = sorted(permit_dates & other_dates)

    permit_vals, other_vals = [], []
    for dt in common_dates:
        p = permit_daily[dt][state_mask]
        o = other_daily[dt][state_mask]
        mask = (p > 0) & (o > 0)
        if mask.any():
            permit_vals.append(p[mask])
            other_vals.append(o[mask])

    if not permit_vals:
        print(f"  WARNING: no daily matching cells for {species} "
              f"Permits vs {label} ({period})")
        return None

    permit_all = np.concatenate(permit_vals)
    other_all  = np.concatenate(other_vals)

    X = other_all.reshape(-1, 1)
    y = permit_all
    model   = sm.OLS(y, X)
    results = model.fit()
    cf      = results.conf_int(alpha=0.05)
    slope   = results.params[0]
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    r2  = reg.score(X, y)

    return dict(
        Comparison=f"Permits vs {label}",
        Species=species,
        Period=period,
        Temporal="Daily",
        N=len(permit_all),
        Common_dates=len(common_dates),
        Slope=round(slope, 4),
        Slope_CI_lo=round(cf[0, 0], 4),
        Slope_CI_hi=round(cf[0, 1], 4),
        R2=round(r2, 4),
    )


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":

    # 1. Load CMAQ grid
    print("=" * 70)
    print("Loading CMAQ grid ...")
    cmaq_lon, cmaq_lat, nrows, ncols = load_cmaq_grid(METCRO2D_FILE)
    tree = grid_kdtree(cmaq_lat, cmaq_lon)

    # 2. State mask
    print("Loading state boundaries ...")
    gdf_states_all = gpd.read_file(STATES_SHP)
    gdf_SE = gdf_states_all[gdf_states_all["STUSPS"].isin(SE_ST_ABBR)]
    state_mask = build_state_mask(cmaq_lon, cmaq_lat, gdf_SE)

    # ==================================================================
    # 3. Load raw data
    # ==================================================================
    print("\n" + "=" * 70)
    print("Loading data ...")

    # Jan-Apr
    permits_ja = load_permits_emis(PERMIT_EMIS_TEMPLATE, YEARS, jan_apr_only=True)
    nei_ja     = load_nei(NEI_TEMPLATE, YEARS, jan_apr_only=True)
    finn_ja    = load_finn(FINN_TEMPLATE, YEARS)

    # Full year
    permits_fy = load_permits_emis(PERMIT_EMIS_TEMPLATE, YEARS, jan_apr_only=False)
    nei_fy     = load_nei(NEI_TEMPLATE, YEARS, jan_apr_only=False)

    # ==================================================================
    # 4. Regrid & compute regression for every species
    # ==================================================================
    stat_rows = []

    for sp_key, (pcol, ncol, fcol) in SPECIES_MAP.items():
        print(f"\n{'=' * 70}")
        print(f"  Species: {sp_key}  "
              f"(Permits={pcol}, NEI={ncol}, FINN={fcol})")
        print("=" * 70)

        # --------------------------------------------------------------
        # Jan-Apr : Yearly
        # --------------------------------------------------------------
        print("  Computing yearly grids (Jan-Apr) ...")
        p_yr_ja = yearly_emission_grids(
            permits_ja, "latitude", "longitude", pcol,
            YEARS, tree, nrows, ncols)
        n_yr_ja = yearly_emission_grids(
            nei_ja, "latitude", "longitude", ncol,
            YEARS, tree, nrows, ncols)
        f_yr_ja = yearly_emission_grids(
            finn_ja, "LATI", "LONGI", fcol,
            YEARS, tree, nrows, ncols)

        row = yearly_regression_stats(
            p_yr_ja, n_yr_ja, state_mask, YEARS,
            "NEI", sp_key, period="Jan-Apr")
        if row:
            stat_rows.append(row)

        row = yearly_regression_stats(
            p_yr_ja, f_yr_ja, state_mask, YEARS,
            "FINN", sp_key, period="Jan-Apr")
        if row:
            stat_rows.append(row)

        # --------------------------------------------------------------
        # Jan-Apr : Daily
        # --------------------------------------------------------------
        print("  Computing daily grids (Jan-Apr) ...")
        p_dy_ja = daily_emission_grids(
            permits_ja, "latitude", "longitude", pcol, "date",
            tree, nrows, ncols)
        n_dy_ja = daily_emission_grids(
            nei_ja, "latitude", "longitude", ncol, "DATE",
            tree, nrows, ncols)
        f_dy_ja = daily_emission_grids(
            finn_ja, "LATI", "LONGI", fcol, "DAY",
            tree, nrows, ncols)

        row = daily_regression_stats(
            p_dy_ja, n_dy_ja, state_mask,
            "NEI", sp_key, period="Jan-Apr")
        if row:
            stat_rows.append(row)

        row = daily_regression_stats(
            p_dy_ja, f_dy_ja, state_mask,
            "FINN", sp_key, period="Jan-Apr")
        if row:
            stat_rows.append(row)

        # --------------------------------------------------------------
        # Full Year : Permits vs NEI (yearly + daily)
        # --------------------------------------------------------------
        print("  Computing yearly grids (Full Year) ...")
        p_yr_fy = yearly_emission_grids(
            permits_fy, "latitude", "longitude", pcol,
            YEARS, tree, nrows, ncols)
        n_yr_fy = yearly_emission_grids(
            nei_fy, "latitude", "longitude", ncol,
            YEARS, tree, nrows, ncols)

        row = yearly_regression_stats(
            p_yr_fy, n_yr_fy, state_mask, YEARS,
            "NEI", sp_key, period="Full Year")
        if row:
            stat_rows.append(row)

        print("  Computing daily grids (Full Year) ...")
        p_dy_fy = daily_emission_grids(
            permits_fy, "latitude", "longitude", pcol, "date",
            tree, nrows, ncols)
        n_dy_fy = daily_emission_grids(
            nei_fy, "latitude", "longitude", ncol, "DATE",
            tree, nrows, ncols)

        row = daily_regression_stats(
            p_dy_fy, n_dy_fy, state_mask,
            "NEI", sp_key, period="Full Year")
        if row:
            stat_rows.append(row)

    # ==================================================================
    # 5. Assemble & output table
    # ==================================================================
    df_stats = pd.DataFrame(stat_rows)

    col_order = ["Comparison", "Species", "Period", "Temporal", "N",
                 "Common_dates", "Slope", "Slope_CI_lo", "Slope_CI_hi", "R2"]
    col_order = [c for c in col_order if c in df_stats.columns]
    df_stats = df_stats[col_order]

    print("\n" + "=" * 70)
    print("  Grid-cell regression: Permits (y) = slope * Other (x), "
          "no intercept")
    print("  Slope 95% CI in brackets. "
          "R² from sklearn (uncentered, no intercept).")
    print()
    print(df_stats.to_string(index=False))

    csv_out = os.path.join(OUT_DIR,
                           "regression_stats_emissions_permits_vs_inventories.csv")
    df_stats.to_csv(csv_out, index=False)
    print(f"\n  Wrote -> {csv_out}")

    print("\n" + "=" * 70)
    print("=== Done ===")