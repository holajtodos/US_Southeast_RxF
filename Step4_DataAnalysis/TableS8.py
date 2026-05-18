#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TableS8_regression_stats_BA.py
=========================
Compute and output a regression statistics table comparing Permits
burned area against NEI, FINN, and SEFM at both yearly and daily
grid-cell resolution.

Model: Permits (y) = slope * Other (x), OLS through origin.
Output: printed table + CSV file.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from pyproj import CRS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================================
# PATHS
# ===========================================================================
BASE_DIR = "/home/jh94030/scripts/python/postdoc_project/rxfire"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figure")
OUT_DIR = FIG_DIR

DIR_SCRIPTS = os.path.join(BASE_DIR, "analysis")
sys.path.append(os.path.join(DIR_SCRIPTS, "step3_BurnDataSelection"))
from util import CMAQGrid2D  # noqa: E402

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

SEFM_GDB_PATH  = os.path.join("/work/chflab/jthuang/breadcrumbs", "SEFM_L_ABA_1994_2024_polys.gdb")

SEFM_DIR = os.path.join(DATA_DIR, "oth_fire_inv/SEFM_gridded_daily")
SEFM_CACHE_DIR = SEFM_DIR

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
YEARS        = [2017, 2018, 2019]
SE_ST_ABBR   = ["FL", "GA", "SC"]
CELL_AREA_KM2   = 12 * 12
CELL_AREA_ACRES = CELL_AREA_KM2 * 247.105
M2_TO_ACRES     = 1.0 / 4046.8564224

# ===========================================================================
# SEFM DECODE / FILTER HELPERS
# ===========================================================================
NLCDR_CODES = {
    11: "Open Water", 12: "Perennial Ice/Snow", 20: "Developed",
    31: "Barren", 40: "Forest", 52: "Perennial Shrub/Scrub",
    71: "Grasslands/Herbaceous", 81: "Cultivated Crops",
    82: "Cultivated Crops", 90: "Woody Wetlands",
    95: "Emergent Herbaceous",
}


def _clamp_day(day: int) -> int:
    if day == 0:
        return 1
    if day >= 32:
        return 31
    return day


def fix_date_raw(val):
    if pd.isna(val):
        return val
    s = str(int(val))
    if len(s) == 8:
        day = _clamp_day(int(s[6:]))
        return float(f"{s[:6]}{day:02d}")
    return val


def decode_date(val):
    if pd.isna(val):
        return "N/A"
    s = str(int(val))
    if len(s) == 8:
        day = _clamp_day(int(s[6:]))
        return f"{s[:4]}-{s[4:6]}-{day:02d}"
    return str(int(val))


def decode_sefm_attributes(gdf):
    """Decode date columns and NLCD category for SEFM data."""
    gdf = gdf.copy()
    drop_cols = [c for c in ("prebd_mean", "prebd_std", "bd_mean", "bd_std")
                 if c in gdf.columns]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)
    for col in ("prebd_min", "prebd_max", "bd_min", "bd_max"):
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(fix_date_raw)

    def _decode_nlcdr_domi(v):
        if pd.isna(v) or v is None:
            return "N/A"
        v = str(v).strip()
        code = int(v.split("_")[-1]) if "_" in v else int(v)
        return NLCDR_CODES.get(code, f"Unknown ({code})")

    if "nlcdr_domi" in gdf.columns:
        gdf["nlcdr_domi_name"] = gdf["nlcdr_domi"].apply(_decode_nlcdr_domi)
    for col in ("prebd_min", "prebd_max", "bd_min", "bd_max"):
        if col in gdf.columns:
            gdf[f"{col}_date"] = gdf[col].apply(decode_date)
    if gdf.crs is not None and gdf.crs.is_projected:
        gdf["area_ha"] = gdf.geometry.area / 10_000.0
    return gdf


def filter_spring_fires(gdf):
    """Keep only Jan-Apr fires and exclude Cultivated Crops."""
    if "bd_min_date" not in gdf.columns:
        raise ValueError("Run decode_sefm_attributes() first")
    dates = pd.to_datetime(gdf["bd_min_date"], format="%Y-%m-%d",
                           errors="coerce")
    mask_date = dates.dt.month.between(1, 4, inclusive="both")
    mask_crop = True
    if "nlcdr_domi_name" in gdf.columns:
        mask_crop = gdf["nlcdr_domi_name"] != "Cultivated Crops"
    filtered = gdf[mask_date & mask_crop].reset_index(drop=True)
    removed = len(gdf) - len(filtered)
    print(f"  filter_spring_fires: kept {len(filtered):,} / {len(gdf):,} "
          f"({removed:,} removed)")
    return filtered


# ===========================================================================
# GRID HELPERS
# ===========================================================================

def load_cmaq_grid(metcro_file):
    info = CMAQGrid2D(metcro_file)
    lon, lat = info["Lon"], info["Lat"]
    nrows, ncols = lat.shape
    return lon, lat, nrows, ncols, info


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


def yearly_acres_grids(df, lat_col, lon_col, acres_col,
                       years, tree, nrows, ncols):
    """KDTree regridding -> dict {year: 2D array of total acres per cell}."""
    df = df.copy().dropna(subset=[lat_col, lon_col, acres_col])
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols
    grids = {}
    for yr in years:
        grid = np.zeros((nrows, ncols))
        sub = df[df["YEAR"] == yr]
        if not sub.empty:
            grouped = sub.groupby(["ROW", "COL"], observed=True)[acres_col].sum()
            for (r, c), val in grouped.items():
                grid[r, c] = val
        grids[yr] = grid
    return grids


def sefm_yearly_acres_grids(alloc_df, years, nrows, ncols):
    """dict {year: 2D array of total allocated SEFM acres per cell}."""
    grids = {}
    for yr in years:
        grid = np.zeros((nrows, ncols))
        sub = alloc_df[alloc_df["YEAR"] == yr]
        if not sub.empty:
            grouped = sub.groupby(["ROW", "COL"],
                                  observed=True)["allocated_acres"].sum()
            for (r, c), val in grouped.items():
                grid[r, c] = val
        grids[yr] = grid
    return grids


def daily_acres_grids(df, lat_col, lon_col, acres_col, date_col,
                      tree, nrows, ncols):
    """KDTree regridding -> dict {date: 2D array of total acres per cell}."""
    df = df.copy().dropna(subset=[lat_col, lon_col, acres_col])
    pts = np.column_stack((df[lat_col].values, df[lon_col].values))
    _, idx_flat = tree.query(pts, k=1)
    df["ROW"] = idx_flat // ncols
    df["COL"] = idx_flat % ncols
    df["_date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    grids = {}
    for dt, sub in df.groupby("_date"):
        grid = np.zeros((nrows, ncols))
        grouped = sub.groupby(["ROW", "COL"], observed=True)[acres_col].sum()
        for (r, c), val in grouped.items():
            grid[r, c] = val
        grids[dt] = grid
    return grids


def sefm_daily_acres_grids(alloc_df, nrows, ncols):
    """dict {date: 2D array of total allocated SEFM acres per cell}."""
    alloc_df = alloc_df.copy()
    alloc_df["_date"] = pd.to_datetime(alloc_df["date"]).dt.tz_localize(None).dt.normalize()
    grids = {}
    for dt, sub in alloc_df.groupby("_date"):
        grid = np.zeros((nrows, ncols))
        grouped = sub.groupby(["ROW", "COL"],
                              observed=True)["allocated_acres"].sum()
        for (r, c), val in grouped.items():
            grid[r, c] = val
        grids[dt] = grid
    return grids


# ===========================================================================
# DATA LOADERS
# ===========================================================================

def load_permits_jan_apr(template, years, date_col="DATE"):
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: permit file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[date_col])
        df["YEAR"] = yr
        df = df[df[date_col].dt.month.between(1, 4)]
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No permit files found.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Permits (Jan-Apr): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_permits_full_year(template, years, date_col="DATE"):
    """Load permit CSVs, keep all months (full year)."""
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: permit file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[date_col])
        df["YEAR"] = yr
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No permit files found.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Permits (Full Year): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_nei_jan_apr(template, years, date_col="DATE"):
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: NEI file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[date_col])
        df["YEAR"] = yr
        df = df[df[date_col].dt.month.between(1, 4)]
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No NEI files found.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  NEI (Jan-Apr): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_nei_full_year(template, years, date_col="DATE"):
    """Load NEI CSVs, keep all months (full year)."""
    frames = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: NEI file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[date_col])
        df["YEAR"] = yr
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No NEI files found.")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  NEI (Full Year): {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


def load_finn(template, years, parse_dates_col="DAY"):
    dfs = []
    for yr in years:
        fpath = template.format(yr)
        if not os.path.isfile(fpath):
            print(f"  WARNING: FINN file not found -> {fpath}")
            continue
        df = pd.read_csv(fpath, parse_dates=[parse_dates_col])
        df["YEAR"] = yr
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No FINN files found.")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  FINN: {len(combined):,} records, "
          f"{combined['YEAR'].nunique()} years")
    return combined


# ===========================================================================
# SEFM REGRIDDING (polygon-based, area-weighted) -- for cold start only
# ===========================================================================

def _crs_from_metcro(info):
    import pyproj
    import netCDF4 as nc
    ds = nc.Dataset(METCRO2D_FILE)
    lat_1 = ds.getncattr('P_ALP')
    lat_2 = ds.getncattr('P_BET')
    lat_0 = ds.getncattr('YCENT')
    lon_0 = ds.getncattr('XCENT')
    return CRS.from_proj4(
        f"+proj=lcc +a=6370000.0 +b=6370000.0 "
        f"+lat_1={lat_1} +lat_2={lat_2} "
        f"+lat_0={lat_0} +lon_0={lon_0} "
        f"+x_0=0 +y_0=0 +units=m +no_defs"
    )


def build_cmaq_grid_gdf(metcro_file, cmaq_info):
    import netCDF4 as nc
    ds = nc.Dataset(metcro_file)
    ncols = int(ds.getncattr('NCOLS'))
    nrows = int(ds.getncattr('NROWS'))
    xcell = float(ds.getncattr('XCELL'))
    ycell = float(ds.getncattr('YCELL'))
    xorig = float(ds.getncattr('XORIG'))
    yorig = float(ds.getncattr('YORIG'))
    crs = _crs_from_metcro(cmaq_info)
    rows_list = []
    for r in range(nrows):
        y_lo = yorig + r * ycell
        y_hi = y_lo + ycell
        for c in range(ncols):
            x_lo = xorig + c * xcell
            x_hi = x_lo + xcell
            rows_list.append({
                "ROW": r, "COL": c,
                "geometry": box(x_lo, y_lo, x_hi, y_hi),
            })
    gdf = gpd.GeoDataFrame(rows_list, crs=crs)
    lon2d = np.asarray(cmaq_info["Lon"])
    lat2d = np.asarray(cmaq_info["Lat"])
    gdf["centre_lon"] = [lon2d[r, c] for r, c in zip(gdf["ROW"], gdf["COL"])]
    gdf["centre_lat"] = [lat2d[r, c] for r, c in zip(gdf["ROW"], gdf["COL"])]
    gdf.attrs["NROWS"] = nrows
    gdf.attrs["NCOLS"] = ncols
    gdf.attrs["XCELL"] = xcell
    gdf.attrs["YCELL"] = ycell
    gdf.attrs["XORIG"] = xorig
    gdf.attrs["YORIG"] = yorig
    gdf.attrs["cmaq_lon"] = lon2d
    gdf.attrs["cmaq_lat"] = lat2d
    return gdf


def subset_grid_to_se(grid_gdf, states_gdf):
    se_ll = states_gdf.to_crs(grid_gdf.crs)
    centroids = grid_gdf.copy()
    centroids["_centroid"] = centroids.geometry.centroid
    centroids = centroids.set_geometry("_centroid")
    joined = gpd.sjoin(centroids, se_ll[["STUSPS", "geometry"]],
                       how="inner", predicate="within")
    joined = joined.rename(columns={"STUSPS": "STATE"})
    joined = joined.set_geometry("geometry")
    joined = joined.drop(columns=["_centroid", "index_right"], errors="ignore")
    return joined.reset_index(drop=True)


def load_sefm_spring(gdb_path, years):
    frames = []
    for yr in years:
        layer_name = f"L_BurnedArea_{yr}_poly"
        print(f"  Loading SEFM {yr} (layer: {layer_name}) ...")
        gdf = gpd.read_file(gdb_path, layer=layer_name)
        gdf = decode_sefm_attributes(gdf)
        gdf = filter_spring_fires(gdf)
        gdf["date"] = pd.to_datetime(gdf["bd_min_date"], format="%Y-%m-%d",
                                     errors="coerce")
        gdf = gdf.dropna(subset=["date"])
        gdf["YEAR"] = yr
        frames.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                crs=frames[0].crs)
    print(f"  Total SEFM spring-fire polygons: {len(combined):,}")
    return combined


def _repair_single_geometry(geom):
    """Return a GEOS-valid geometry, using make_valid when available."""
    if geom is None or geom.is_empty or geom.is_valid:
        return geom
    try:
        from shapely import make_valid
        return make_valid(geom)
    except Exception:
        try:
            from shapely.validation import make_valid
            return make_valid(geom)
        except Exception:
            return geom.buffer(0)


def repair_polygon_gdf(gdf, label="geometries"):
    """Repair invalid geometries and remove empty or zero-area features.

    This is needed for SEFM polygons because a small number of records can
    contain self-intersections. Those invalid rings can crash GEOS during
    vectorized intersection even when most polygons are valid.
    """
    gdf = gdf.copy()
    n0 = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()

    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        print(f"  Repairing {n_invalid:,} invalid {label} ...")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(
            _repair_single_geometry
        )

    # Re-check after repair. buffer(0) can occasionally return empty geometry.
    valid = gdf.geometry.notna() & gdf.geometry.is_valid & ~gdf.geometry.is_empty
    positive_area = gdf.geometry.area > 0
    keep = valid & positive_area
    n_drop = n0 - int(keep.sum())
    if n_drop > 0:
        print(f"  Dropping {n_drop:,} empty/invalid/zero-area {label}.")
    return gdf.loc[keep].reset_index(drop=True)


def safe_intersection_areas(fire_g, grid_g):
    """Compute intersection areas robustly.

    The fast path uses Shapely 2 vectorized operations. If any remaining
    geometry triggers a GEOS TopologyException, the code falls back to a
    pair-by-pair calculation and repairs the specific failing geometry.
    """
    from shapely import intersection
    from shapely import area as shapely_area
    from shapely.errors import GEOSException

    try:
        return np.asarray(shapely_area(intersection(fire_g, grid_g)), dtype=float)
    except GEOSException as exc:
        print(f"    WARNING: vectorized intersection failed ({exc}).")
        print("    Falling back to pair-wise repaired intersections for this batch.")
        out = np.zeros(len(fire_g), dtype=float)
        for i, (fg, gg) in enumerate(zip(fire_g, grid_g)):
            try:
                out[i] = fg.intersection(gg).area
            except GEOSException:
                try:
                    fg2 = _repair_single_geometry(fg)
                    gg2 = _repair_single_geometry(gg)
                    if fg2 is not None and gg2 is not None:
                        out[i] = fg2.intersection(gg2).area
                except Exception:
                    out[i] = 0.0
        return out


def regrid_sefm_to_grid(fire_gdf, grid_gdf, batch_size=10_000):
    fire_lcc = fire_gdf.to_crs(grid_gdf.crs).copy()
    fire_lcc = repair_polygon_gdf(fire_lcc, label="SEFM fire polygons")
    grid_clean = repair_polygon_gdf(grid_gdf, label="CMAQ grid cells")

    fire_lcc["fire_idx"] = np.arange(len(fire_lcc))
    fire_lcc["fire_area_m2"] = fire_lcc.geometry.area
    fire_lcc = fire_lcc[fire_lcc["fire_area_m2"] > 0].reset_index(drop=True)

    n_fires = len(fire_lcc)
    n_batches = (n_fires + batch_size - 1) // batch_size
    print(f"  Regridding {n_fires:,} fire polygons in {n_batches} batches ...")

    results = []
    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, n_fires)
        batch = fire_lcc.iloc[lo:hi].copy()

        pairs = gpd.sjoin(
            batch[["fire_idx", "geometry"]],
            grid_clean[["ROW", "COL", "STATE", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        if pairs.empty:
            continue

        fire_geoms = batch.set_index("fire_idx")["geometry"]
        grid_geoms = grid_clean["geometry"]
        fire_g = fire_geoms.loc[pairs["fire_idx"]].values
        grid_g = grid_geoms.loc[pairs["index_right"]].values

        pairs = pairs.copy()
        pairs["intersect_area_m2"] = safe_intersection_areas(fire_g, grid_g)
        pairs = pairs[pairs["intersect_area_m2"] > 0].copy()
        if pairs.empty:
            continue

        pairs = pairs.merge(
            batch[["fire_idx", "fire_area_m2", "date", "YEAR"]],
            on="fire_idx",
            how="left",
        )
        pairs["frac"] = pairs["intersect_area_m2"] / pairs["fire_area_m2"]
        pairs["frac"] = pairs["frac"].clip(lower=0, upper=1)
        pairs["allocated_acres"] = pairs["intersect_area_m2"] * M2_TO_ACRES

        results.append(
            pairs[[
                "fire_idx", "ROW", "COL", "STATE", "date", "YEAR",
                "fire_area_m2", "intersect_area_m2", "frac",
                "allocated_acres",
            ]]
        )

        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"    batch {b+1}/{n_batches} done  "
                  f"({hi:,} / {n_fires:,} fires)")

    if not results:
        raise RuntimeError("SEFM regridding produced no grid-cell intersections.")
    return pd.concat(results, ignore_index=True)


# ===========================================================================
# REGRESSION STATISTICS
# ===========================================================================

def yearly_regression_stats(permit_grids, other_grids, state_mask, years,
                            label, period="Jan-Apr"):
    """
    OLS regression (no intercept) for yearly grid-cell comparison:
    Permits (y) vs other inventory (x).
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
        print(f"  WARNING: no matching grid cells for Permits vs {label}")
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
        Period=period,
        Temporal="Yearly",
        N=len(permit_all),
        Slope=round(slope, 4),
        Slope_CI_lo=round(cf[0, 0], 4),
        Slope_CI_hi=round(cf[0, 1], 4),
        R2=round(r2, 4),
    )


def daily_regression_stats(permit_daily, other_daily, state_mask, label,
                           period="Jan-Apr"):
    """
    OLS regression (no intercept) for daily grid-cell comparison:
    Permits (y) vs other inventory (x).
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
        print(f"  WARNING: no daily matching grid cells for "
              f"Permits vs {label}")
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
    print("=" * 60)
    print("Loading CMAQ grid ...")
    cmaq_lon, cmaq_lat, nrows, ncols, cmaq_info = load_cmaq_grid(METCRO2D_FILE)
    tree = grid_kdtree(cmaq_lat, cmaq_lon)

    # 2. Load state boundaries
    print("Loading state boundaries ...")
    gdf_states_all = gpd.read_file(STATES_SHP)
    gdf_SE = gdf_states_all[gdf_states_all["STUSPS"].isin(SE_ST_ABBR)]
    state_mask = build_state_mask(cmaq_lon, cmaq_lat, gdf_SE)

    # 3. Load inventories (Jan-Apr)
    print("\n" + "=" * 60)
    print("Loading Permits (Jan-Apr) ...")
    permits_df = load_permits_jan_apr(PERMIT_TEMPLATE, YEARS)

    print("\n" + "=" * 60)
    print("Loading NEI (Jan-Apr) ...")
    nei_df = load_nei_jan_apr(NEI_TEMPLATE, YEARS)

    print("\n" + "=" * 60)
    print("Loading FINN (Jan-Apr) ...")
    finn_df = load_finn(FINN_TEMPLATE, YEARS)

    # 3b. Load Permits & NEI full year
    print("\n" + "=" * 60)
    print("Loading Permits (Full Year) ...")
    permits_fy_df = load_permits_full_year(PERMIT_TEMPLATE, YEARS)

    print("\n" + "=" * 60)
    print("Loading NEI (Full Year) ...")
    nei_fy_df = load_nei_full_year(NEI_TEMPLATE, YEARS)

    # 4. Load / compute SEFM
    print("\n" + "=" * 60)
    sefm_yearly_npz = os.path.join(SEFM_CACHE_DIR, "sefm_yearly_acres_grids.npz")
    sefm_alloc_parquet = os.path.join(SEFM_CACHE_DIR, "sefm_alloc_df.parquet")

    have_sefm_cache = (os.path.isfile(sefm_yearly_npz)
                       and os.path.isfile(sefm_alloc_parquet))

    if have_sefm_cache:
        print("Loading pre-computed SEFM outputs ...")
        _data = np.load(sefm_yearly_npz)
        sefm_yearly = {yr: _data[str(yr)] for yr in YEARS}
        sefm_alloc_df = pd.read_parquet(sefm_alloc_parquet)
    else:
        print("Running SEFM regridding pipeline (this may take a while) ...")
        grid_full = build_cmaq_grid_gdf(METCRO2D_FILE, cmaq_info)
        grid_se   = subset_grid_to_se(grid_full, gdf_SE)
        print(f"  SE grid cells: {len(grid_se):,}")
        sefm_gdf = load_sefm_spring(SEFM_GDB_PATH, YEARS)
        alloc_df = regrid_sefm_to_grid(sefm_gdf, grid_se)
        print(f"  Total allocated acres: "
              f"{alloc_df['allocated_acres'].sum():,.0f}")
        sefm_yearly = sefm_yearly_acres_grids(alloc_df, YEARS, nrows, ncols)
        sefm_alloc_df = alloc_df
        # Cache
        cache_dir = SEFM_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(sefm_yearly_npz,
                 **{str(yr): sefm_yearly[yr] for yr in YEARS})
        sefm_alloc_df.to_parquet(sefm_alloc_parquet, index=False)
        print("  Cached SEFM outputs for future runs.")

    # ==================================================================
    # 5. Compute yearly acres grids (Jan-Apr)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Computing yearly acres grids (Jan-Apr) ...")

    permits_yearly = yearly_acres_grids(
        df=permits_df, lat_col="LATITUDE", lon_col="LONGITUDE",
        acres_col="ACRES", years=YEARS, tree=tree,
        nrows=nrows, ncols=ncols)
    nei_yearly = yearly_acres_grids(
        df=nei_df, lat_col="latitude", lon_col="longitude",
        acres_col="ACRESBURNED", years=YEARS, tree=tree,
        nrows=nrows, ncols=ncols)
    finn_yearly = yearly_acres_grids(
        df=finn_df, lat_col="LATI", lon_col="LONGI",
        acres_col="AREA", years=YEARS, tree=tree,
        nrows=nrows, ncols=ncols)

    # ==================================================================
    # 5b. Compute yearly acres grids (Full Year - Permits vs NEI only)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Computing yearly acres grids (Full Year) ...")

    permits_fy_yearly = yearly_acres_grids(
        df=permits_fy_df, lat_col="LATITUDE", lon_col="LONGITUDE",
        acres_col="ACRES", years=YEARS, tree=tree,
        nrows=nrows, ncols=ncols)
    nei_fy_yearly = yearly_acres_grids(
        df=nei_fy_df, lat_col="latitude", lon_col="longitude",
        acres_col="ACRESBURNED", years=YEARS, tree=tree,
        nrows=nrows, ncols=ncols)

    # ==================================================================
    # 6. Compute daily acres grids (Jan-Apr)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Computing daily acres grids (Jan-Apr) ...")

    permits_daily = daily_acres_grids(
        df=permits_df, lat_col="LATITUDE", lon_col="LONGITUDE",
        acres_col="ACRES", date_col="DATE", tree=tree,
        nrows=nrows, ncols=ncols)
    nei_daily = daily_acres_grids(
        df=nei_df, lat_col="latitude", lon_col="longitude",
        acres_col="ACRESBURNED", date_col="DATE", tree=tree,
        nrows=nrows, ncols=ncols)
    finn_daily = daily_acres_grids(
        df=finn_df, lat_col="LATI", lon_col="LONGI",
        acres_col="AREA", date_col="DAY", tree=tree,
        nrows=nrows, ncols=ncols)
    sefm_daily = sefm_daily_acres_grids(sefm_alloc_df, nrows, ncols)

    print(f"  Permits: {len(permits_daily):,} unique dates")
    print(f"  NEI:     {len(nei_daily):,} unique dates")
    print(f"  FINN:    {len(finn_daily):,} unique dates")
    print(f"  SEFM:    {len(sefm_daily):,} unique dates")

    # ==================================================================
    # 6b. Compute daily acres grids (Full Year - Permits vs NEI only)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Computing daily acres grids (Full Year) ...")

    permits_fy_daily = daily_acres_grids(
        df=permits_fy_df, lat_col="LATITUDE", lon_col="LONGITUDE",
        acres_col="ACRES", date_col="DATE", tree=tree,
        nrows=nrows, ncols=ncols)
    nei_fy_daily = daily_acres_grids(
        df=nei_fy_df, lat_col="latitude", lon_col="longitude",
        acres_col="ACRESBURNED", date_col="DATE", tree=tree,
        nrows=nrows, ncols=ncols)

    print(f"  Permits: {len(permits_fy_daily):,} unique dates")
    print(f"  NEI:     {len(nei_fy_daily):,} unique dates")

    # ==================================================================
    # 7. Regression statistics table
    # ==================================================================
    print("\n" + "=" * 60)
    print("Computing regression statistics ...")

    stat_rows = []

    # -- Jan-Apr: Yearly --
    for lbl, other_yr in [("NEI", nei_yearly),
                          ("FINN", finn_yearly),
                          ("SEFM", sefm_yearly)]:
        row = yearly_regression_stats(
            permits_yearly, other_yr, state_mask, YEARS, lbl,
            period="Jan-Apr")
        if row is not None:
            stat_rows.append(row)

    # -- Jan-Apr: Daily --
    for lbl, other_dy in [("NEI", nei_daily),
                          ("FINN", finn_daily),
                          ("SEFM", sefm_daily)]:
        row = daily_regression_stats(
            permits_daily, other_dy, state_mask, lbl,
            period="Jan-Apr")
        if row is not None:
            stat_rows.append(row)

    # -- Full Year: Yearly (Permits vs NEI only) --
    row = yearly_regression_stats(
        permits_fy_yearly, nei_fy_yearly, state_mask, YEARS, "NEI",
        period="Full Year")
    if row is not None:
        stat_rows.append(row)

    # -- Full Year: Daily (Permits vs NEI only) --
    row = daily_regression_stats(
        permits_fy_daily, nei_fy_daily, state_mask, "NEI",
        period="Full Year")
    if row is not None:
        stat_rows.append(row)

    df_stats = pd.DataFrame(stat_rows)

    col_order = ["Comparison", "Period", "Temporal", "N", "Common_dates",
                 "Slope", "Slope_CI_lo", "Slope_CI_hi", "R2"]
    col_order = [c for c in col_order if c in df_stats.columns]
    df_stats = df_stats[col_order]

    # Print table
    print("\n  Grid-cell regression: Permits (y) = slope * Other (x), "
          "no intercept")
    print("  Slope 95% CI in brackets. "
          "R2 from sklearn (uncentered, no intercept).")
    print()
    print(df_stats.to_string(index=False))

    # Save to CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUT_DIR,
                           "regression_stats_permits_vs_inventories.csv")
    df_stats.to_csv(csv_out, index=False)
    print(f"\n  Wrote -> {csv_out}")

    print("\n" + "=" * 60)
    print("=== Done ===")