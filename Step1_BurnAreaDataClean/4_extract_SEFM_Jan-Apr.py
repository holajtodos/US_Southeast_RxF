# -*- coding: utf-8 -*-
from __future__ import annotations
###############################################################################
# extract_SEFM_Jan-Apr.py
# purpose: Regrid SEFM polygon burned area (Jan–Apr, 2017-2019) to CMAQ 12US1
#          grid via area-weighted intersection, output daily gridded CSV files
#
# Approach
# --------
# 1. Read the CMAQ METCRO2D file to obtain grid dimensions, cell size, origin,
#    and projection attributes, then build CMAQ grid-cell polygons (LCC CRS).
# 2. Subset grid cells to SE states (FL, GA, SC) via centroid-in-state join.
# 3. Load SEFM burned-area polygons for each year, decode attributes, filter
#    to Jan–Apr, and exclude Cultivated Crops land cover.
# 4. Reproject SEFM polygons to the CMAQ LCC CRS.
# 5. Overlay (intersect) each fire polygon with the SE grid cells. Allocate
#    the fire's burned area proportionally to intersection-area fractions.
# 6. Assign each fire to a single date using bd_min_date.
# 7. Aggregate to daily (date, ROW, COL, STATE) totals and write one CSV
#    per calendar day that contains fires.
# 8. Compute mean-annual percent-of-cell-area burned and save as .npy array.
# 9. Aggregate and save annual total allocated acres by state as CSV.
###############################################################################

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely import intersection as shapely_intersection
from shapely import area as shapely_area
from pyproj import CRS
import pyproj
import netCDF4 as nc
from datetime import datetime

# ------------------------- Environment / Paths -------------------------
dir_python_scripts = "/home/jh94030/scripts/python/postdoc_project/US_Southeast_RxF/Step1_BurnAreaDataClean"

os.chdir(dir_python_scripts)
GDB_PATH = r"/home/jh94030/work/breadcrumbs/SEFM/SEFM_L_ABA_1994_2024_polys.gdb"

# ---------------------------------------------------------------------------
# Lookup / decode dictionaries
# ---------------------------------------------------------------------------

ECOL3_CODES = {
    80301: "Northern Piedmont",
    80303: "Interior Plateau",
    80304: "Piedmont",
    80305: "Southeastern Plains",
    80306: "Mississippi Valley Loess Plains",
    80307: "South Central Plains",
    80308: "East Texas Central Plains",
    80401: "Ridge and Valley",
    80402: "Central Appalachians",
    80403: "Western Allegheny Plateau",
    80404: "Blue Ridge",
    80409: "Southwestern Appalachians",
    80501: "Middle Atlantic Coastal Plain",
    80502: "Mississippi Alluvial Plain",
    80503: "Southern Coastal Plain",
    90405: "Cross Timbers",
    90407: "Texas Blackland Prairies",
    90501: "Western Gulf Coastal Plain",
    150401: "Southern Florida Coastal Plain",
}

NLCDR_CODES = {
    11: "Open Water",
    12: "Perennial Ice/Snow",
    20: "Developed",
    31: "Barren",
    40: "Forest",
    52: "Perennial Shrub/Scrub",
    71: "Grasslands/Herbaceous",
    81: "Cultivated Crops",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous",
}

# ---------------------------------------------------------------------------
# Helper: decode an yyyymmdd float to a proper date string
# ---------------------------------------------------------------------------

def _clamp_day(day: int) -> int:
    """Clamp an invalid day-of-month: 0 → 1, 32 → 31."""
    if day == 0:
        return 1
    if day >= 32:
        return 31
    return day


def fix_date_raw(val: float) -> float:
    """Return a corrected YYYYMMDD float, clamping day 00→01 and 32→31."""
    if pd.isna(val):
        return val
    s = str(int(val))
    if len(s) == 8:
        day = _clamp_day(int(s[6:]))
        return float(f"{s[:6]}{day:02d}")
    return val


def decode_date(val: float) -> str:
    """Convert a YYYYMMDD float value to an ISO date string 'YYYY-MM-DD'.
    Days of 00 are corrected to 01; days of 32 are corrected to 31.
    """
    if pd.isna(val):
        return "N/A"
    s = str(int(val))
    if len(s) == 8:
        day = _clamp_day(int(s[6:]))
        return f"{s[:4]}-{s[4:6]}-{day:02d}"
    return str(int(val))


def decode_ecol3(val: float) -> str:
    """Decode an ecol3 float code to its ecoregion name."""
    if pd.isna(val):
        return "N/A"
    return ECOL3_CODES.get(int(val), f"Unknown ({int(val)})")

# ---------------------------------------------------------------------------
# Read a single layer into a GeoDataFrame
# ---------------------------------------------------------------------------

def read_layer(gdb_path: str, layer_name: str) -> gpd.GeoDataFrame:
    """
    Read one layer from the GDB and return a GeoDataFrame.

    Parameters
    ----------
    gdb_path   : path to the .gdb folder
    layer_name : e.g. 'L_BurnedArea_2023_poly'

    Returns
    -------
    GeoDataFrame  (CRS = Albers Equal Area, WGS-84 datum)
    """
    gdf = gpd.read_file(gdb_path, layer=layer_name)
    return gdf


def read_year(gdb_path: str, year: int) -> gpd.GeoDataFrame:
    """Convenience wrapper – load by year integer (1994-2024)."""
    layer_name = f"L_BurnedArea_{year}_poly"
    return read_layer(gdb_path, layer_name)


# ---------------------------------------------------------------------------
# Decode / enrich a GeoDataFrame with human-readable labels
# ---------------------------------------------------------------------------

def decode_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Decode / enrich a GeoDataFrame:
      - Drops   : prebd_mean, prebd_std, bd_mean, bd_std
      - Fixes   : prebd_min, prebd_max, bd_min, bd_max raw floats
                  (day 00 → 01, day 32 → 31)
      - Adds    : ecol3_name, nlcdr_domi_name
      - Adds    : prebd_min_date, prebd_max_date, bd_min_date, bd_max_date
                  as 'YYYY-MM-DD' ISO strings
      - Adds    : area_ha (polygon area in hectares)
    """
    gdf = gdf.copy()

    # -- Drop unwanted statistical date columns
    drop_cols = [c for c in ("prebd_mean", "prebd_std", "bd_mean", "bd_std") if c in gdf.columns]
    if drop_cols:
        gdf = gdf.drop(columns=drop_cols)

    # -- Fix raw date float values (clamp invalid day values)
    for col in ("prebd_min", "prebd_max", "bd_min", "bd_max"):
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(fix_date_raw)

    # -- Ecoregion label
    if "ecol3" in gdf.columns:
        gdf["ecol3_name"] = gdf["ecol3"].apply(decode_ecol3)

    # -- Dominant NLCD reclassified label
    # Values are stored as e.g. 'nlcdr_40'; extract the trailing numeric code.
    def _decode_nlcdr_domi(v):
        if pd.isna(v) or v is None:
            return "N/A"
        v = str(v).strip()
        # Handle both "nlcdr_40" and bare "40" formats
        if "_" in v:
            code = int(v.split("_")[-1])
        else:
            code = int(v)
        return NLCDR_CODES.get(code, f"Unknown ({code})")

    if "nlcdr_domi" in gdf.columns:
        gdf["nlcdr_domi_name"] = gdf["nlcdr_domi"].apply(_decode_nlcdr_domi)

    # -- Decode kept date fields to ISO strings
    for col in ("prebd_min", "prebd_max", "bd_min", "bd_max"):
        if col in gdf.columns:
            gdf[f"{col}_date"] = gdf[col].apply(decode_date)

    # -- Area in hectares (CRS is in metres)
    if gdf.crs is not None and gdf.crs.is_projected:
        gdf["area_ha"] = gdf.geometry.area / 10_000.0

    return gdf


# ---------------------------------------------------------------------------
# Filter: Jan–Apr burn window, exclude Cultivated Crops
# ---------------------------------------------------------------------------

def filter_spring_fires(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Retain only fire detections that meet both criteria:
      1. bd_min_date falls between YYYY-01-01 and YYYY-04-30 (inclusive),
         where YYYY is taken from the ISO date string itself.
      2. nlcdr_domi_name is NOT 'Cultivated Crops'.

    Must be called after decode_attributes() so that 'bd_min_date' and
    'nlcdr_domi_name' exist.
    """
    if "bd_min_date" not in gdf.columns:
        raise ValueError("Run decode_attributes() before filter_spring_fires() "
                         "so that 'bd_min_date' exists.")
    if "nlcdr_domi_name" not in gdf.columns:
        raise ValueError("Run decode_attributes() before filter_spring_fires().")

    # Parse ISO dates; invalid / 'N/A' strings become NaT and are excluded
    dates = pd.to_datetime(gdf["bd_min_date"], format="%Y-%m-%d", errors="coerce")

    # Jan–Apr window: month must be 1–4 (April has only 30 days, so month≤4 is enough)
    mask_date = dates.dt.month.between(1, 4, inclusive="both")
    mask_crop = gdf["nlcdr_domi_name"] != "Cultivated Crops"

    filtered = gdf[mask_date & mask_crop].reset_index(drop=True)
    removed = len(gdf) - len(filtered)
    print(f"  filter_spring_fires: kept {len(filtered):,} / {len(gdf):,} rows "
          f"({removed:,} removed)")
    return filtered

# ===========================================================================
# CONFIG
# ===========================================================================
YEARS = [2017, 2018, 2019]

# ---- MCIP grid file ----
met_filedir        = "/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01"
METCRO2D_FILE  = f"{met_filedir}/METCRO2D_20170101.nc"

# SE states to clip to
SE_ST_ABBR = ["FL", "GA", "SC"]
SE_ST_FIPS = ["12", "13", "45"]

# Square-metres → acres conversion
M2_TO_ACRES = 1.0 / 4046.8564224

# Output
OUT_DIR = os.path.join("/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv", "SEFM_gridded_daily")

# State boundaries (Census Bureau 500k)
STATES_SHP = (
    "/work/chflab/jthuang/breadcrumbs/mapping_state/"
    "cb_2020_us_state_500k/cb_2020_us_state_500k.shp"
)

# ===========================================================================
# 1. READ METCRO2D & BUILD CMAQ GRID
# ===========================================================================

def CMAQGrid2D(mcip_gridcro2d):
    """
    Read a CMAQ MCIP GRIDCRO2D / METCRO2D file and return a dictionary
    with projection, grid centres, boundaries, lon/lat, and time.

    :param mcip_gridcro2d: path to CMAQ MCIP GRIDCRO2D output
    :return: dict with keys: crs, X_ctr, Y_ctr, X_bdry, Y_bdry,
             Lat, Lon, X_uniq, Y_uniq, time,
             NCOLS, NROWS, XCELL, YCELL, XORIG, YORIG,
             P_ALP, P_BET, YCENT, XCENT
    """
    ds = nc.Dataset(mcip_gridcro2d)
    lat_1 = ds.getncattr('P_ALP')
    lat_2 = ds.getncattr('P_BET')
    lat_0 = ds.getncattr('YCENT')
    lon_0 = ds.getncattr('XCENT')
    crs = pyproj.Proj("+proj=lcc +a=6370000.0 +b=6370000.0 +lat_1=" + str(lat_1)
                      + " +lat_2=" + str(lat_2) + " +lat_0=" + str(lat_0) +
                      " +lon_0=" + str(lon_0))
    xcell = ds.getncattr('XCELL')
    ycell = ds.getncattr('YCELL')
    xorig = ds.getncattr('XORIG')
    yorig = ds.getncattr('YORIG')

    ncols = ds.getncattr('NCOLS')
    nrows = ds.getncattr('NROWS')

    # > for X, Y cell centers
    x_center_range = np.linspace(xorig + xcell / 2,
                                 (xorig + xcell / 2) + xcell * (ncols - 1), ncols)
    y_center_range = np.linspace(yorig + ycell / 2,
                                 (yorig + ycell / 2) + ycell * (nrows - 1), nrows)

    Xcenters, Ycenters = np.meshgrid(x_center_range, y_center_range)

    # > for X, Y cell boundaries (i.e., cell corners)
    x_bound_range = np.linspace(xorig, xorig + xcell * ncols, ncols + 1)
    y_bound_range = np.linspace(yorig, yorig + ycell * nrows, nrows + 1)

    Xbounds, Ybounds = np.meshgrid(x_bound_range, y_bound_range)

    x_max = np.max(Xbounds)
    x_min = np.min(Xbounds)
    y_max = np.max(Ybounds)
    y_min = np.min(Ybounds)

    lon_ctr, lat_ctr = crs(Xcenters, Ycenters, inverse=True)

    time_data = ds['TFLAG'][:]
    cmaq_time_array = []
    for i in range(0, time_data.shape[0]):
        time_data_tmp = time_data[i, 0, :]
        time_str = str(time_data_tmp[0]) + str(time_data_tmp[1]).rjust(6, '0')
        parsed = datetime.strptime(time_str, '%Y%j%H%M%S')
        cmaq_time_array.append(parsed)

    res_dict = {
        "crs": crs,
        "X_ctr": Xcenters, "Y_ctr": Ycenters,
        "X_bdry": [x_min, x_max], "Y_bdry": [y_min, y_max],
        "Lat": lat_ctr, "Lon": lon_ctr,
        "X_uniq": x_center_range, "Y_uniq": y_center_range,
        "time": cmaq_time_array,
        # Grid parameters for downstream polygon construction
        "NCOLS": ncols, "NROWS": nrows,
        "XCELL": xcell, "YCELL": ycell,
        "XORIG": xorig, "YORIG": yorig,
        "P_ALP": lat_1, "P_BET": lat_2,
        "YCENT": lat_0, "XCENT": lon_0,
    }
    return res_dict


def _crs_from_metcro(info: dict) -> CRS:
    """Build a pyproj CRS from METCRO2D global attributes.
    Uses spherical earth (a=b=6370000) matching the CMAQ/WRF convention."""
    return CRS.from_proj4(
        f"+proj=lcc +a=6370000.0 +b=6370000.0 "
        f"+lat_1={info['P_ALP']} +lat_2={info['P_BET']} "
        f"+lat_0={info['YCENT']} +lon_0={info['XCENT']} "
        f"+x_0=0 +y_0=0 +units=m +no_defs"
    )


def build_cmaq_grid_gdf(
    metcro_file: str = METCRO2D_FILE,
) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of CMAQ grid-cell rectangles by reading the
    METCRO2D file.  Grid dimensions, cell sizes, origin, and projection
    are all extracted from the NetCDF global attributes.

    Each row has columns: ROW, COL, geometry (Polygon in LCC metres).
    Also stores cell-centre lon/lat for plotting convenience.

    Additionally stores grid-wide metadata as GeoDataFrame attrs so that
    other functions can access NROWS, NCOLS, XCELL, etc.
    """
    info   = CMAQGrid2D(metcro_file)
    ncols  = int(info["NCOLS"])
    nrows  = int(info["NROWS"])
    xcell  = float(info["XCELL"])
    ycell  = float(info["YCELL"])
    xorig  = float(info["XORIG"])
    yorig  = float(info["YORIG"])
    crs    = _crs_from_metcro(info)
    rows_list = []
    for r in range(nrows):
        y_lo = yorig + r * ycell
        y_hi = y_lo + ycell
        for c in range(ncols):
            x_lo = xorig + c * xcell
            x_hi = x_lo + xcell
            rows_list.append({
                "ROW": r,
                "COL": c,
                "geometry": box(x_lo, y_lo, x_hi, y_hi),
            })

    gdf = gpd.GeoDataFrame(rows_list, crs=crs)

    # Cell centres in lon/lat — use the METCRO2D arrays directly
    lon2d = np.asarray(info["Lon"])    # (nrows, ncols)
    lat2d = np.asarray(info["Lat"])
    gdf["centre_lon"] = [lon2d[r, c] for r, c in zip(gdf["ROW"], gdf["COL"])]
    gdf["centre_lat"] = [lat2d[r, c] for r, c in zip(gdf["ROW"], gdf["COL"])]

    # Store grid metadata as attrs for downstream use
    gdf.attrs["NROWS"]  = nrows
    gdf.attrs["NCOLS"]  = ncols
    gdf.attrs["XCELL"]  = xcell
    gdf.attrs["YCELL"]  = ycell
    gdf.attrs["XORIG"]  = xorig
    gdf.attrs["YORIG"]  = yorig
    gdf.attrs["cmaq_lon"] = lon2d
    gdf.attrs["cmaq_lat"] = lat2d

    return gdf


def subset_grid_to_se(
    grid_gdf: gpd.GeoDataFrame,
    states_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Return only grid cells that intersect the SE-states union polygon.
    Adds a STATE column via spatial join (majority area method is optional;
    here we use centroid-in-state).
    """
    se_ll = states_gdf.to_crs(grid_gdf.crs)
    # Spatial join — keep grid cells whose centroid falls inside a state
    centroids = grid_gdf.copy()
    centroids["_centroid"] = centroids.geometry.centroid
    centroids = centroids.set_geometry("_centroid")

    joined = gpd.sjoin(centroids, se_ll[["STUSPS", "geometry"]],
                       how="inner", predicate="within")
    joined = joined.rename(columns={"STUSPS": "STATE"})
    joined = joined.set_geometry("geometry")          # restore cell polygon
    joined = joined.drop(columns=["_centroid", "index_right"], errors="ignore")
    return joined.reset_index(drop=True)


# ===========================================================================
# 2. LOAD & FILTER SEFM DATA   (uses read_SEFM.py helpers)
# ===========================================================================

def load_sefm_spring(years: list[int] = YEARS) -> gpd.GeoDataFrame:
    """
    Load SEFM polygon layers for the given years, decode attributes,
    filter to Jan–Apr and exclude Cultivated Crops.

    Returns a GeoDataFrame in its native AEA CRS with a 'date' column
    (pd.Timestamp derived from bd_min_date).
    """
    frames = []
    for yr in years:
        print(f"  Loading SEFM {yr} …")
        gdf = read_year(GDB_PATH, yr)
        gdf = decode_attributes(gdf)
        gdf = filter_spring_fires(gdf)

        # Parse bd_min_date → proper date
        gdf["date"] = pd.to_datetime(gdf["bd_min_date"], format="%Y-%m-%d",
                                     errors="coerce")
        gdf = gdf.dropna(subset=["date"])
        gdf["YEAR"] = yr
        frames.append(gdf)

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                crs=frames[0].crs)
    print(f"  Total SEFM spring-fire polygons: {len(combined):,}")
    return combined


# ===========================================================================
# 3. AREA-WEIGHTED REGRIDDING  (polygon → CMAQ cells)
# ===========================================================================

def regrid_polygons_to_grid(
    fire_gdf: gpd.GeoDataFrame,
    grid_gdf: gpd.GeoDataFrame,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """
    Intersect fire polygons with CMAQ grid cells and allocate each fire's
    burned area proportionally to the overlapping fraction.

    Uses spatial-index joins + per-pair geometric intersection for efficiency.

    Parameters
    ----------
    fire_gdf   : SEFM polygons (any CRS — will be reprojected)
    grid_gdf   : CMAQ grid cell polygons (LCC)
    batch_size  : process fires in batches to cap memory

    Returns
    -------
    DataFrame with columns:
        fire_idx, ROW, COL, STATE, date, YEAR,
        fire_area_m2, intersect_area_m2, frac, allocated_acres
    """
    # Reproject fires to grid CRS (LCC)
    fire_lcc = fire_gdf.to_crs(grid_gdf.crs).copy()
    fire_lcc["fire_idx"] = np.arange(len(fire_lcc))
    fire_lcc["fire_area_m2"] = fire_lcc.geometry.area

    # Build a spatial index on the grid
    results = []
    n_fires = len(fire_lcc)
    n_batches = (n_fires + batch_size - 1) // batch_size

    print(f"  Regridding {n_fires:,} fire polygons in {n_batches} batches …")

    for b in range(n_batches):
        lo = b * batch_size
        hi = min(lo + batch_size, n_fires)
        batch = fire_lcc.iloc[lo:hi]

        # Spatial join: find candidate grid cells for each fire polygon
        pairs = gpd.sjoin(batch[["fire_idx", "geometry"]],
                          grid_gdf[["ROW", "COL", "STATE", "geometry"]],
                          how="inner", predicate="intersects")

        if pairs.empty:
            continue

        # For each (fire, grid-cell) pair compute the actual intersection area
        fire_geoms = batch.set_index("fire_idx")["geometry"]
        grid_geoms = grid_gdf["geometry"]

        fire_g = fire_geoms.loc[pairs["fire_idx"]].values
        grid_g = grid_geoms.iloc[pairs["index_right"]].values

        # Vectorised Shapely intersection
        isect_geoms = shapely_intersection(fire_g, grid_g)
        isect_areas = shapely_area(isect_geoms)

        pairs = pairs.copy()
        pairs["intersect_area_m2"] = isect_areas

        # Drop zero-area touches
        pairs = pairs[pairs["intersect_area_m2"] > 0].copy()

        # Merge fire-level attributes
        pairs = pairs.merge(
            batch[["fire_idx", "fire_area_m2", "date", "YEAR"]],
            on="fire_idx", how="left",
        )
        pairs["frac"] = pairs["intersect_area_m2"] / pairs["fire_area_m2"]
        pairs["allocated_acres"] = (
            pairs["fire_area_m2"] * pairs["frac"] * M2_TO_ACRES
        )

        results.append(pairs[["fire_idx", "ROW", "COL", "STATE",
                               "date", "YEAR", "fire_area_m2",
                               "intersect_area_m2", "frac",
                               "allocated_acres"]])

        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"    batch {b+1}/{n_batches} done  "
                  f"({hi:,} / {n_fires:,} fires processed)")

    result = pd.concat(results, ignore_index=True)
    print(f"  Total intersection fragments: {len(result):,}")
    return result


# ===========================================================================
# 4. AGGREGATE TO DAILY GRID & WRITE FILES
# ===========================================================================

def write_daily_gridded(
    alloc_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
    out_dir: str = OUT_DIR,
) -> pd.DataFrame:
    """
    Aggregate allocated acres by (date, ROW, COL, STATE) and write one
    Parquet + CSV per calendar day.

    Returns the aggregated DataFrame.
    """
    os.makedirs(out_dir, exist_ok=True)

    daily = (alloc_df
             .groupby(["date", "YEAR", "ROW", "COL", "STATE"], observed=True)
             .agg(
                 acres_burned=("allocated_acres", "sum"),
                 n_fires=("fire_idx", "nunique"),
             )
             .reset_index())

    # Merge grid cell geometry back for GeoDataFrame output
    grid_slim = grid_gdf[["ROW", "COL", "geometry", "centre_lon", "centre_lat"]]
    daily_geo = daily.merge(grid_slim, on=["ROW", "COL"], how="left")
    daily_geo = gpd.GeoDataFrame(daily_geo, geometry="geometry", crs=grid_gdf.crs)

    dates = sorted(daily_geo["date"].unique())
    for dt in dates:
        ds = str(pd.Timestamp(dt).date())   # 'YYYY-MM-DD'
        sub = daily_geo[daily_geo["date"] == dt]
        fname = f"SEFM_{ds}"
        # write a lightweight CSV (no geometry)
        sub.drop(columns="geometry").to_csv(
            os.path.join(out_dir, f"{fname}.csv"), index=False)

    print(f"  Wrote {len(dates)} daily files to {out_dir}/")
    return daily


# ===========================================================================
# 5. BUILD THE % BURNED GRID
# ===========================================================================

def percent_grid_from_alloc(
    alloc_df: pd.DataFrame,
    years: list[int],
    grid_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Compute the mean-annual percent-of-cell-area burned from the allocated
    fire fragments, comparable to Figure1.py's ``percent_grid_from_points``.

    Parameters
    ----------
    alloc_df  : allocation table from regrid_polygons_to_grid()
    years     : list of year ints
    grid_gdf  : full CMAQ grid GeoDataFrame (carries .attrs with NROWS, etc.)

    Returns a (nrows, ncols) array with NaN outside populated cells.
    """
    nrows = grid_gdf.attrs["NROWS"]
    ncols = grid_gdf.attrs["NCOLS"]
    xcell = grid_gdf.attrs["XCELL"]
    ycell = grid_gdf.attrs["YCELL"]
    cell_area_km2   = (xcell / 1000.0) * (ycell / 1000.0)
    cell_area_acres = cell_area_km2 * 247.105

    # Annual total acres per cell
    annual = (alloc_df
              .groupby(["YEAR", "ROW", "COL"], observed=True)["allocated_acres"]
              .sum()
              .unstack(level=0)
              .reindex(columns=years)
              .fillna(0.0))

    mean_acres = annual.mean(axis=1).values
    mean_pct   = (mean_acres / cell_area_acres) * 100.0

    grid = np.full((nrows, ncols), np.nan)
    for (row, col), pct in zip(annual.index, mean_pct):
        grid[row, col] = pct

    return grid


# ===========================================================================
# 6. ANNUAL TOTAL ACRES BY STATE
# ===========================================================================

def annual_acres_by_state(alloc_df: pd.DataFrame) -> pd.DataFrame:
    """Return annual total allocated acres by STATE and YEAR."""
    summary = (alloc_df
               .groupby(["STATE", "YEAR"], observed=True)["allocated_acres"]
               .sum()
               .reset_index()
               .rename(columns={"allocated_acres": "ACRES"}))
    return summary


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # A. Build CMAQ grid cell polygons
    # ------------------------------------------------------------------
    print("Building CMAQ 12US1 grid polygons from METCRO2D…")
    grid_full = build_cmaq_grid_gdf(METCRO2D_FILE)
    nrows = grid_full.attrs["NROWS"]
    ncols = grid_full.attrs["NCOLS"]
    print(f"  Full grid: {len(grid_full):,} cells "
          f"({nrows} rows × {ncols} cols)")

    # ------------------------------------------------------------------
    # B. Subset to SE states (FL, GA, SC)
    # ------------------------------------------------------------------
    print("Loading state boundaries & subsetting grid …")
    gdf_states_all = gpd.read_file(STATES_SHP)

    gdf_SE = gdf_states_all[gdf_states_all["STUSPS"].isin(SE_ST_ABBR)]
    grid_se = subset_grid_to_se(grid_full, gdf_SE)
    print(f"  SE grid cells: {len(grid_se):,}")

    # ------------------------------------------------------------------
    # C. Load SEFM fire polygons (Jan–Apr 2017-2019, no Cultivated Crops)
    # ------------------------------------------------------------------
    print("\nLoading SEFM spring-fire polygons …")
    sefm_gdf = load_sefm_spring(YEARS)

    # ------------------------------------------------------------------
    # D. Area-weighted regridding
    # ------------------------------------------------------------------
    print("\nRegridding SEFM polygons to CMAQ grid …")
    alloc_df = regrid_polygons_to_grid(sefm_gdf, grid_se)
    print(f"  Total allocated acres: {alloc_df['allocated_acres'].sum():,.0f}")

    # ------------------------------------------------------------------
    # E. Write daily gridded outputs
    # ------------------------------------------------------------------
    print("\nWriting daily gridded files …")
    daily_df = write_daily_gridded(alloc_df, grid_se)

    # ------------------------------------------------------------------
    # F. Summary statistics
    # ------------------------------------------------------------------
    sefm_state_annual = annual_acres_by_state(alloc_df)
    print("\nSEFM annual acres by state (Jan–Apr):")
    print(sefm_state_annual.to_string(index=False))

    # ------------------------------------------------------------------
    # G. Build % burned grid for later plotting
    # ------------------------------------------------------------------
    sefm_pct_grid = percent_grid_from_alloc(alloc_df, YEARS, grid_full)
    np.save(os.path.join(OUT_DIR, "sefm_pct_grid.npy"), sefm_pct_grid)
    print(f"\nSaved SEFM % burned grid → {OUT_DIR}/sefm_pct_grid.npy")

    # ------------------------------------------------------------------
    # H. Save state-level summary CSV
    # ------------------------------------------------------------------
    sefm_state_annual.to_csv(
        os.path.join(OUT_DIR, "sefm_annual_acres_by_state_JanApr.csv"),
        index=False)

    print("\n=== Done ===")