###############################################################################
# Grid_GEUFE_to_CMAQ.py
#
# Purpose: Regrid GEUFE point-based daily emissions (CSV files) onto the CMAQ
#          12US1 Lambert Conformal Conic grid defined by a METCRO2D file.
#          Only prescribed-fire emissions are kept (Ag_flag=False AND
#          Wildfire_flag=False).  Emissions within the same grid cell are
#          summed; FRP is averaged.
#
# Method:
#   1. Read the CMAQ grid parameters (LCC projection, XORIG, YORIG, XCELL,
#      YCELL, NCOLS, NROWS) from a METCRO2D NetCDF file.
#   2. For each daily GEUFE CSV:
#        a. Filter to prescribed-fire only (Ag_flag=False & Wildfire_flag=False)
#        b. Project fire lat/lon to LCC (x, y) metres.
#        c. Assign each fire to a grid cell:  col = (x-XORIG)/XCELL,
#                                              row = (y-YORIG)/YCELL.
#        d. Sum emissions per grid cell (mean for FRP).
#        e. Convert units: emissions kg/day -> kg/m2/s  (divide by cell area
#           and 86400 s/day).
#   3. Write one netCDF per day to the output directory.
#
# Inputs:
#   - GEUFE CSV directory: /home/jh94030/work/breadcrumbs/GEUFE/csv
#   - METCRO2D file:       /scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc
#
# Outputs:
#   - /home/jh94030/scripts/python/postdoc_project/rxfire/data/gridded_CMAQ_12US1/GEUFE_CMAQ12US1_<YYYYMMDD>.nc
###############################################################################

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from pyproj import Transformer

# ===========================================================================
# USER CONFIG
# ===========================================================================

# Date range matching the available GEUFE CSV files
DATE_START = '2019-08-01'
DATE_END   = '2020-07-31'

# Paths
METCRO2D_FILE = r'/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc'
GEUFE_CSV_DIR = r'/home/jh94030/work/breadcrumbs/GEUFE/csv'
OUTPUT_DIR    = os.path.join('/home/jh94030/scripts/python/postdoc_project/rxfire/data', 'gridded_CMAQ_12US1', 'GEUFE')

# If True, overwrite existing gridded files; if False, skip dates with
# existing output.
OVERWRITE = True

# If True, create a file with zeros for days that have no qualifying fires
GRID_EMPTY_DAYS = True

# Emission species to grid (summed per grid cell)
SPECIES_DICT = {
    'OC':      'Organic Carbon',
    'BC':      'Black Carbon',
    'PM25':    'Particulate Matter 2.5 um',
    'CO':      'Carbon Monoxide',
    'CO2':     'Carbon Dioxide',
    'SO2':     'Sulfur Dioxide',
    'NOx':     'Nitrogen Oxides',
    'NH3':     'Ammonia',
    'DM':      'Dry Matter Consumed',
    'meanFRP': 'Fire Radiative Power mean (ABI-measured)',
    'FRE':     'Fire Radiative Energy (integrated, estimated)',
}


# ===========================================================================
# 1. READ CMAQ GRID from METCRO2D
# ===========================================================================

def read_cmaq_grid(metcro_path):
    """
    Extract grid parameters and build cell-centre lat/lon from a
    CMAQ METCRO2D / GRIDCRO2D NetCDF file.

    Returns
    -------
    info : dict  with keys NCOLS, NROWS, XCELL, YCELL, XORIG, YORIG,
           P_ALP, P_BET, P_GAM, XCENT, YCENT, lon2d, lat2d,
           x_centers, y_centers, cell_area_m2, transformer_to_lcc
    """
    ds = nc.Dataset(metcro_path)

    p_alp  = float(ds.getncattr('P_ALP'))
    p_bet  = float(ds.getncattr('P_BET'))
    p_gam  = float(ds.getncattr('P_GAM'))
    xcent  = float(ds.getncattr('XCENT'))
    ycent  = float(ds.getncattr('YCENT'))
    xorig  = float(ds.getncattr('XORIG'))
    yorig  = float(ds.getncattr('YORIG'))
    xcell  = float(ds.getncattr('XCELL'))
    ycell  = float(ds.getncattr('YCELL'))
    ncols  = int(ds.getncattr('NCOLS'))
    nrows  = int(ds.getncattr('NROWS'))
    ds.close()

    # --- LCC projection (CMAQ spherical earth a=b=6370000) ----------------
    proj4_lcc = (
        f"+proj=lcc +a=6370000.0 +b=6370000.0 "
        f"+lat_1={p_alp} +lat_2={p_bet} "
        f"+lat_0={ycent} +lon_0={xcent} "
        f"+x_0=0 +y_0=0 +units=m +no_defs"
    )
    # Transformer: WGS-84 lon/lat  -->  LCC metres
    transformer_to_lcc = Transformer.from_proj(
        "epsg:4326",   # WGS-84
        proj4_lcc,
        always_xy=True  # input order (lon, lat)
    )
    # Transformer: LCC metres --> WGS-84 (for cell-centre lon/lat)
    transformer_to_ll = Transformer.from_proj(
        proj4_lcc,
        "epsg:4326",
        always_xy=True
    )

    # --- Cell centres in LCC metres ----------------------------------------
    x_centers = np.linspace(xorig + xcell / 2,
                            xorig + xcell / 2 + xcell * (ncols - 1), ncols)
    y_centers = np.linspace(yorig + ycell / 2,
                            yorig + ycell / 2 + ycell * (nrows - 1), nrows)
    X2d, Y2d = np.meshgrid(x_centers, y_centers)

    # Cell-centre lon/lat
    lon2d, lat2d = transformer_to_ll.transform(X2d, Y2d)

    # --- Cell area (m²) using trapezoidal approximation --------------------
    cell_area_m2 = xcell * ycell  # For LCC, cell area in projected coords is
    # exactly XCELL*YCELL (m²) because the projection is conformal.
    # (For very high accuracy one could use geodetic areas, but for 12 km
    # CMAQ grids this is the standard approach.)

    info = dict(
        NCOLS=ncols, NROWS=nrows,
        XCELL=xcell, YCELL=ycell,
        XORIG=xorig, YORIG=yorig,
        P_ALP=p_alp, P_BET=p_bet, P_GAM=p_gam,
        XCENT=xcent, YCENT=ycent,
        lon2d=lon2d, lat2d=lat2d,
        x_centers=x_centers, y_centers=y_centers,
        cell_area_m2=cell_area_m2,
        transformer_to_lcc=transformer_to_lcc,
        proj4_lcc=proj4_lcc,
    )
    return info


# ===========================================================================
# 2. ASSIGN FIRES TO GRID CELLS
# ===========================================================================

def assign_fires_to_grid(df, grid_info):
    """
    Given a fire-point DataFrame (must have 'lat', 'lon' columns) and
    CMAQ grid_info dict, compute the (row, col) grid-cell index for each
    fire.  Fires outside the grid are dropped.

    Adds columns 'cmaq_row' and 'cmaq_col' (integer indices 0-based)
    to the returned DataFrame.
    """
    transformer = grid_info['transformer_to_lcc']
    xorig = grid_info['XORIG']
    yorig = grid_info['YORIG']
    xcell = grid_info['XCELL']
    ycell = grid_info['YCELL']
    ncols = grid_info['NCOLS']
    nrows = grid_info['NROWS']

    df = df.copy()

    # Project lon/lat -> LCC metres
    x_lcc, y_lcc = transformer.transform(df['lon'].values, df['lat'].values)

    # Compute column/row indices  (0-based)
    col_idx = np.floor((x_lcc - xorig) / xcell).astype(int)
    row_idx = np.floor((y_lcc - yorig) / ycell).astype(int)

    df['cmaq_col'] = col_idx
    df['cmaq_row'] = row_idx

    # Keep only fires that fall inside the grid
    mask = (
        (col_idx >= 0) & (col_idx < ncols) &
        (row_idx >= 0) & (row_idx < nrows)
    )
    df = df[mask].copy()
    return df


# ===========================================================================
# 3. SUM / AVERAGE EMISSIONS PER GRID CELL -> 2-D ARRAY
# ===========================================================================

def grid_species(df, species, grid_info, resample='sum'):
    """
    Aggregate a single emission species onto the CMAQ 2-D grid.

    Parameters
    ----------
    df : DataFrame  with columns 'cmaq_row', 'cmaq_col', and `species`.
    species : str   column name.
    grid_info : dict
    resample : str  'sum' or 'mean'.

    Returns
    -------
    array2d : numpy array of shape (NROWS, NCOLS)
    """
    nrows = grid_info['NROWS']
    ncols = grid_info['NCOLS']
    array2d = np.zeros((nrows, ncols), dtype=np.float64)

    if len(df) == 0:
        return array2d

    if resample == 'sum':
        grouped = df.groupby(['cmaq_row', 'cmaq_col'])[species].sum()
    elif resample == 'mean':
        grouped = df.groupby(['cmaq_row', 'cmaq_col'])[species].mean()
    else:
        raise ValueError(f"Unknown resample method: {resample}")

    for (r, c), val in grouped.items():
        array2d[r, c] = val

    return array2d


# ===========================================================================
# 4. MAIN LOOP
# ===========================================================================

def main():
    # --- Create output directory -------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Read grid ---------------------------------------------------------
    print('Reading CMAQ grid from METCRO2D …')
    grid_info = read_cmaq_grid(METCRO2D_FILE)
    nrows = grid_info['NROWS']
    ncols = grid_info['NCOLS']
    cell_area = grid_info['cell_area_m2']   # scalar (m²)
    print(f'  Grid: {ncols} cols x {nrows} rows,  cell size: '
          f'{grid_info["XCELL"]/1000:.0f} x {grid_info["YCELL"]/1000:.0f} km')

    # Prepare coordinate arrays for the output netCDF
    # We label dimensions as ROW / COL (CMAQ convention) and also store
    # 2-D lon/lat fields for reference.
    lon2d = grid_info['lon2d']  # (NROWS, NCOLS)
    lat2d = grid_info['lat2d']

    # --- Date loop ---------------------------------------------------------
    date_list = pd.date_range(DATE_START, DATE_END, freq='D')
    print(f'Processing {len(date_list)} dates: {DATE_START} to {DATE_END}\n')

    for date in date_list:
        ymd = date.strftime('%Y%m%d')
        out_file = os.path.join(OUTPUT_DIR, f'GEUFE_CMAQ12US1_{ymd}.nc')

        # Check for existing file
        if os.path.exists(out_file) and not OVERWRITE:
            print(f'  {ymd}: file exists, skipping.')
            continue

        print(f'=== {ymd} ===')

        # --- Read CSV ------------------------------------------------------
        csv_path = os.path.join(GEUFE_CSV_DIR, f'GEUFE_{ymd}.csv')
        found_df = False

        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)

            # Filter: prescribed fires only (not Ag, not Wildfire)
            # and constrain to FL, GA, SC
            df = df[(df['Ag_flag'] == False) & (df['Wildfire_flag'] == False)]
            df = df[df['state'].isin(['FL', 'GA', 'SC'])]

            if len(df) > 0:
                found_df = True
                # Assign fires to CMAQ grid cells
                df = assign_fires_to_grid(df, grid_info)
                print(f'  {len(df)} prescribed-fire points inside CMAQ grid')
            else:
                print(f'  No prescribed fires after filtering')
        else:
            print(f'  CSV not found: {csv_path}')

        if not found_df and not GRID_EMPTY_DAYS:
            print(f'  Skipping {ymd} (no data and GRID_EMPTY_DAYS=False)')
            continue

        # --- Grid each emission species ------------------------------------
        ds_vars = {}

        for species, longname in SPECIES_DICT.items():

            if not found_df:
                arr = np.zeros((nrows, ncols), dtype=np.float64)
            else:
                if species == 'meanFRP':
                    arr = grid_species(df, species, grid_info, resample='mean')
                else:
                    arr = grid_species(df, species, grid_info, resample='sum')

            # Unit conversion
            if species == 'meanFRP':
                units = 'MW'
            elif species == 'FRE':
                units = 'MJ'
            else:
                # kg/day  ->  kg/m²/s
                arr = arr / cell_area / 86400.0
                units = 'kg/m2/s'

            da = xr.DataArray(
                arr[np.newaxis, :, :],      # (time, ROW, COL)
                dims=['TSTEP', 'ROW', 'COL'],
                attrs={'units': units, 'long_name': longname},
            )
            ds_vars[species] = da

        # Add 2-D lon/lat fields
        ds_vars['LON'] = xr.DataArray(
            lon2d, dims=['ROW', 'COL'],
            attrs={'units': 'degrees_east', 'long_name': 'longitude of cell centre'},
        )
        ds_vars['LAT'] = xr.DataArray(
            lat2d, dims=['ROW', 'COL'],
            attrs={'units': 'degrees_north', 'long_name': 'latitude of cell centre'},
        )

        # Assemble dataset
        ds_out = xr.Dataset(ds_vars)
        ds_out = ds_out.assign_coords(TSTEP=[pd.Timestamp(ymd)])

        # Global attributes
        ds_out.attrs['TITLE']   = 'GEUFE prescribed-fire emissions on CMAQ 12US1 grid'
        ds_out.attrs['HISTORY'] = f'Created by Grid_GEUFE_to_CMAQ.py on {pd.Timestamp.now()}'
        ds_out.attrs['GDTYP']   = np.int32(2)   # LCC
        ds_out.attrs['P_ALP']   = grid_info['P_ALP']
        ds_out.attrs['P_BET']   = grid_info['P_BET']
        ds_out.attrs['P_GAM']   = grid_info['P_GAM']
        ds_out.attrs['XCENT']   = grid_info['XCENT']
        ds_out.attrs['YCENT']   = grid_info['YCENT']
        ds_out.attrs['XORIG']   = grid_info['XORIG']
        ds_out.attrs['YORIG']   = grid_info['YORIG']
        ds_out.attrs['XCELL']   = grid_info['XCELL']
        ds_out.attrs['YCELL']   = grid_info['YCELL']
        ds_out.attrs['NCOLS']   = np.int32(ncols)
        ds_out.attrs['NROWS']   = np.int32(nrows)
        ds_out.attrs['FILTER']  = 'Ag_flag=False AND Wildfire_flag=False, states=FL/GA/SC (prescribed fires only)'

        # Write with compression
        encoding = {v: dict(zlib=True, complevel=4, shuffle=True)
                    for v in ds_out.data_vars}
        ds_out.to_netcdf(out_file, encoding=encoding)
        print(f'  -> saved {out_file}')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()