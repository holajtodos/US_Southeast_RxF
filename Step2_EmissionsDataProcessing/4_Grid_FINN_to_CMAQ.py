###############################################################################
# Grid_FINN_to_CMAQ.py
#
# Purpose: Regrid FINN point-based fire emissions (Jan–Apr, 2017-2019) onto
#          the CMAQ 12US1 Lambert Conformal Conic grid defined by a METCRO2D
#          file.  Agricultural fires have already been filtered out in the
#          input CSVs — the remaining data include both prescribed (rx) and
#          wildfire emissions together.
#
# Method:
#   1. Read CMAQ grid parameters (LCC projection, XORIG, YORIG, XCELL,
#      YCELL, NCOLS, NROWS) from a METCRO2D NetCDF file.
#   2. For each year's FINN CSV:
#        a. Parse dates, loop over each unique day.
#        b. Project fire LATI/LONGI to LCC (x, y) metres.
#        c. Assign each fire to a grid cell (col, row).
#        d. Sum emissions per grid cell; average for AREA.
#        e. Convert units:
#             - aerosols (BC, OC, PM25, PM10) & NOXasNO: kg/day  → kg/m²/s
#             - gas-phase (CO, CO2, SO2, NO, NO2, NH3):  mol/day → kg/m²/s
#               (multiplied by molecular weight / 1000)
#             - BMASS (kg/m² × burned-area m²):  kept as total kg per cell
#             - AREA:  kept as m² per cell  (no rate conversion).
#   3. Write one netCDF per day.
#
# Inputs:
#   - FINN CSV dir: /home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/FINN_rxf_inv/SE_Combined_FINN_rx_wf_{year}_Jan-Apr.csv
#   - METCRO2D:     /scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc
#
# Outputs:
#   - /home/jh94030/scripts/python/postdoc_project/rxfire/data/gridded_CMAQ_12US1/FINN_CMAQ12US1_<YYYYMMDD>.nc
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

YEARS = [2017, 2018, 2019]

# Paths
METCRO2D_FILE = r'/scratch/jh94030/CMAQ-input/met/12US1/nc_classic/2017/mcip_v51_wrf_v411_noltng/01/METCRO2D_20170101.nc'
FINN_CSV_DIR  = r'/home/jh94030/scripts/python/postdoc_project/rxfire/data/oth_fire_inv/FINN_rxf_inv'
OUTPUT_DIR    = os.path.join('/home/jh94030/scripts/python/postdoc_project/rxfire/data', 'gridded_CMAQ_12US1', 'FINN')

# If True, overwrite existing gridded files
OVERWRITE = True

# If True, create a file with zeros for days within Jan−Apr that have no fires
GRID_EMPTY_DAYS = True

# Unit conversion constants
ACRES_TO_M2 = 4046.8564224  # 1 acre = 4046.8564224 m²
SEC_PER_DAY = 86400.0

# Molecular weights (g/mol) for gas-phase species reported in moles/day
MOL_WEIGHT = {
    'CO':  28.010,
    'CO2': 44.010,
    'SO2': 64.066,
    'NO':  30.006,
    'NO2': 46.006,
    'NH3': 17.031,
}

# ---------- species classification by native FINN units ----------
# Aerosols + NOXasNO are reported in kg/day in FINN CSVs.
# All other gas-phase species are reported in moles/day.
MASS_SPECIES = {'OC', 'BC', 'PM25', 'PM10', 'NOXasNO'}
# (AREA and BMASS are treated separately — see below)

# Emission species to grid  (all summed per cell)
SPECIES_DICT = {
    'OC':      'Organic Carbon',
    'BC':      'Black Carbon',
    'PM25':    'Particulate Matter 2.5 um',
    'PM10':    'Particulate Matter 10 um',
    'CO':      'Carbon Monoxide',
    'CO2':     'Carbon Dioxide',
    'SO2':     'Sulfur Dioxide',
    'NO':      'Nitric Oxide',
    'NO2':     'Nitrogen Dioxide',
    'NOXasNO': 'Nitrogen Oxides as NO',
    'NH3':     'Ammonia',
    'BMASS':   'Biomass consumed', 
    'AREA':    'Burned area', 
}


# ===========================================================================
# 1. READ CMAQ GRID from METCRO2D  (same as Grid_GEUFE_to_CMAQ.py)
# ===========================================================================

def read_cmaq_grid(metcro_path):
    """
    Extract grid parameters and build cell-centre lat/lon from a
    CMAQ METCRO2D / GRIDCRO2D NetCDF file.
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

    proj4_lcc = (
        f"+proj=lcc +a=6370000.0 +b=6370000.0 "
        f"+lat_1={p_alp} +lat_2={p_bet} "
        f"+lat_0={ycent} +lon_0={xcent} "
        f"+x_0=0 +y_0=0 +units=m +no_defs"
    )
    transformer_to_lcc = Transformer.from_proj(
        "epsg:4326", proj4_lcc, always_xy=True
    )
    transformer_to_ll = Transformer.from_proj(
        proj4_lcc, "epsg:4326", always_xy=True
    )

    x_centers = np.linspace(xorig + xcell / 2,
                            xorig + xcell / 2 + xcell * (ncols - 1), ncols)
    y_centers = np.linspace(yorig + ycell / 2,
                            yorig + ycell / 2 + ycell * (nrows - 1), nrows)
    X2d, Y2d = np.meshgrid(x_centers, y_centers)
    lon2d, lat2d = transformer_to_ll.transform(X2d, Y2d)

    cell_area_m2 = xcell * ycell

    return dict(
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


# ===========================================================================
# 2. ASSIGN FIRES TO GRID CELLS
# ===========================================================================

def assign_fires_to_grid(df, grid_info):
    """
    Project fire lat/lon to LCC, compute (row, col) indices, drop points
    outside the grid.  Adds 'cmaq_row' and 'cmaq_col' columns.
    """
    transformer = grid_info['transformer_to_lcc']
    xorig = grid_info['XORIG']
    yorig = grid_info['YORIG']
    xcell = grid_info['XCELL']
    ycell = grid_info['YCELL']
    ncols = grid_info['NCOLS']
    nrows = grid_info['NROWS']

    df = df.copy()
    x_lcc, y_lcc = transformer.transform(df['LONGI'].values, df['LATI'].values)

    col_idx = np.floor((x_lcc - xorig) / xcell).astype(int)
    row_idx = np.floor((y_lcc - yorig) / ycell).astype(int)

    df['cmaq_col'] = col_idx
    df['cmaq_row'] = row_idx

    mask = (
        (col_idx >= 0) & (col_idx < ncols) &
        (row_idx >= 0) & (row_idx < nrows)
    )
    return df[mask].copy()


# ===========================================================================
# 3. SUM / AVERAGE EMISSIONS PER GRID CELL -> 2-D ARRAY
# ===========================================================================

def grid_species(df, species, grid_info, resample='sum'):
    """Aggregate a single species onto the CMAQ 2-D grid."""
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('Reading CMAQ grid from METCRO2D …')
    grid_info = read_cmaq_grid(METCRO2D_FILE)
    nrows = grid_info['NROWS']
    ncols = grid_info['NCOLS']
    cell_area = grid_info['cell_area_m2']
    print(f'  Grid: {ncols} cols x {nrows} rows,  cell: '
          f'{grid_info["XCELL"]/1000:.0f} x {grid_info["YCELL"]/1000:.0f} km\n')

    lon2d = grid_info['lon2d']
    lat2d = grid_info['lat2d']

    for year in YEARS:
        csv_path = os.path.join(
            FINN_CSV_DIR, f'SE_Combined_FINN_rx_wf_{year}_Jan-Apr.csv'
        )
        if not os.path.isfile(csv_path):
            print(f'  CSV not found: {csv_path}  — skipping year {year}')
            continue

        print(f'======= Loading FINN {year} =======')
        df_all = pd.read_csv(csv_path)
        df_all['DAY'] = pd.to_datetime(df_all['DAY'])

        # --- Unit conversions on raw data --------------------------------
        # AREA: acres → m²
        df_all['AREA'] = df_all['AREA'] * ACRES_TO_M2
        # BMASS: kg/m² (fuel loading) → total kg consumed per fire pixel
        #        multiply by burned area (now in m²) so it can be summed
        #        and converted to kg/m²/s like other emission species
        df_all['BMASS'] = df_all['BMASS'] * df_all['AREA']

        print(f'  Total fire points: {len(df_all)}')

        # Build complete Jan–Apr date range
        date_range = pd.date_range(f'{year}-01-01', f'{year}-04-30', freq='D')

        for date in date_range:
            ymd = date.strftime('%Y%m%d')
            out_file = os.path.join(OUTPUT_DIR, f'FINN_CMAQ12US1_{ymd}.nc')

            if os.path.exists(out_file) and not OVERWRITE:
                continue

            df_day = df_all[df_all['DAY'] == date]
            found_df = False

            if len(df_day) > 0:
                df_day = assign_fires_to_grid(df_day, grid_info)
                if len(df_day) > 0:
                    found_df = True
                    print(f'  {ymd}:  {len(df_day)} fire points on grid')
                else:
                    print(f'  {ymd}:  all fires outside CMAQ grid')
            else:
                print(f'  {ymd}:  no fires this day')

            if not found_df and not GRID_EMPTY_DAYS:
                continue

            # --- Grid each species -----------------------------------------
            ds_vars = {}
            for species, longname in SPECIES_DICT.items():
                if not found_df:
                    arr = np.zeros((nrows, ncols), dtype=np.float64)
                else:
                    # All species (including AREA and BMASS) are summed
                    arr = grid_species(df_day, species, grid_info,
                                       resample='sum')

                # ---------- Unit conversion --------------------------------
                if species == 'AREA':
                    # Already converted to m²; just sum per cell
                    units = 'm2'
                elif species == 'BMASS':
                    # Already converted to total kg (kg/m² × m²); keep as-is
                    units = 'kg'
                elif species in MASS_SPECIES:
                    # Aerosols + NOXasNO: native kg/day → kg/m²/s
                    arr = arr / cell_area / SEC_PER_DAY
                    units = 'kg/m2/s'
                else:
                    # Gas-phase species: native moles/day → kg/m²/s
                    #   mol/day × (MW g/mol × 1e-3 kg/g) ÷ cell_area ÷ 86400
                    mw = MOL_WEIGHT[species]
                    arr = arr * (mw * 1.0e-3) / cell_area / SEC_PER_DAY
                    units = 'kg/m2/s'

                da = xr.DataArray(
                    arr[np.newaxis, :, :],   # (time, ROW, COL)
                    dims=['TSTEP', 'ROW', 'COL'],
                    attrs={'units': units, 'long_name': longname},
                )
                ds_vars[species] = da

            # 2-D coordinate fields
            ds_vars['LON'] = xr.DataArray(
                lon2d, dims=['ROW', 'COL'],
                attrs={'units': 'degrees_east',
                       'long_name': 'longitude of cell centre'},
            )
            ds_vars['LAT'] = xr.DataArray(
                lat2d, dims=['ROW', 'COL'],
                attrs={'units': 'degrees_north',
                       'long_name': 'latitude of cell centre'},
            )

            ds_out = xr.Dataset(ds_vars)
            ds_out = ds_out.assign_coords(TSTEP=[pd.Timestamp(ymd)])

            # Global attributes
            ds_out.attrs['TITLE']  = ('FINN rx+wf fire emissions (ag-filtered) '
                                      'on CMAQ 12US1 grid')
            ds_out.attrs['HISTORY'] = (f'Created by Grid_FINN_to_CMAQ.py on '
                                       f'{pd.Timestamp.now()}')
            ds_out.attrs['GDTYP']  = np.int32(2)
            ds_out.attrs['P_ALP']  = grid_info['P_ALP']
            ds_out.attrs['P_BET']  = grid_info['P_BET']
            ds_out.attrs['P_GAM']  = grid_info['P_GAM']
            ds_out.attrs['XCENT']  = grid_info['XCENT']
            ds_out.attrs['YCENT']  = grid_info['YCENT']
            ds_out.attrs['XORIG']  = grid_info['XORIG']
            ds_out.attrs['YORIG']  = grid_info['YORIG']
            ds_out.attrs['XCELL']  = grid_info['XCELL']
            ds_out.attrs['YCELL']  = grid_info['YCELL']
            ds_out.attrs['NCOLS']  = np.int32(ncols)
            ds_out.attrs['NROWS']  = np.int32(nrows)
            ds_out.attrs['FILTER'] = ('Agricultural fires excluded; '
                                      'includes rx + wildfire')

            encoding = {v: dict(zlib=True, complevel=4, shuffle=True)
                        for v in ds_out.data_vars}
            ds_out.to_netcdf(out_file, encoding=encoding)

        print(f'  Year {year} complete.\n')

    print('=== Done ===')


if __name__ == '__main__':
    main()