#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib import font_manager

import cartopy
import cartopy.crs as ccrs

import sys
sys.path.append("/home/jh94030/scripts/python/postWRF/WRF-tools-master/WRF_input_tools/")
import WRFDomainLib


# ----------------------------
# User paths (edit as needed)
# ----------------------------
DEM_FILE = Path("/work/chflab/jthuang/breadcrumbs/ETOPO1/ETOPO1.0_1degree.nc")
WPS_FILE = Path("/home/jh94030/scripts/models/WRF_NAM/WPS_scripts/namelists/namelist_wrf_cmaq.wps")
OUT_PNG = Path("/home/jh94030/scripts/python/postdoc_project/rxfire/figure/WRF-CMAQ_domain.png")

# ------------------------- Fonts -------------------------
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

def open_dem(da_path: Path):
    """
    Open ETOPO1-like DEM and return (dem, lon2d, lat2d).
    Accepts variable names 'DEM' or 'z'.
    """
    if not da_path.exists():
        raise FileNotFoundError(f"DEM file not found: {da_path}")

    with xr.open_dataset(da_path) as ds:
        # try common var names
        if "DEM" in ds:
            dem = ds["DEM"].load().values
        elif "z" in ds:
            dem = ds["z"].load().values
        else:
            raise KeyError("DEM variable not found (looked for 'DEM' or 'z').")

        # lon/lat variable names
        lon_name = "lon" if "lon" in ds.variables else "longitude"
        lat_name = "lat" if "lat" in ds.variables else "latitude"
        if lon_name not in ds or lat_name not in ds:
            raise KeyError("Longitude/latitude variables not found in DEM file.")

        lon = ds[lon_name].values
        lat = ds[lat_name].values

    lon2d, lat2d = np.meshgrid(lon, lat)
    # clip negative elevation (below sea level) to 0 m
    dem = np.clip(dem, 0, None)
    return dem, lon2d, lat2d


def compute_domains(wps_file: Path):
    """
    Read WPS and compute projections, corners, and lengths.
    """
    if not wps_file.exists():
        raise FileNotFoundError(f"WPS file not found: {wps_file}")

    wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y = (
        WRFDomainLib.calc_wps_domain_info(str(wps_file))
    )
    return wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y


def add_domain_box(ax, proj_x, proj_y, dx, dy, *, lw=3, edgecolor="white", zorder=4, label=None):
    """
    Draw a rectangular domain box given lower-left (x,y) in map projection coords and dx/dy lengths.
    """
    rect = matplotlib.patches.Rectangle(
        (proj_x, proj_y), dx, dy, fill=False, lw=lw, edgecolor=edgecolor, zorder=zorder
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            proj_x + dx * 0.03, proj_y + dy * 0.04, label,
            fontweight="semibold", size=10, color=edgecolor, zorder=zorder
        )


def plot_map(dem, dem_lons, dem_lats,
             wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y,
             out_png: Path,
             vmin=0, vmax=3500):

    # Figure & axis
    fig = plt.figure(figsize=(6, 8), dpi=600)
    ax = plt.subplot(1, 1, 1, projection=wpsproj)

    # DEM pcolormesh
    cmap = matplotlib.colormaps.get_cmap("gist_yarg")
    ax.pcolormesh(
        dem_lons, dem_lats, dem,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        transform=ccrs.PlateCarree(), zorder=0
    )

    # Big map extent with small margins around d01 (index 0)
    corner_x1, corner_y1 = WRFDomainLib.reproject_corners(
        corner_lon_full[0, :], corner_lat_full[0, :], wpsproj, latlonproj
    )
    ax.set_xlim([corner_x1[0] - length_x[0] / 15, corner_x1[3] + length_x[0] / 15])
    ax.set_ylim([corner_y1[0] - length_y[0] / 15, corner_y1[3] + length_y[0] / 15])

    # Domain boxes
    # CMAQ d01 box (index 1)
    cx1_, cy1_ = WRFDomainLib.reproject_corners(
        corner_lon_full[1, :], corner_lat_full[1, :], wpsproj, latlonproj
    )
    add_domain_box(ax, cx1_[0], cy1_[0], length_x[1], length_y[1], edgecolor="white")
    # label example (optional): add_domain_box(..., label="EQUATES")

    # CMAQ d02 box (index 3) with your custom shift & scale
    cx2_, cy2_ = WRFDomainLib.reproject_corners(
        corner_lon_full[3, :] - 1.2, corner_lat_full[3, :] + 0.4, wpsproj, latlonproj
    )
    add_domain_box(ax, cx2_[0], cy2_[0], length_x[3] * 1.15, length_y[3] * 1.15, edgecolor="red")

    # Map decorations
    ax.add_feature(cartopy.feature.LAND, edgecolor="k", facecolor="none")
    ax.add_feature(cartopy.feature.LAKES, edgecolor="k", facecolor="deepskyblue")
    ax.add_feature(cartopy.feature.OCEAN, edgecolor="k", facecolor="deepskyblue", zorder=1)
    ax.coastlines("10m", color="black")
    ax.add_feature(cartopy.feature.STATES.with_scale("10m"))

    # Legends (axes-relative positions)
    # lines just for legend keys
    cmaq_d01_line = mlines.Line2D([], [], color="white", linewidth=3, label="EQUATES")
    cmaq_d02_line = mlines.Line2D([], [], color="red", linewidth=3, label="Study Area")

    leg1= ax.legend(
        handles=[cmaq_d01_line],
        loc='upper left',
        handlelength=0,
        handletextpad=0,
        bbox_to_anchor=(0.065, 0.195),
        bbox_transform=ax.transAxes,
        fontsize=10,
        frameon=False
    )
    ax.add_artist(leg1)
    
    leg2 = ax.legend(
        handles=[cmaq_d02_line],
        loc='upper left',
        handlelength=0,
        handletextpad=0,
        bbox_to_anchor=(0.555, 0.2),
        bbox_transform=ax.transAxes,
        fontsize=8,
        frameon=False
    )
    
    # bold + color text to match lines
    for leg in (leg1, leg2):
        for text, handle in zip(leg.get_texts(), leg.legend_handles):
            text.set_fontweight("bold")
            text.set_color(handle.get_color())

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linestyle="--", alpha=1, zorder=0
    )
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.x_inline = False
    gl.y_inline = False
    gl.xlocator = ticker.MultipleLocator(5)
    gl.ylocator = ticker.MultipleLocator(5)
    gl.xlabel_style = {"size": 9, "color": "black", "rotation": 45}
    gl.ylabel_style = {"size": 9, "color": "black"}

    # Colorbar (simple horizontal bar under plot)
    cax = fig.add_axes([0.22, 0.18, 0.6, 0.012])
    cb = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ),
        cax=cax, ticks=np.arange(0, vmax + 1, 1000),
        extend="max", orientation="horizontal"
    )
    cax.tick_params(labelsize=9)
    cax.text(0.5, 2.2, "Elevation (m)", ha="center", va="bottom", size=10, transform=cax.transAxes)

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    dem, lon2d, lat2d = open_dem(DEM_FILE)
    (wpsproj, latlonproj,
     corner_lat_full, corner_lon_full, length_x, length_y) = compute_domains(WPS_FILE)

    plot_map(
        dem, lon2d, lat2d,
        wpsproj, latlonproj, corner_lat_full, corner_lon_full, length_x, length_y,
        OUT_PNG
    )


if __name__ == "__main__":
    main()