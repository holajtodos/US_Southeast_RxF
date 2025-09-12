import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
from shapely.errors import ShapelyDeprecationWarning
import warnings

# ----------------------------- Environment & Paths -----------------------------
warnings.filterwarnings("ignore")

# PROJ paths: set via Python so they persist
os.environ["PROJ_LIB"] = "/home/jh94030/.conda/envs/myenv/share/proj"
os.environ["PROJ_DATA"] = "/home/jh94030/.conda/envs/myenv/share/proj"

# 1) Register your local copy of Arial
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

# 2) Force Matplotlib to use it
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Change directory
print('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure/SE_permit_data_2010-2020'
dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print('cwd is %s ' % (os.getcwd()))

# ----------------------------- Data Sources -----------------------------------
shapefile_dir = '/work/chflab/jthuang/breadcrumbs/mapping_state'
shapefile_path = os.path.join(shapefile_dir, 'cb_2020_us_state_500k', 'cb_2020_us_state_500k.shp')
assert os.path.exists(shapefile_path), f"Shapefile not found: {shapefile_path}"

# CSV templates for 2017–2019 fire points
csv_tpl = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/SE_Combined_Permit_lf_rx_{year}.csv"
for _y in (2017, 2018, 2019):
    _p = csv_tpl.format(year=_y)
    assert os.path.exists(_p), f"CSV not found: {_p}"

# ----------------------------- Load Boundaries --------------------------------
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    gdf = gpd.read_file(shapefile_path)  # states polygon layer (likely EPSG:4269/4326)

# Define Southeastern US states
SE_states = ['FL', 'GA', 'SC', 'NC', 'MS', 'TN', 'AL']
gdf_SE = gdf[gdf['STUSPS'].isin(SE_states)]

# ----------------------------- Style Constants --------------------------------
year_colors = {2017: '#0D1282', 2018: '#F0DE36', 2019: '#D71313'}
year_sizes  = {2017: 0.5, 2018: 0.005, 2019: 0.005}    # do not change per request
year_zorders = {2017: 1, 2018: 2, 2019: 3}          # do not change per request

state_labels = {
    "FL": (-82, 28.5), "GA": (-83.8, 32.5), "SC": (-81.24, 33.7),
    "TN": (-87, 35.8), "NC": (-79.5, 35.5), "MS": (-90.1, 33), "AL": (-87.2, 32.9)
}

# Rectangle for CMAQ domain (numbers unchanged)
rect_lon_max = [-76.3572 - 0.5, -72.791]
rect_lat_max = [35.0002 + 1.5, 37.3771 + 1.5]
rect_lon_min = [-89.6785 - 3.5, -88.3638 - 4]
rect_lat_min = [22.6929, 24.5888]

# ----------------------------- Figure & Axes ----------------------------------
fig = plt.figure(figsize=(3.33, 3))
# Main map: Albers Equal Area for CONUS
ax_main = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                       projection=ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37))

# Base features (zorder chosen to keep states/points visible)
ax_main.add_feature(cfeature.LAND, facecolor="lightgray")
ax_main.add_feature(cfeature.LAKES, edgecolor='gray', facecolor="white", linewidth=0.5)
ax_main.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=4)
ax_main.add_feature(cfeature.STATES, edgecolor="white", linewidth=0.5, zorder=4)

# Plot SE state boundaries; shapefile is lon/lat so use a geographic transform
gdf_SE.boundary.plot(ax=ax_main, edgecolor='k', linewidth=0.7, transform=ccrs.Geodetic(), zorder=5)

# ----------------------------- State Labels -----------------------------------
for state, (lon, lat) in state_labels.items():
    color = 'k' if state in ['FL', 'GA', 'SC'] else 'slategrey'
    weight = 'bold' if state in ['FL', 'GA', 'SC'] else 'semibold'
    t = ax_main.text(lon, lat, state, color=color, fontweight=weight, fontsize=6,
                     transform=ccrs.Geodetic(), zorder=10)
    # White halo for readability on any background
    t.set_path_effects([path_effects.Stroke(linewidth=1,
                                            foreground='white' if state in ['FL', 'GA', 'SC'] else 'snow'),
                        path_effects.Normal()])

# ----------------------------- Fire Points ------------------------------------
for year in [2017, 2018, 2019]:
    file_path = csv_tpl.format(year=year)
    df_fire = pd.read_csv(file_path)

    # Minimal sanity check for expected columns and valid coordinates
    if not {'LONGITUDE', 'LATITUDE'}.issubset(df_fire.columns):
        raise KeyError(f"CSV {file_path} must contain 'LONGITUDE' and 'LATITUDE' columns.")
    df_fire = df_fire.dropna(subset=['LONGITUDE', 'LATITUDE'])

    # Points are in lon/lat degrees; tell Cartopy via a geographic transform
    ax_main.scatter(
        df_fire['LONGITUDE'], df_fire['LATITUDE'],
        facecolors='none',                        # no fill
        edgecolors=year_colors[year],             # colored edges by year
        s=year_sizes[year],
        linewidths=0.5,                           # control edge thickness
        alpha=0.9,
        transform=ccrs.Geodetic(),
        zorder=year_zorders[year]
    )

# ----------------------------- Legend -----------------------------------------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='2017',
           markerfacecolor='w', markersize=3, markeredgecolor=year_colors[2017]),
    Line2D([0], [0], marker='o', color='w', label='2018',
           markerfacecolor='w', markersize=3, markeredgecolor=year_colors[2018]),
    Line2D([0], [0], marker='o', color='w', label='2019',
           markerfacecolor='w', markersize=3, markeredgecolor=year_colors[2019])
]
ax_main.legend(handles=legend_elements, title="Rx Fire Records", loc='center right',
               fontsize=4.5, title_fontsize=5)

# ----------------------------- Inset Map --------------------------------------
ax_inset = fig.add_axes([0.25, 0.1, 0.25, 0.25],
                        projection=ccrs.LambertConformal(central_longitude=-97.0, central_latitude=40.0))
ax_inset.set_extent([-125, -70, 19, 51], crs=ccrs.PlateCarree())
ax_inset.add_feature(cfeature.LAND, facecolor="lightgray")
ax_inset.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)

# Draw domain rectangle with given coordinates (unchanged)
ax_inset.plot(
    [rect_lon_min[1], rect_lon_min[0], rect_lon_max[0], rect_lon_max[1], rect_lon_min[1]],
    [rect_lat_max[1], rect_lat_min[1], rect_lat_min[0], rect_lat_max[0], rect_lat_max[1]],
    color="k", linewidth=1, transform=ccrs.PlateCarree()
)

# ----------------------------- Title & Source ---------------------------------
# ax_main.set_title("The Southeast Prescribed (Rx) Fire Permit Database: 2017–2019",
#                   loc='left', fontsize=5, pad=20)
# ax_main.text(-92.05, 38.7,
#              "Source: Tall Timbers Geospatial Lab and Fire Ecology Program",
#              color='grey', fontsize=5, transform=ccrs.Geodetic())

# ----------------------------- Save -------------------------------------------
fig.savefig("SE_RxF_spatial_map_2017_2019.png", dpi=600, bbox_inches='tight')