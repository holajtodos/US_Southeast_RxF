# Figures S6
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter  # kept for parity with your original

# -------------------- Config --------------------
ARIAL_PATH_1 = "/home/jh94030/fonts/Arial.ttf"
ARIAL_PATH_2 = "/home/jh94030/fonts/Arial Bold.ttf"

dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/data'
DATA_DIR = f"{dir_python_local}/SE_permit_data_2010-2020/update_criteria"   # expects dir_python_local in env
YEARS = [2017, 2018, 2019]

STATE_ORDER  = ['FL', 'GA', 'SC']
STATE_LABELS = {'FL': 'FL', 'GA': 'GA', 'SC': 'SC'}
COLORS       = {'FL': '#D3E6D0', 'GA': '#E2D5E7', 'SC': '#FFF1CC'}

OUT_DIR = "/home/jh94030/scripts/python/postdoc_project/rxfire/figure"
OUT_MONTHLY = os.path.join(OUT_DIR, "monthly_temporal_profiles.png")
OUT_WEEKLY  = os.path.join(OUT_DIR, "weekly_temporal_profiles.png")

DOW_FULL = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DOW_ABBR = ['M', 'T', 'W', 'T', 'F', 'S', 'S']
DOW_TO_IDX = {d: i for i, d in enumerate(DOW_FULL)}

# -------------------- Helpers --------------------
def use_arial(path: str):
    """Register and set Arial as default font."""
    font_manager.fontManager.addfont(path)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']

def load_years(data_dir: str, years):
    """Read and concatenate the CSVs for the specified years with DATE parsing."""
    frames = []
    for yr in years:
        fn = os.path.join(data_dir, f"SE_Combined_Permit_lf_3states_rx_{yr}.csv")
        df = pd.read_csv(fn, parse_dates=['DATE'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# -------------------- Main workflow --------------------
use_arial(ARIAL_PATH_1)
use_arial(ARIAL_PATH_2)

data = load_years(DATA_DIR, YEARS)

# ---------- Figure S6: Monthly temporal profiles ----------
data['month'] = data['DATE'].dt.month

# Sum by STATE × YEAR × month
monthly = (
    data.groupby(['STATE', 'YEAR', 'month'], as_index=False)['ACRES']
        .sum()
)

# Ensure full coverage (all states, all years, all 12 months)
full_month_index = pd.MultiIndex.from_product(
    [STATE_ORDER, YEARS, np.arange(1, 13)], names=['STATE', 'YEAR', 'month']
)
monthly = (
    monthly.set_index(['STATE', 'YEAR', 'month'])
           .reindex(full_month_index, fill_value=0)
           .reset_index()
)

# Mean & std across years for each STATE × month
stats_month = (
    monthly.groupby(['STATE', 'month'])['ACRES']
           .agg(mean='mean', std='std')
           .reset_index()
)
stats_month['std'] = stats_month['std'].fillna(0)

# Plot (keep your sizes/lines)
fig, ax = plt.subplots(figsize=(7, 2.8))

for st in STATE_ORDER:
    sub = stats_month[stats_month['STATE'] == st]
    months = sub['month'].to_numpy()
    mean = sub['mean'].to_numpy()
    std  = sub['std'].to_numpy()

    ax.plot(months, mean, label=STATE_LABELS[st], color=COLORS[st], linewidth=4)
    ax.fill_between(months, mean - std, mean + std, color=COLORS[st], alpha=0.3)

# Season boundary & label (uses current y-lims)
ax.axvline(x=4.5, color='#AA3A49', linestyle='-', linewidth=2)
ax.text(
    4.5, ax.get_ylim()[1] * 1.02, 'high-burn → low-burn',
    rotation=0, va='bottom', ha='center', fontsize=15, color='#AA3A49'
)

# Axis styling (unchanged)
ax.set_ylim(-0.5e5, 6e5)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax.set_xlabel('Month', fontsize=14, labelpad=10)
ax.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
ax.yaxis.get_major_formatter().set_useMathText(True)
ax.set_ylabel('Reported acres burned', fontsize=14, labelpad=10)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, axis='y', linestyle=':', linewidth=0.5)
ax.legend(fontsize=12, ncol=3, loc='upper right', frameon=False)

plt.savefig(OUT_MONTHLY, bbox_inches='tight', dpi=600)

# # ---------- Figure S6 (Inset): Weekly temporal profiles ----------
# # Day-of-week as ordered categorical
# data['day'] = pd.Categorical(
#     data['DATE'].dt.day_name(),
#     categories=DOW_FULL,
#     ordered=True
# )

# # Sum ACRES by STATE × YEAR × day
# daily = (
#     data.groupby(['STATE', 'YEAR', 'day'], as_index=False, observed=True)['ACRES']
#         .sum()
# )

# # Ensure each STATE × YEAR has all 7 days
# full_dow_index = pd.MultiIndex.from_product(
#     [STATE_ORDER, YEARS, DOW_FULL], names=['STATE', 'YEAR', 'day']
# )
# daily = (
#     daily.set_index(['STATE', 'YEAR', 'day'])
#          .reindex(full_dow_index, fill_value=0)
#          .reset_index()
# )

# # Weekly totals by STATE × YEAR
# weekly_totals = (
#     daily.groupby(['STATE', 'YEAR'])['ACRES']
#          .sum()
#          .reset_index()
#          .rename(columns={'ACRES': 'WEEK_TOTAL'})
# )

# # Merge and compute daily fraction with zero-guard
# daily = pd.merge(daily, weekly_totals, on=['STATE', 'YEAR'], how='left')
# daily['WEEK_TOTAL'] = daily['WEEK_TOTAL'].replace(0, np.nan)
# daily['FRACTION'] = daily['ACRES'] / daily['WEEK_TOTAL']

# # Mean & std across years for STATE × day
# stats_week = (
#     daily.groupby(['STATE', 'day'])['FRACTION']
#          .agg(mean='mean', std='std')
#          .reset_index()
# )
# stats_week[['mean', 'std']] = stats_week[['mean', 'std']].fillna(0)
# stats_week['day_idx'] = stats_week['day'].map(DOW_TO_IDX)

# # Plot (unchanged sizes)
# fig, ax = plt.subplots(figsize=(3.6, 2))

# for st in STATE_ORDER:
#     sub = stats_week[stats_week['STATE'] == st].sort_values('day_idx')
#     x = sub['day_idx'].to_numpy()
#     mean = sub['mean'].to_numpy()
#     std  = sub['std'].to_numpy()

#     ax.plot(x, mean, label=STATE_LABELS[st], color=COLORS[st], linewidth=2)
#     ax.fill_between(x, mean - std, mean + std, color=COLORS[st], alpha=0.3)

# # Axis styling (unchanged)
# ax.set_ylim(0, 0.28)
# ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
# ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
# ax.set_xlabel('Day of Week', fontsize=12, labelpad=10)
# ax.set_xticks(np.arange(7))
# ax.set_xticklabels(DOW_ABBR)
# ax.set_ylabel('Fraction', fontsize=12, labelpad=10)
# ax.tick_params(axis='x', labelsize=10)
# ax.tick_params(axis='y', labelsize=9)
# ax.grid(True, axis='y', linestyle=':', linewidth=0.5)
# ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 0.6), frameon=False)

# plt.tight_layout()
# plt.savefig(OUT_WEEKLY, bbox_inches='tight', dpi=600)