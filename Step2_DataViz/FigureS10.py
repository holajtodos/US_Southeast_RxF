# -*- coding: utf-8 -*-
###############################################################################
# ts_model_vs_obs_P-tot.py 
# author: Jingting HUANG
# purpose: To plot time-series PM2.5 daily variations
#
# version history: 
#   05/06/2025 - original  
# data required: 
#   -
# usage:  
#   -
# to do: 
#   - 
# notes: 
#   -
# debugging: 
#   -
############################################################################### 
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.ticker as mticker
from matplotlib.patheffects import SimpleLineShadow, Normal

# Change directory 
os.getcwd()
print ('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/data'

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print ('cwd is %s ' % (os.getcwd()))
###############################################################################
# Set plot style and font
# SAFETY: don't crash if font files aren't present
try:
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
    font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")
    plt.rcParams['font.family'] = 'Arial'
except Exception as e:
    print(f"Warning: could not register Arial fonts: {e}")

years = [2017, 2018, 2019]

# Load all years of data
all_data = []
for year in years:
    file_path = os.path.join(dir_python_local, 'collocated_mod_obs', f'aq_SE_{year}_RXF', f'AQS_Daily_aq_SE_{year}_RXF_with_smoke_day.csv')
    df = pd.read_csv(file_path)
    # FIX: coerce numeric columns; prevents strings wrecking mean/clip
    for col in ['PM_TOT_ob', 'PM_TOT_mod', 'PM_TOT_mod_rxf', 'SYYYY', 'SMM', 'SDD']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# FIX: only clip if column exists
if 'PM_TOT_mod_rxf' in df_all.columns:
    df_all['PM_TOT_mod_rxf'] = df_all['PM_TOT_mod_rxf'].clip(lower=0)

# Create season column
# FIX: always create a proper datetime column named 'date' to rely on later
if 'date' not in df_all.columns:
    if set(['SYYYY', 'SMM', 'SDD']).issubset(df_all.columns):
        df_all['date'] = pd.to_datetime(dict(year=df_all['SYYYY'], month=df_all['SMM'], day=df_all['SDD']), errors='coerce')
    elif 'DATE' in df_all.columns:
        df_all['date'] = pd.to_datetime(df_all['DATE'], errors='coerce')
    else:
        raise KeyError("No columns found to construct 'date' (need SYYYY/SMM/SDD or DATE).")

if 'season' not in df_all.columns:
    df_all['season'] = df_all['date'].dt.month.map(lambda m: 'High-burn' if m <= 4 else 'Low-burn')

state_abbr = ['FL', 'GA', 'SC']

# FIX: handle STATE vs State column naming
state_col = 'State' if 'State' in df_all.columns else ('STATE' if 'STATE' in df_all.columns else None)
if state_col is None:
    raise KeyError("Neither 'State' nor 'STATE' column found in input data.")

for i, state in enumerate(['Florida', 'Georgia', 'South Carolina']):
    subset_state = df_all[df_all[state_col] == state]
    # SAFETY: skip gracefully if no data for a state
    if subset_state.empty:
        print(f"Warning: no records for {state}; skipping plot.")
        continue

    df_daily_avg = subset_state.groupby('date')[['PM_TOT_ob', 'PM_TOT_mod', 'PM_TOT_mod_rxf']].mean().reset_index()
    # FIX: ensure chronological order for plotting
    df_daily_avg = df_daily_avg.sort_values('date')

    # Plot time-series : CMAQ vs AQS for PM2.5
    fig, ax = plt.subplots(figsize=(7, 1.75), dpi=600)

    # Plot filled areas for modeled PM
    ax.fill_between(df_daily_avg['date'], df_daily_avg['PM_TOT_mod'],
                    color='#FEE4A6', alpha=0.7, label='CMAQ-predicted total ' + r'$\mathrm{PM}_{2.5}$ mean', zorder=2)
#                     color='#E1D7C0', alpha=0.7, label='CMAQ-predicted total ' + r'$\mathrm{PM}_{2.5}$ mean', zorder=2)

    if 'PM_TOT_mod_rxf' in df_daily_avg.columns:
        ax.fill_between(df_daily_avg['date'], df_daily_avg['PM_TOT_mod_rxf'],
                        color='red', alpha=0.5, label='CMAQ-predicted Rx fire ' + r'$\mathrm{PM}_{2.5}$ mean', zorder=3)

    # Line plot for observed PM
    ax.plot(df_daily_avg['date'], df_daily_avg['PM_TOT_ob'], linestyle='-', linewidth=1.5,
            color='#2B6688', label='Observed total ' + r'$\mathrm{PM}_{2.5}$ mean', zorder=4)
         #   path_effects=[SimpleLineShadow(shadow_color="grey", linewidth=0.5), Normal()], zorder=4)
    
    ax.set_xlim([pd.Timestamp('2016-12-31'), pd.Timestamp('2019-12-31')])
    
    if i == 0:
        ax.set_ylim([-2, 40])
    elif i == 1:
        ax.set_ylim([-2, 50])
    else:
        ax.set_ylim([-2, 40])

    # set y-axis ticks every 10
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax.set_ylabel(r"$\mathrm{PM}_{2.5}$"+" (µg/m³)", fontsize=11)

    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.set_title(f"MOD vs. OBS in {state_abbr[i]}", loc='left', fontsize=12, fontweight='bold')
    if i == 0:
        ax.legend(loc='upper center', fontsize=7, bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=False)
    
    # Add season indicators as horizontal lines at the bottom (instead of using x-axis label)
    ymin, ymax = ax.get_ylim()
    y_pos = ymin - (ymax - ymin) * 0.02  # slightly below bottom axis

    season_ranges = [
        ('2017 High-burn', '2017-01-01', '2017-04-30'),
        ('2017 Low-burn',  '2017-05-01', '2017-12-31'),
        ('2018 High-burn', '2018-01-01', '2018-04-30'),
        ('2018 Low-burn',  '2018-05-01', '2018-12-31'),
        ('2019 High-burn', '2019-01-01', '2019-04-30'),
        ('2019 Low-burn',  '2019-05-01', '2019-12-31'),
    ]

    for label, start_str, end_str in season_ranges:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        ax.hlines(y=y_pos, xmin=start, xmax=end, colors='k', linewidth=1.5, clip_on=False)
        ax.text(x=start + (end - start)/2, y=y_pos - (ymax - ymin)*0.03, s=label,
                ha='center', va='top', fontsize=6)
        
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='gray', zorder=1)

    # Hide all x tick labels
    ax.tick_params(axis='x', which='both', labelbottom=False)

    # FIX: keep season labels from getting cropped by tight bbox
    plt.subplots_adjust(bottom=0.22)

    out_fn = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure/'+f'{state_abbr[i]}_ts_mod_vs_obs_pm25.png'
    plt.savefig(out_fn, bbox_inches='tight', dpi=600)
    plt.close(fig)  # FIX: free memory across loop
    print(f"Saved {out_fn}")