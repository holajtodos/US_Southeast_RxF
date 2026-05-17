# Figure S8
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import os

dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/data'

# Load Arial font
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial.ttf")
font_manager.fontManager.addfont("/home/jh94030/fonts/Arial Bold.ttf")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Define pastel state colors
state_colors = {'FL': '#CFF800', 'GA': '#B983FF', 'SC': '#FCD307'}

# Load and combine data
years = [2017, 2018, 2019]
states = ['FL', 'GA', 'SC']
dfs = []
for yr in years:
    df = pd.read_csv(f"{dir_python_local}/SE_permit_data_2010-2020/update_criteria/SE_Combined_Permit_lf_3states_rx_{yr}.csv", parse_dates=['DATE'])
    df['YEAR'] = yr
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

# Extract month
data['month'] = data['DATE'].dt.month

# Compute monthly burn count per year per state
monthly_counts = (
    data.groupby(['STATE', 'YEAR', 'month'])
    .size()
    .reset_index(name='burn_count')
)

# Compute mean and std across years for each state and month
stats = (monthly_counts
         .groupby(['STATE','month'])['burn_count']
         .agg(mean='mean', std='std')
         .reset_index()
         .sort_values(['STATE','month']))

# Month labels
month_labels = ['J','F','M','A','M','J',
                'J','A','S','O','N','D']

state_markers = {
    'FL': 'o',   # circle
    'GA': 's',   # square
    'SC': 'D'    # diamond
}

# Plotting
fig, ax = plt.subplots(figsize=(7, 3.5))
    
for st in states:
    sub = stats[stats['STATE'] == st]
    months = sub['month']
    means = sub['mean']
    stds = sub['std']
    marker = state_markers[st]

    # Plot error bars only
    ax.errorbar(months, means, yerr=stds, fmt='none',
                ecolor=state_colors[st], elinewidth=1.5, capsize=6, zorder=1)

    # Plot scatter points with custom marker
    ax.plot(months, means, linestyle='-', linewidth=2.5,
            marker=marker, markersize=8,
            markerfacecolor=state_colors[st],
            markeredgecolor='black',
            color=state_colors[st],
            label=st, zorder=2)

# Add season boundary line
ax.axvline(x=4.5, color='#AA3A49', linestyle='-', linewidth=2)
ax.text(4.5, 1.02, 'high-burn → low-burn',
        transform=ax.get_xaxis_transform(),  # y relative to axes
        va='bottom', ha='center', fontsize=15, color='#AA3A49')

# Aesthetics
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels, fontsize=12)
ax.set_ylabel("# Burns", fontsize=14)
ax.legend(fontsize=15, loc='upper right', ncol=3, frameon=False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.5)
ax.tick_params(axis='y', labelsize=12)

# Save figure
output_path = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'
os.makedirs(output_path, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'monthly_burn_frequency_with_errorbars.png'),
            bbox_inches='tight', dpi=600)