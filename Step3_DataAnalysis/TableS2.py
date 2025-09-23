# --- Imports ---
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# Change directory 
os.getcwd()
print ('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'

# Append the location of our function directory
dir_python_scripts = '/home/jh94030/scripts/python/postdoc_project/rxfire/analysis'
sys.path.append(os.path.join(dir_python_scripts,'step3_BurnDataSelection'))

from util import CMAQGrid2D

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print ('cwd is %s ' % (os.getcwd()))

# ------------- Config you can tweak -------------
dir_data = os.path.join("/home/jh94030/scripts/python/postdoc_project/rxfire/data", "collocated_mod_obs")  # base dir that contains aq_SE_2017/, aq_SE_2018/, ...
years = [2017, 2018, 2019]
states = ["Florida", "Georgia", "South Carolina"]  # we'll add "Full region" automatically
seasons_labels = {"High-burn": lambda m: m <= 4, "Low-burn": lambda m: m >= 5}
out_csv = os.path.join(dir_python_local, "Table_S4_P-tot_PM25_metrics.csv")
# -----------------------------------------------

# ---------- Metric helpers ----------
def _remove_nan_values(p, o):
    p = np.asarray(p, dtype=float)
    o = np.asarray(o, dtype=float)
    mask = (~np.isnan(p)) & (~np.isnan(o))
    return p[mask], o[mask]

def _clip_small(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    return np.clip(x, eps, None)

def _npairs(p, o):
    p, o = _remove_nan_values(p, o)
    return int(len(p))

def _nmb(p, o):
    p, o = _remove_nan_values(p, o)
    den = np.sum(o)
    if den == 0:
        return np.nan
    return (np.sum(p - o) / den) * 100.0

def _nme(p, o):
    p, o = _remove_nan_values(p, o)
    den = np.sum(o)
    if den == 0:
        return np.nan
    return (np.sum(np.abs(p - o)) / den) * 100.0

def _spearman_r(p, o):
    p, o = _remove_nan_values(p, o)
    if len(p) < 2:
        return np.nan
    r, _ = stats.spearmanr(p, o)
    return float(r)

def compute_metrics_from_df(df_slice):
    # Use original (non-log) values; clip tiny/negative to small positive to avoid division issues
    pred = _clip_small(df_slice["PM_TOT_mod"].values)
    obs  = _clip_small(df_slice["PM_TOT_ob"].values)
    return {
        "# Pairs": _npairs(pred, obs),
        "NMB (%)": _nmb(pred, obs),
        "NME (%)": _nme(pred, obs),
        "r": _spearman_r(pred, obs),
    }

# ---------- Load & prepare all data ----------
all_list = []
for y in years:
    fp = os.path.join(dir_data, f"aq_SE_{y}", f"AQS_Daily_aq_SE_{y}_with_smoke_day.csv")
    df = pd.read_csv(fp)
    df["YEAR"] = y  # ensure year present
    all_list.append(df)

df_all = pd.concat(all_list, ignore_index=True)

# Create season column if not present
if "season" not in df_all.columns:
    df_all["date"] = pd.to_datetime(dict(year=df_all["SYYYY"], month=df_all["SMM"], day=df_all["SDD"]))
    df_all["season"] = np.where(df_all["date"].dt.month <= 4, "High-burn", "Low-burn")

# ---------- Build Table S4 ----------
rows = []
state_labels = states + ["Full region"]

for season_label in ["High-burn", "Low-burn"]:
    is_season = df_all["season"].str.lower() == season_label.lower()

    # Full region first (overall across all states)
    df_season_all = df_all[is_season]
    overall_metrics = compute_metrics_from_df(df_season_all)

    # Per-year metrics for full region
    yr_metrics = {}
    for y in years:
        df_y = df_season_all[df_season_all["YEAR"] == y]
        m = compute_metrics_from_df(df_y)
        yr_metrics[f"NMB_{str(y)[-2:]} (%)"] = m["NMB (%)"]
        yr_metrics[f"NME_{str(y)[-2:]} (%)"] = m["NME (%)"]
        yr_metrics[f"r_{str(y)[-2:]}"]       = m["r"]

    rows.append({
        "State": "Full region",
        "Burn season": season_label,
        "# Pairs": overall_metrics["# Pairs"],
        "NMB (%)": overall_metrics["NMB (%)"],
        "NME (%)": overall_metrics["NME (%)"],
        "r": overall_metrics["r"],
        **yr_metrics
    })

    # Now each state
    for st in states:
        df_state_season = df_season_all[df_season_all["State"] == st]
        st_overall = compute_metrics_from_df(df_state_season)

        st_yr_metrics = {}
        for y in years:
            df_st_y = df_state_season[df_state_season["YEAR"] == y]
            m = compute_metrics_from_df(df_st_y)
            st_yr_metrics[f"NMB_{str(y)[-2:]} (%)"] = m["NMB (%)"]
            st_yr_metrics[f"NME_{str(y)[-2:]} (%)"] = m["NME (%)"]
            st_yr_metrics[f"r_{str(y)[-2:]}"]       = m["r"]

        rows.append({
            "State": st,
            "Burn season": season_label,
            "# Pairs": st_overall["# Pairs"],
            "NMB (%)": st_overall["NMB (%)"],
            "NME (%)": st_overall["NME (%)"],
            "r": st_overall["r"],
            **st_yr_metrics
        })

# Make DataFrame with your exact column order
col_order = [
    "State", "Burn season", "# Pairs",
    "NMB (%)", "NME (%)", "r",
    "NMB_17 (%)", "NME_17 (%)", "r_17",
    "NMB_18 (%)", "NME_18 (%)", "r_18",
    "NMB_19 (%)", "NME_19 (%)", "r_19",
]
table_s4 = pd.DataFrame(rows)

# Ensure all expected columns exist (in case of totally empty subsets)
for c in col_order:
    if c not in table_s4.columns:
        table_s4[c] = np.nan

table_s4 = table_s4[col_order]

# Round numeric columns nicely (2 decimals for % and r), keep # Pairs as int
pct_cols = [c for c in table_s4.columns if "(%)" in c]
r_cols   = ["r", "r_17", "r_18", "r_19"]

for c in pct_cols + r_cols:
    table_s4[c] = table_s4[c].astype(float).round(2)

table_s4["# Pairs"] = table_s4["# Pairs"].fillna(0).astype(int)

# Sort rows in a readable order: FL, GA, SC, Full region; High-burn then Low-burn
state_cat = pd.CategoricalDtype(["Florida", "Georgia", "South Carolina", "Full region"], ordered=True)
season_cat = pd.CategoricalDtype(["High-burn", "Low-burn"], ordered=True)
table_s4["State"] = table_s4["State"].astype(state_cat)
table_s4["Burn season"] = table_s4["Burn season"].astype(season_cat)
table_s4 = table_s4.sort_values(["State", "Burn season"]).reset_index(drop=True)

# Save to CSV
table_s4.to_csv(out_csv, index=False)
print(f"\nSaved to: {out_csv}")