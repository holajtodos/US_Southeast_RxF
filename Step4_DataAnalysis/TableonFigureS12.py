# --- Imports ---
import os
import numpy as np
import pandas as pd
from scipy import stats

# Change directory 
os.getcwd()
print ('cwd is %s ' % (os.getcwd()))
dir_python_local = '/home/jh94030/scripts/python/postdoc_project/rxfire/figure'

dir_work = os.path.join(dir_python_local)
os.chdir(dir_work)
print ('cwd is %s ' % (os.getcwd()))

# ---------- Config ----------
base_dir = os.path.join("/home/jh94030/scripts/python/postdoc_project/rxfire/data", "collocated_mod_obs")  # base dir that contains aq_SE_2017/, aq_SE_2018/, ...
years = [2017, 2018, 2019]
states = ["Florida", "Georgia", "South Carolina"]
out_csv = os.path.join(dir_python_local, "Table_overall_P-tot_PM25_metrics_AllYears.csv")

# ---------- Metric helpers ----------
def remove_nan_values(prediction, observation):
    p = np.asarray(prediction, dtype=float)
    o = np.asarray(observation, dtype=float)
    mask = (~np.isnan(p)) & (~np.isnan(o))
    return p[mask], o[mask]

def Npairs(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    return int(len(p))

def MB(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) == 0: return np.nan
    return float(np.sum(p - o) / len(p))

def ME(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) == 0: return np.nan
    return float(np.sum(np.abs(p - o)) / len(p))

def RMSE(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) == 0: return np.nan
    return float(np.sqrt(np.sum((p - o) ** 2) / len(p)))

def CRMSE(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) == 0: return np.nan
    p_mean, o_mean = np.mean(p), np.mean(o)
    return float(np.sqrt(np.sum(((p - p_mean) - (o - o_mean)) ** 2) / len(p)))

def NMB(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    den = np.sum(o)
    if den == 0: return np.nan
    return float(np.sum(p - o) / den * 100.0)

def NME(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    den = np.sum(o)
    if den == 0: return np.nan
    return float(np.sum(np.abs(p - o)) / den * 100.0)

def MNB(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    mask = o > 0
    if not np.any(mask): return np.nan
    p, o = p[mask], o[mask]
    return float(np.mean((p - o) / o) * 100.0)

def MNE(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    mask = o > 0
    if not np.any(mask): return np.nan
    p, o = p[mask], o[mask]
    return float(np.mean(np.abs(p - o) / o) * 100.0)

def FB(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    denom = p + o
    mask = denom > 0
    if not np.any(mask): return np.nan
    return float(2.0 * np.mean((p[mask] - o[mask]) / denom[mask]) * 100.0)

def FE(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    denom = p + o
    mask = denom > 0
    if not np.any(mask): return np.nan
    return float(2.0 * np.mean(np.abs(p[mask] - o[mask]) / denom[mask]) * 100.0)

def IOA(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) == 0: return np.nan
    num = np.sum((p - o) ** 2)
    denom = np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    if denom == 0: return np.nan
    return float(1.0 - num / denom)

def spearman_r(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) < 2: return np.nan
    r, _ = stats.spearmanr(p, o)
    return float(r)

def spearman_p(prediction, observation):
    p, o = remove_nan_values(prediction, observation)
    if len(p) < 2: return np.nan
    _, pval = stats.spearmanr(p, o)
    return float(pval)

metrics_dict = {
    "# Pairs": Npairs,
    "NMB (%)": NMB,
    "NME (%)": NME,
    "r (Spearman)": spearman_r,
    "MB": MB,
    "ME": ME,
    "RMSE": RMSE,
    "CRMSE": CRMSE,
    "MNB (%)": MNB,
    "MNE (%)": MNE,
    "FB (%)": FB,
    "FE (%)": FE,
    "IOA": IOA,
    "Spearman p": spearman_p,
}

# ---------- Load & combine data ----------
dfs = []
for y in years:
    fp = os.path.join(base_dir, f"aq_SE_{y}", f"AQS_Daily_aq_SE_{y}_with_smoke_day.csv")
    df = pd.read_csv(fp)
    df["YEAR"] = y
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Optional: clip tiny negatives to avoid log/ratio headaches elsewhere
for col in ["PM_TOT_mod", "PM_TOT_ob"]:
    if col in df_all.columns:
        df_all[col] = np.clip(df_all[col].astype(float), 1e-6, None)

# ---------- Compute metrics for Overall + States ----------
def compute_metrics_for_slice(df_slice):
    pred = df_slice["PM_TOT_mod"].values
    obs  = df_slice["PM_TOT_ob"].values
    out = {}
    for name, fn in metrics_dict.items():
        try:
            out[name] = fn(pred, obs)
        except Exception:
            out[name] = np.nan
    return out

rows = []

# Overall (AllYears)
overall_metrics = compute_metrics_for_slice(df_all)
overall_metrics.update({"Year": "AllYears", "State": "Overall"})
rows.append(overall_metrics)

# Per state (AllYears)
for st in states:
    m = compute_metrics_for_slice(df_all[df_all["State"] == st])
    m.update({"Year": "AllYears", "State": st})
    rows.append(m)

# ---------- Assemble table ----------
table = pd.DataFrame(rows)

# Preferred column order (edit as you like)
col_order = [
    "Year", "State", "# Pairs",
    "NMB (%)", "NME (%)", "r (Spearman)", "Spearman p",
    "MB", "ME", "RMSE", "CRMSE",
    "MNB (%)", "MNE (%)", "FB (%)", "FE (%)", "IOA",
]
for c in col_order:
    if c not in table.columns:
        table[c] = np.nan
table = table[col_order]

# Rounding: ints for pairs, 2 decimals for others
table["# Pairs"] = table["# Pairs"].fillna(0).astype(int)
for c in table.columns:
    if c not in ["Year", "State", "# Pairs"]:
        table[c] = table[c].astype(float).round(2)

# Sort rows: FL, GA, SC, Overall
state_cat = pd.CategoricalDtype(["Florida", "Georgia", "South Carolina", "Overall"], ordered=True)
table["State"] = table["State"].astype(state_cat)
table = table.sort_values(["State"]).reset_index(drop=True)

# Save CSV (UTF-8 with BOM helps Excel show symbols correctly)
table.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {out_csv}")