# -*- coding: utf-8 -*-
"""
Build Table S5-S7: Size distribution of Rx burned areas (Permits vs NEI)
for Florida, Georgia, South Carolina (2017–2019).

Input:
- Permit files (one per year): SE_Combined_Permit_lf_3states_rx_{YYYY}.csv
  Columns required: STATE, YEAR, ACRES
- NEI files (one per year): SE_Combined_NEI_rx_3states_{YYYY}.csv
  Columns required: STATE, YEAR, ACRESBURNED

Output:
- CSVs:
  - Table_S5_Florida_Size_Distribution_2017_2019.csv
  - Table_S6_Georgia_Size_Distribution_2017_2019.csv
  - Table_S7_South_Carolina_Size_Distribution_2017_2019.csv
"""

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1) CONFIGURE YOUR PATHS / PATTERNS
# -------------------------------------------------------------------
BASE_DIR = "/home/jh94030/scripts/python/postdoc_project/rxfire/data"

# Permits: one file per year 2017..2019, same columns across years
PERMIT_DIR = os.path.join(BASE_DIR, "SE_permit_data_2010-2020", "update_criteria")
PERMIT_PATTERN = "SE_Combined_Permit_lf_3states_rx_{year}.csv"  # change to .parquet if needed

# NEI: yearly files
NEI_DIR = os.path.join(BASE_DIR, "oth_fire_inv", "NEI_rxf_inv")
NEI_PATTERN = "SE_Combined_NEI_rx_3states_{year}.csv"  # has ACRESBURNED

YEARS = [2017, 2018, 2019]
STATES = {
    "FL": "Florida",
    "GA": "Georgia",
    "SC": "South Carolina",
}

# Output directory
OUT_DIR = os.path.join("/home/jh94030/scripts/python/postdoc_project/rxfire/figure", "table_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Table filenames (S4/S5/S6)
OUT_FILES = {
    "FL": os.path.join(OUT_DIR, "Table_S5_Florida_Size_Distribution_2017_2019.csv"),
    "GA": os.path.join(OUT_DIR, "Table_S6_Georgia_Size_Distribution_2017_2019.csv"),
    "SC": os.path.join(OUT_DIR, "Table_S7_South_Carolina_Size_Distribution_2017_2019.csv"),
}

# -------------------------------------------------------------------
# 2) SIZE BINS (ACRES)
# -------------------------------------------------------------------
BINS = [0, 5, 10, 25, 50, 100, 250, 500, 1000, np.inf]
BIN_LABELS = [
    "0-5",
    "5-10",
    "10-25",
    "25-50",
    "50-100",
    "100-250",
    "250-500",
    "500-1000",
    "1000+"
]
SIZE_CAT = pd.api.types.CategoricalDtype(categories=BIN_LABELS, ordered=True)

# -------------------------------------------------------------------
# 3) HELPERS
# -------------------------------------------------------------------
def _read_year_file(dirpath, pattern, year):
    """
    Try reading CSV; if not found, try Parquet with same stub.
    Returns DataFrame or empty DataFrame if missing.
    """
    f_csv = os.path.join(dirpath, pattern.format(year=year))
    f_parq = os.path.splitext(f_csv)[0] + ".parquet"

    if os.path.exists(f_csv):
        return pd.read_csv(f_csv)
    elif os.path.exists(f_parq):
        return pd.read_parquet(f_parq)
    else:
        print(f"[WARN] File not found for year {year}: {f_csv} (or {f_parq})")
        return pd.DataFrame()

def load_permits(years=YEARS):
    dfs = []
    for y in years:
        df = _read_year_file(PERMIT_DIR, PERMIT_PATTERN, y)
        if df.empty:
            continue
        df.columns = df.columns.str.upper()
        keep = {"STATE", "YEAR", "ACRES"}
        # coerce YEAR
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        dfs.append(df[list(keep)])
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["STATE","YEAR","ACRES"])
    return out

def load_nei(years=YEARS):
    dfs = []
    for y in years:
        df = _read_year_file(NEI_DIR, NEI_PATTERN, y)
        if df.empty:
            continue
        df.columns = df.columns.str.upper()
        # map ACRESBURNED -> ACRES
        if "ACRESBURNED" not in df.columns:
            raise ValueError(f"NEI file {y} missing 'ACRESBURNED' column")
        df = df.rename(columns={"ACRESBURNED": "ACRES"})
        keep = {"STATE", "YEAR", "ACRES"}
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        dfs.append(df[list(keep)])
    if dfs:
        out = pd.concat(dfs, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["STATE","YEAR","ACRES"])
    return out

def size_counts(df):
    """
    df: columns STATE, YEAR, ACRES
    return: counts per (STATE, Size, YEAR) with all combos present
    """
    if df.empty:
        idx = pd.MultiIndex.from_product(
            [list(STATES.keys()), pd.Categorical(BIN_LABELS, categories=BIN_LABELS, ordered=True), YEARS],
            names=["STATE","Size","YEAR"]
        )
        return pd.DataFrame(index=idx).reset_index().assign(Count=0)

    tmp = df.dropna(subset=["STATE","YEAR","ACRES"]).copy()
    tmp["STATE"] = tmp["STATE"].astype(str).str.upper().str.strip()
    tmp["YEAR"]  = tmp["YEAR"].astype(int)

    # Bin areas
    tmp["Size"] = pd.cut(
        tmp["ACRES"], bins=BINS, labels=BIN_LABELS,
        right=True, include_lowest=True
    ).astype(SIZE_CAT)

    grp = (tmp.groupby(["STATE","Size","YEAR"], dropna=False)
              .size().rename("Count").reset_index())

    # Ensure all combinations exist
    idx = pd.MultiIndex.from_product(
        [list(STATES.keys()), pd.Categorical(BIN_LABELS, categories=BIN_LABELS, ordered=True), YEARS],
        names=["STATE","Size","YEAR"]
    )
    grp = grp.set_index(["STATE","Size","YEAR"]).reindex(idx, fill_value=0).reset_index()
    return grp

def counts_and_percents_by_state(counts_df, state_abbrev):
    """
    Return tuple (counts_wide, perc_wide) for a single state.
    - counts_wide: rows = Size bins, cols = 2017..2019
    - perc_wide: same shape, values = % of column total (0 if total=0)
    """
    sub = counts_df[counts_df["STATE"] == state_abbrev].copy()
    wide = sub.pivot_table(index="Size", columns="YEAR", values="Count", aggfunc="sum", fill_value=0).reindex(BIN_LABELS)
    # Ensure columns order
    wide = wide.reindex(columns=YEARS)

    totals = wide.sum(axis=0)  # per column (year)
    # Avoid division by zero
    denom = totals.replace(0, np.nan)
    perc = (wide.divide(denom, axis=1) * 100.0).fillna(0.0).round(1)

    return wide, perc, totals

def build_percent_table(tbl_perm_counts, tbl_nei_counts, state_abbrev):
    """
    Combine Permits + NEI percent tables side-by-side,
    plus a bottom row with total counts for each column.
    """
    # Permits
    p_counts, p_perc, p_totals = counts_and_percents_by_state(tbl_perm_counts, state_abbrev)
    # NEI
    n_counts, n_perc, n_totals = counts_and_percents_by_state(tbl_nei_counts, state_abbrev)

    # Label columns
    p_perc.columns = pd.MultiIndex.from_product([["Permits"], p_perc.columns])
    n_perc.columns = pd.MultiIndex.from_product([["NEI"], n_perc.columns])

    # Combine (percentages in body)
    combined_perc = pd.concat([p_perc, n_perc], axis=1)

    # Flatten headers to "Permits 2017", ...
    combined_perc.columns = [f"{src} {yr}" for (src, yr) in combined_perc.columns]

    # Add "Total # of fires" row (counts, not %)
    total_counts_row = pd.Series(
        data=list(p_totals.values) + list(n_totals.values),
        index=[f"Permits {y}" for y in YEARS] + [f"NEI {y}" for y in YEARS],
        name="Total # of fires"
    )
    combined_with_total = pd.concat([combined_perc, total_counts_row.to_frame().T], axis=0)

    # Insert first column with size labels
    combined_with_total.insert(0, "Size (acres)", list(BIN_LABELS) + ["Total # of fires"])

    # Ensure index reset
    return combined_with_total.reset_index(drop=True)

# -------------------------------------------------------------------
# 4) RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load sources
    permits = load_permits(YEARS)
    nei     = load_nei(YEARS)

    # Compute counts per bin (not % yet)
    perm_counts = size_counts(permits)
    nei_counts  = size_counts(nei)

    # Build and save % tables for each state
    for st_abbrev, st_full in STATES.items():
        table_percent = build_percent_table(perm_counts, nei_counts, st_abbrev)
        out_path = OUT_FILES[st_abbrev]
        table_percent.to_csv(out_path, index=False)
        print(f"[OK] Wrote {st_full}: {out_path}")

    print("\nDone.")