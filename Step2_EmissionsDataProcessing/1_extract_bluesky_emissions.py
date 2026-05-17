# -*- coding: utf-8 -*-
###############################################################################
# author: Jingting Huang
# purpose:
#   Extract selected point-based emissions from BlueSky JSON:
#   CO, CO2, NH3, NOx, PM2.5, PM10, SO2, VOC
###############################################################################

import os
import glob
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ----------------------------- USER SETTINGS -------------------------------
# ---------------------------------------------------------------------------

YEARS = [2017, 2018, 2019]

# BlueSky root; yearly folder assumed: {year}_SE/bsp_output
bluesky_root = "/scratch/jh94030/bsp"
file_glob_tpl = "SE_{year}_??_??_out.json"

# Optional: restrict within-year date window (inclusive). Set None for full year.
START_DATE = None  # e.g., "2017-01-01"
END_DATE   = None  # e.g., "2017-12-31"

# Optional: keep only these states if JSON has state codes (e.g., "FL","GA","SC"); set None for all
STATE_FILTER = {"FL", "GA", "SC"}  # or None

# Species to extract (exact keys used in BlueSky emissions.summary)
SPECIES = ["CO", "CO2", "NH3", "NOx", "PM2.5", "PM10", "SO2", "VOC"]

# Output
out_dir = "/home/jh94030/scripts/python/postdoc_project/rxfire/data/SE_permit_data_2010-2020/output_emis"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# ------------------------------ HELPERS ------------------------------------
# ---------------------------------------------------------------------------

def _parse_file_date_from_name(path: str) -> Optional[datetime]:
    try:
        date_str = os.path.basename(path)[3:13]  # 'YYYY_MM_DD'
        return datetime.strptime(date_str, "%Y_%m_%d")
    except Exception:
        return None

def _in_window(dt: datetime, start_dt: Optional[datetime], end_dt: Optional[datetime]) -> bool:
    if start_dt and dt < start_dt:
        return False
    if end_dt and dt > end_dt:
        return False
    return True

def _safe_float(x) -> float:
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _get_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    emis = obj.get("emissions") or {}
    summary = emis.get("summary") or {}
    return summary if isinstance(summary, dict) else {}

def extract_selected_point_emis(
    data: Dict[str, Any],
    file_date: datetime,
    state_filter: Optional[set] = None,
    species: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    One row per specified_point with selected species columns.
    Columns: date, year, fire_id, fire_type, activity_i, active_area_i, point_i,
             lat, lon, area_acres, state, ignition_start, ignition_end, [SPECIES...]
    """
    species = species or SPECIES
    recs: List[Dict[str, Any]] = []

    for fire in (data.get("fires", []) or []):
        fire_id = fire.get("id")
        fire_type = fire.get("type")

        for ai, activity in enumerate(fire.get("activity", []) or []):
            for aai, active_area in enumerate(activity.get("active_areas", []) or []):
                area_state = active_area.get("state")
                # ignition_start = active_area.get("ignition_start") or active_area.get("start")
                # ignition_end = active_area.get("ignition_end") or active_area.get("end")

                for pi, point in enumerate(active_area.get("specified_points", []) or []):
                    lat = point.get("lat")
                    lon = point.get("lng")
                    area_acres = point.get("area")
                    point_state = point.get("state", area_state)

                    if state_filter is not None and point_state is not None and point_state not in state_filter:
                        continue

                    summary = _get_summary(point)
                    if not summary:
                        continue

                    row = {
                        "state": point_state,
                        "date": file_date.strftime("%Y-%m-%d"),
                        "year": file_date.year,
                        "fire_id": fire_id,
                        "latitude": _safe_float(lat),
                        "longitude": _safe_float(lon),
                        "area_acres": _safe_float(area_acres),
                        # "fire_type": fire_type,
                        # "activity_i": ai,
                        # "active_area_i": aai,
                        # "point_i": pi,
                        # "ignition_start": ignition_start,
                        # "ignition_end": ignition_end,
                    }

                    for sp in species:
                        row[sp] = _safe_float(summary.get(sp))

                    recs.append(row)

    return pd.DataFrame.from_records(recs)

# ---------------------------------------------------------------------------
# --------------------------------- MAIN ------------------------------------
# ---------------------------------------------------------------------------

def process_year(year: int) -> None:
    bluesky_output_dir = os.path.join(bluesky_root, f"{year}_SE", "archived", "bsp_output_new_criteria", "bsp_output")
    pattern = os.path.join(bluesky_output_dir, file_glob_tpl.format(year=year))
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[{year}] WARNING: no files found: {pattern}")
        return

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d") if START_DATE else None
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d") if END_DATE else None

    dfs = []
    for fp in files:
        fdt = _parse_file_date_from_name(fp)
        if fdt is None or not _in_window(fdt, start_dt, end_dt):
            continue

        try:
            with open(fp, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[{year}] WARNING: failed to read {fp}: {e}")
            continue

        df = extract_selected_point_emis(
            data=data,
            file_date=fdt,
            state_filter=STATE_FILTER,
            species=SPECIES,
        )
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print(f"[{year}] No point emissions extracted.")
        return

    out_csv = os.path.join(out_dir, f"SE_{year}_bluesky_rx_emis.csv")
    pd.concat(dfs, ignore_index=True).to_csv(out_csv, index=False)
    print(f"[{year}] Saved: {out_csv}")

def main():
    for y in YEARS:
        process_year(y)

if __name__ == "__main__":
    main()