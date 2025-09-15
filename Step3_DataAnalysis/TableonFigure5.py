def compute_and_store_metrics(pred, obs, year, state):
    pred, obs = remove_nan_values(pred, obs)
    N = len(pred)

    row_mask = (
        (metrics_df["Year"].astype(str) == str(year)) &  # Ensure string match for 'AllYears'
        (metrics_df["State"] == state)
    )

    if N == 0:
        metrics_df.loc[row_mask, "# Pairs"] = 0
        return

    for key, entry in metrics_dict.items():
        try:
            val = entry["func"](pred, obs)
            metrics_df.loc[row_mask, key] = val
        except Exception:
            metrics_df.loc[row_mask, key] = np.nan
import itertools

years = ["AllYears"]
states = ['Florida', 'Georgia', 'South Carolina', 'Overall']

rows = list(itertools.product(years, states))
metrics_df = pd.DataFrame(rows, columns=["Year", "State"])

# Initialize metric columns
for metric in metrics_dict:
    metrics_df[metric] = np.nan

# Predefined
states = ['Florida', 'Georgia', 'South Carolina']
years = [2017, 2018, 2019]

# Load all years of data
all_data = []
for year in years:
    file_path = os.path.join(dir_python_local, 'collocated_mod_obs', f'aq_SE_{year}', f'AQS_Daily_aq_SE_{year}_with_smoke_day.csv')
    df = pd.read_csv(file_path)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

compute_and_store_metrics(
        df_all['PM_TOT_mod'].values,
        df_all['PM_TOT_ob'].values,
        'AllYears', 'Overall'
    )

for state in states:
    df_state = df_all[df_all['State'] == state]
    print(f"  → State: {state}")
    compute_and_store_metrics(
        df_state['PM_TOT_mod'].values,
        df_state['PM_TOT_ob'].values,
        'AllYears', state
    )