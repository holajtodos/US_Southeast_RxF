import scipy.stats
from sklearn.metrics import mean_squared_error

# reference: https://www.tandfonline.com/doi/full/10.1080/10962247.2016.1265027
def remove_nan_values(prediction, observation):
    valid_idx = (~np.isnan(prediction)) & (~np.isnan(observation))
    valid_p = prediction[valid_idx]
    valid_o = observation[valid_idx]
    return valid_p, valid_o

def Npairs(prediction, observation):
    return len(prediction)

def MB(prediction, observation):
    return np.sum(prediction - observation) / len(prediction)


def ME(prediction, observation):
    return np.sum(np.abs(prediction - observation)) / len(prediction)


def RMSE(prediction, observation):
    return np.sqrt(np.sum((prediction - observation) ** 2) / len(prediction))


def CRMSE(prediction, observation):
    p_mean, o_mean = np.mean(prediction), np.mean(observation)
    return np.sqrt((1 / len(prediction) * np.sum(((prediction - p_mean) - (observation - o_mean)) ** 2)))


def NMB(prediction, observation):
    if np.sum(observation) == 0:
        return np.nan
    return (np.sum(prediction - observation) / np.sum(observation)) * 100


def NME(prediction, observation):
    if np.sum(observation) == 0:
        return np.nan
    return ((np.sum(np.abs(prediction - observation))) / np.sum(observation)) * 100


def MNB(prediction, observation):
    cur_predict, cur_obs = prediction[observation > 0], observation[observation > 0]
    return (1 / len(cur_predict)) * np.sum((cur_predict - cur_obs) / cur_obs) * 100


def MNE(prediction, observation):
    cur_predict, cur_obs = prediction[observation > 0], observation[observation > 0]
    return (1 / len(cur_predict)) * np.sum(np.abs(cur_predict - cur_obs) / cur_obs) * 100


def FB(prediction, observation):
    return (2 / len(prediction)) * np.sum((prediction - observation) / (prediction + observation)) * 100


def FE(prediction, observation):
    return (2 / len(prediction)) * np.sum(np.abs(prediction - observation) / (prediction + observation)) * 100


def IOA(prediction, observation):
    numerator = np.sum((prediction - observation) ** 2)
    prediction_shift = np.abs((prediction - np.mean(observation)))
    observation_shift = np.abs((observation - np.mean(observation)))
    denominator = np.sum((prediction_shift + observation_shift) ** 2)
    return 1 - numerator / denominator


def spearman_r(prediction, observation):
    r = scipy.stats.spearmanr(prediction, observation)
    return r[0]

def spearman_p(prediction, observation):
    r = scipy.stats.spearmanr(prediction, observation)
    return r[1]

# define metrics dict
metrics_dict = {
    "# Pairs": {"func": Npairs, "vals": []},
    "MB": {"func": MB, "vals": []},
    "ME": {"func": ME, "vals": []},
    "RMSE": {"func": RMSE, "vals": []},
    "CRMSE": {"func": CRMSE, "vals": []},
    "NMB": {"func": NMB, "vals": []},
    "NME": {"func": NME, "vals": []},
    "MNB": {"func": MNB, "vals": []},
    "MNE": {"func": MNE, "vals": []},
    "FB": {"func": FB, "vals": []},
    "FE": {"func": FE, "vals": []},
    "IOA": {"func": IOA, "vals": []},
    "Spearman R": {"func": spearman_r, "vals": []},
    "Spearman p": {"func": spearman_p, "vals": []}
}

# def compute_and_store_metrics(pred, obs, year, fire_category, season, state):
def compute_and_store_metrics(pred, obs, year, season, state):

    pred, obs = remove_nan_values(pred, obs)
    N = len(pred)

    row_mask = (
        (metrics_df["Year"] == year) &
#         (metrics_df["Fire Category"] == fire_category) &
        (metrics_df["Season"].str.lower() == season.lower()) &
        (metrics_df["State"] == state)
    )

    if N == 0:
        metrics_df.loc[row_mask, "# Pairs"] = 0
        return

    # Loop through all defined metrics
    for key, entry in metrics_dict.items():
        try:
            val = entry["func"](pred, obs)
            metrics_df.loc[row_mask, key] = val
        except Exception:
            metrics_df.loc[row_mask, key] = np.nan

# Define a list of all combinations to initialize the DataFrame
years = [2017, 2018, 2019]
# fire_categories = ["Near Fire", "No Fire"]
states = ["Florida", "Georgia", "South Carolina", "Overall"]
seasons = ["high-burn", "low-burn"]

# Define all metric names
metric_names = ["# Pairs", "MB", "ME", "RMSE", "CRMSE", "NMB", "NME", "MNB", "MNE", "FB", "FE", "IOA", "Spearman R"]

# Create a list of all combinations
rows = []
for year in years:
#     for fire_cat in fire_categories:
        for state in states:
            for season in seasons:
#                 row = {"Year": year, "Fire Category": fire_cat, "State": state, "Season": season}
                row = {"Year": year, "State": state, "Season": season}
                for metric in metric_names:
                    row[metric] = np.nan
                rows.append(row)

# Create the empty DataFrame with structure
metrics_df = pd.DataFrame(rows)

# List of project folders to loop over
years = [2017, 2018, 2019]
for year in years:
    file_path = os.path.join(dir_python_local, 'collocated_mod_obs', f'aq_SE_{year}', f'AQS_Daily_aq_SE_{year}_with_smoke_day.csv')
    df_result = pd.read_csv(file_path)

    # Your existing plotting code starts here
#     smoke_flags = [1, 0]
#     for i, flag in enumerate(smoke_flags):
#         subset = df_result[df_result['smoke_day'] == flag].copy()
    subset = df_result
    subset['PM_TOT_mod'] = subset['PM_TOT_mod'].clip(lower=0.000001)
    subset['PM_TOT_ob'] = subset['PM_TOT_ob'].clip(lower=0.000001)

    subset['log_PM_TOT_mod'] = np.log10(subset['PM_TOT_mod'])
    subset['log_PM_TOT_ob'] = np.log10(subset['PM_TOT_ob'])

    # Evaluate using original (non-log) values
    # Add seasonal column to subset if not already there
    if 'season' not in subset.columns:
        subset['date'] = pd.to_datetime(dict(year=subset['SYYYY'], month=subset['SMM'], day=subset['SDD']))
        subset['season'] = subset['date'].dt.month.map(lambda m: 'High-burn' if m <= 4 else 'Low-burn')

    seasons = ['High-burn', 'Low-burn']
    states = ['Florida', 'Georgia', 'South Carolina']

#     fire_category = "Near Fire" if flag == 1 else "No Fire"
#     print(f"\nEvaluation Metrics for Year {year} - {fire_category}")
    print(f"\nEvaluation Metrics for Year {year}")

    for season in seasons:
        subset_season = subset[subset['season'].str.lower() == season.lower()]

        # Overall for the season
        compute_and_store_metrics(
            subset_season['PM_TOT_mod'].values,
            subset_season['PM_TOT_ob'].values,
            year, season, "Overall"
        )

        for state in states:
            subset_state = subset_season[subset_season['State'] == state]
            compute_and_store_metrics(
                subset_state['PM_TOT_mod'].values,
                subset_state['PM_TOT_ob'].values,
                year, season, state
            )


def compute_and_store_metrics(pred, obs, year, season, state):
    pred, obs = remove_nan_values(pred, obs)
    N = len(pred)

    row_mask = (
        (metrics_df["Year"].astype(str) == str(year)) &  # Ensure string match for 'AllYears'
        (metrics_df["Season"].str.lower() == season.lower()) &
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
seasons = ['High-burn', 'Low-burn']
states = ['Florida', 'Georgia', 'South Carolina', 'Overall']

rows = list(itertools.product(years, seasons, states))
metrics_df = pd.DataFrame(rows, columns=["Year", "Season", "State"])

# Initialize metric columns
for metric in metrics_dict:
    metrics_df[metric] = np.nan

# Predefined
states = ['Florida', 'Georgia', 'South Carolina']
seasons = ['High-burn', 'Low-burn']
years = [2017, 2018, 2019]

# Load all years of data
all_data = []
for year in years:
    file_path = os.path.join(dir_python_local, 'collocated_mod_obs', f'aq_SE_{year}', f'AQS_Daily_aq_SE_{year}_with_smoke_day.csv')
    df = pd.read_csv(file_path)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# Create season column
if 'season' not in df_all.columns:
    df_all['date'] = pd.to_datetime(dict(year=df_all['SYYYY'], month=df_all['SMM'], day=df_all['SDD']))
    df_all['season'] = df_all['date'].dt.month.map(lambda m: 'High-burn' if m <= 4 else 'Low-burn')

# --- Compute metrics for all states and seasons combined ---
for season in seasons:
    df_season = df_all[df_all['season'].str.lower() == season.lower()]
    print(f"\nEvaluation Metrics for Season: {season} (Overall)")
    compute_and_store_metrics(
        df_season['PM_TOT_mod'].values,
        df_season['PM_TOT_ob'].values,
        'AllYears', season, 'Overall'
    )

    for state in states:
        df_state = df_season[df_season['State'] == state]
        print(f"  → State: {state}")
        compute_and_store_metrics(
            df_state['PM_TOT_mod'].values,
            df_state['PM_TOT_ob'].values,
            'AllYears', season, state
        )