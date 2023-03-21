import numpy as np
import pandas as pd


def compute_metrics(directory, n_turbines, max_h):
    """
    Function to compute the metrics for a farm and return MAE and RMSE
    for all time horizons
    :param directory:  Directory containing metrics files
    :param n_turbines: Total number of turbines
    :param max_h:      Maximum time horizon
    :return:           Pandas dataframe of MAE and RMSE for all time horizons
    """
    mets = np.zeros((max_h, 2))
    for h in range(max_h):
        met = np.zeros((n_turbines, 2))
        for t in range(n_turbines):
            m_file = f'{directory}/metrics{t + 1}_{h + 1}.csv'
            met[t, :] = pd.read_csv(m_file).to_numpy()
        mets[h, :] = np.mean(met, axis=0)
    metrics = pd.DataFrame(mets, columns=['MAE', 'RMSE'])
    return metrics


def compute_metrics_df(path):
    """
    Function to compute the metrics for a farm from Deep Forecast results
    for all time horizons
    :param path: Directory containing metrics files
    :return: Pandas dataframe of MAE and RMSE for all time horizons
    """
    data_dump = []
    for i in range(12):
        dat = np.array(pd.read_csv(f"{path}/metrics_{i + 1}.csv"))
        data_dump.append(dat)
    met = np.reshape(data_dump, newshape=[-1, 2])
    metrics = pd.DataFrame(met, columns=['MAE', 'RMSE'])
    return metrics


# DF Speed
df_s_1_path = f"dataset1/results_df"
df_s_2_path = f"dataset2/results_df"
df_s_3_path = f"dataset3/results_df"
metrics_df_s_1 = compute_metrics_df(df_s_1_path)
metrics_df_s_2 = compute_metrics_df(df_s_2_path)
metrics_df_s_3 = compute_metrics_df(df_s_3_path)


# WDNN Speed
wdnn_s_1_path = "dataset1/results_speed/ffnn"
wdnn_s_2_path = "dataset2/results_speed/ffnn"
wdnn_s_3_path = "dataset3/results_speed/ffnn"
metrics_wdnn_s_1 = compute_metrics(wdnn_s_1_path, 36, 12)
metrics_wdnn_s_2 = compute_metrics(wdnn_s_2_path, 160, 12)
metrics_wdnn_s_3 = compute_metrics(wdnn_s_3_path, 100, 12)


# WDNN Power
wdnn_p_1_path = "dataset1/results_speed/ffnn"
wdnn_p_2_path = "dataset2/results_speed/ffnn"
metrics_wdnn_p_1 = compute_metrics(wdnn_p_1_path, 36, 12)
metrics_wdnn_p_2 = compute_metrics(wdnn_p_2_path, 160, 12)
metrics_wdnn_p = pd.concat([metrics_wdnn_p_1, metrics_wdnn_p_2], axis=1)
metrics_wdnn_p_1.to_csv('dataset1/results_power/ffnn/metrics_wdnn_p.csv')


# STAN Speed
stan_s_1_path = "dataset1/results_stan"
stan_s_2_path = "dataset2/results_stan"
stan_s_3_path = "dataset3/results_stan"
metrics_stan_s_1 = compute_metrics(stan_s_1_path, 36, 12)
metrics_stan_s_2 = compute_metrics(stan_s_2_path, 160, 12)
metrics_stan_s_3 = compute_metrics(stan_s_3_path, 100, 12)


# PSTN Speed
metrics_pstn_s_3 = pd.read_csv("dataset3/results_pstn/metrics_pstn_s_3.csv")


# Persistence Speed
metrics_per = pd.read_csv("metrics_persistence.csv.csv")

# Table 8 MAE
hrs = pd.DataFrame({"h (hours)": [i for i in range(1, 13)]})
tab8m = pd.concat([hrs, metrics_per["MAE_s_3"]], axis=1)
tab8m = pd.concat([tab8m, metrics_df_s_3["MAE"]], axis=1)
tab8m = pd.concat([tab8m, metrics_stan_s_3["MAE"]], axis=1)
tab8m = pd.concat([tab8m, metrics_wdnn_s_3["MAE"]], axis=1)
tab8m = pd.concat([tab8m, metrics_pstn_s_3["MAE"]], axis=1)

# Table 8 RMSE
tab8r = pd.concat([hrs, metrics_per["RMSE_s_3"]], axis=1)
tab8r = pd.concat([tab8r, metrics_df_s_3["RMSE"]], axis=1)
tab8r = pd.concat([tab8r, metrics_stan_s_3["RMSE"]], axis=1)
tab8r = pd.concat([tab8r, metrics_wdnn_s_3["RMSE"]], axis=1)
tab8r = pd.concat([tab8r, metrics_pstn_s_3["RMSE"]], axis=1)

tab8m.to_csv('tables/table8.csv')
