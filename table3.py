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
    metrics_df = pd.DataFrame(mets, columns=['MAE', 'RMSE'])
    return metrics_df


def metrics(dataset, variable, n, h=12):
    """
    Function to compute MAE and RMSE for FFNN, RNN, and LSTM
    :param dataset:  dataset number i.e. 1, 2, or 3
    :param variable: type of forecast string i.e. "speed" or "power"
    :param n: number of turbines
    :param h: maximum time horizon
    :return: lists of MAE and RMSE
    """
    directory_ffnn = f"dataset{dataset}/results_{variable}/ffnn"  # FFNN directory
    directory_rnn = f"dataset{dataset}/results_{variable}/rnn"    # RNN directory
    directory_lstm = f"dataset{dataset}/results_{variable}/lstm"  # LSTM directory

    metrics_ffnn = compute_metrics(directory_ffnn, n, h)
    metrics_rnn = compute_metrics(directory_rnn, n, h)
    metrics_lstm = compute_metrics(directory_lstm, n, h)

    mae = pd.concat([metrics_ffnn['MAE'], metrics_rnn['MAE'], metrics_lstm['MAE']], axis=1)
    rmse = pd.concat([metrics_ffnn['RMSE'], metrics_rnn['RMSE'], metrics_lstm['RMSE']], axis=1)
    mae_average = mae.mean()
    rmse_average = rmse.mean()

    mae_list = [mae_average[0], mae_average[1], mae_average[2]]
    rmse_list = [rmse_average[0], rmse_average[1], rmse_average[2]]
    return mae_list, rmse_list


metrics_1 = metrics(1, "speed", 36)    # dataset_1
metrics_2 = metrics(1, "speed", 160)   # dataset_2
metrics_3 = metrics(1, "speed", 100)   # dataset_3

mae_list = [metrics_1[0], metrics_2[0], metrics_3[0]]
rmse_list = [metrics_1[0], metrics_2[0], metrics_3[0]]

dataset_df = pd.DataFrame({"Data": ["Dataset-1", "Dataset-2", "Dataset-3"]})
mae = pd.DataFrame(data=mae_list, columns=["FFNN", "RNN", "LSTM"])
rmse = pd.DataFrame(data=rmse_list, columns=["FFNN", "RNN", "LSTM"])
mae_df = pd.concat(dataset_df, mae)
rmse_df = pd.concat(dataset_df, rmse)
metrics_df = pd.concat(mae_df, rmse)

metrics_df.to_csv('tables/table3.csv', index=True)
