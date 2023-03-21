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


h = 12
n = 36

directory_ffnn = "results_speed/ffnn"   # FFNN directory
directory_rnn = "results_speed/rnn"     # RNN directory
directory_lstm = "results_speed/lstm"   # LSTM directory

metrics_ffnn = compute_metrics(directory_ffnn, n, h)
metrics_rnn = compute_metrics(directory_rnn, n, h)
metrics_lstm = compute_metrics(directory_lstm, n, h)

time_horizons = pd.DataFrame({'h (hours)': [x for x in range(1, 13)]})
mae = pd.concat([time_horizons, metrics_ffnn['MAE'], metrics_rnn['MAE'],
                 metrics_lstm['MAE']], axis=1)
rmse = pd.concat([time_horizons, metrics_ffnn['RMSE'], metrics_rnn['RMSE'],
                  metrics_lstm['RMSE']], axis=1)

mae.to_csv('tables/table2.csv', index=True)
