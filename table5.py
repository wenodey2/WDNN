import glob
import numpy as np
import pandas as pd


# Time function
def computation_time(folder, variable):
    """
    Function to compute running time of WD code
    :param folder: Folder containing time results
    :param variable: 'speed' or 'power'
    :return:
    """
    if variable == 'speed':
        files_path = f"{folder}/speed_time_*.csv"
    if variable == 'power':
        files_path = f"{folder}/power_time_*.csv"

    file_list = glob.glob(files_path)
    times_list = []
    for file in file_list:
        times_list.append(pd.read_csv(file).to_numpy()[0, 0])
    mean_time = np.mean(np.array([times_list]))
    std_time = np.std(np.array([times_list]))
    return mean_time, std_time


speed_list = ['dataset1/results_ffnn/dataset1_Speed', 'dataset1/results_rnn/dataset1_Speed',
              'dataset1/results_lstm/dataset1_Speed', 'dataset2/results_ffnn/dataset2_Speed',
              'dataset2/results_rnn/dataset2_Speed', 'dataset2/results_lstm/dataset2_Speed',
              'dataset3/results_ffnn', 'dataset3/results_rnn', 'dataset3/results_lstm']
power_list = ['dataset1/results_ffnn/dataset1_Power', 'dataset1/results_rnn/dataset1_Power',
              'dataset1/results_lstm/dataset1_Power', 'dataset2/results_ffnn/dataset2_Power',
              'dataset2/results_rnn/dataset2_Power', 'dataset2/results_lstm/dataset2_Power']

speed_times = [computation_time(x, 'speed') for x in speed_list]
power_times = [computation_time(x, 'power') for x in power_list]

speed_rows = ['dataset1_Speed_FFNN', 'dataset1_Speed_RNN', 'dataset1_Speed_LSTM',
              'dataset2_Speed_FFNN', 'dataset2_Speed_RNN', 'dataset2_Speed_LSTM',
              'dataset3_Speed_FFNN', 'dataset3_Speed_RNN', 'dataset3_Speed_LSTM']
power_rows = ['dataset1_Power_FFNN', 'dataset1_Power_RNN', 'dataset1_Power_LSTM',
              'dataset2_Power_FFNN', 'dataset2_Power_RNN', 'dataset2_Power_LSTM']

speed_times_df = pd.DataFrame(speed_times, columns=['mean time', 'std time'],
                              index=pd.Index(speed_rows))
power_times_df = pd.DataFrame(power_times, columns=['mean time', 'std time'],
                              index=pd.Index(power_rows))

times_df = pd.concat([power_times_df, speed_times_df])
times_df.to_csv('tables/table5.csv', index=True)
