# Import Libraries
import time
import os
import sys
import pandas as pd
import numpy as np
from pywt import wavedec
import pywt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.backend.clear_session()  # For easy reset of notebook state.

tic = time.perf_counter()

turbine = 1    # run for all values from 1 to 160
hourahead = 1  # run for all values from 1 to 12

# Import Data
data = pd.read_csv('dataset_2.csv')
data = data.fillna(0)
Power_data = data[f'Turbine{turbine}_Power']
Speed_data = data[f'Turbine{turbine}_Speed']
dSpeed_data = data[f'Turbine{turbine}_D1Speed']

N = len(data)  # data size, number of samples
NL = 4  # number of layers for decomposition

waveshape = 'db4'


# Wavelet depcomposition function
def wavedcom(Pwave, Swave, dSwave, dlevel, length):
    # Layer Naming Function
    def index_str(level):
        l = ['cA' + str(level)]
        for i in range(level):
            l.append('cD' + str(level - i))
        return l

    # wavelet decomposition
    Power_coeffs = wavedec(Pwave, waveshape, level=dlevel)
    Speed_coeffs = wavedec(Swave, waveshape, level=dlevel)
    dSpeed_coeffs = wavedec(dSwave, waveshape, level=dlevel)
    Power_coeffs_m = pd.DataFrame(columns=index_str(dlevel), index=range(length))
    Speed_coeffs_m = pd.DataFrame(columns=index_str(dlevel), index=range(length))
    dSpeed_coeffs_m = pd.DataFrame(columns=index_str(dlevel), index=range(length))

    # wavelet redecomposition
    for k in range(NL + 1):
        if k == 0:
            Power_coeffs_m[Power_coeffs_m.columns[0]] = pywt.upcoef(
                'a', Power_coeffs[k], waveshape, level=dlevel, take=length)
            Speed_coeffs_m[Speed_coeffs_m.columns[0]] = pywt.upcoef(
                'a', Speed_coeffs[k], waveshape, level=dlevel, take=length)
            dSpeed_coeffs_m[dSpeed_coeffs_m.columns[0]] = pywt.upcoef(
                'a', dSpeed_coeffs[k], waveshape, level=dlevel, take=length)
        else:
            Power_coeffs_m[Power_coeffs_m.columns[k]] = pywt.upcoef(
                'd', Power_coeffs[k], waveshape, level=dlevel + 1 - k, take=length)
            Speed_coeffs_m[Speed_coeffs_m.columns[k]] = pywt.upcoef(
                'd', Speed_coeffs[k], waveshape, level=dlevel + 1 - k, take=length)
            dSpeed_coeffs_m[dSpeed_coeffs_m.columns[k]] = pywt.upcoef(
                'd', dSpeed_coeffs[k], waveshape, level=dlevel + 1 - k, take=length)

    return Power_coeffs_m, Speed_coeffs_m, dSpeed_coeffs_m


MAE = []
RMSE = []

slotahead = hourahead  # number of slots ahead for prediction
lag = 6  # training lag for model. use up to lag slots before as model input

# Train-Test Split
N_train = 15592  # train size
N_test = 12 + lag + 7228  # test size
split = 15592
train_index = np.random.choice(range(split), N_train, replace=False)
test_index = range(split, split + N_test)

# Reshape the training data and test data
cA_x_train = np.zeros((N_train, lag * (NL + 1) * 3))
cA_y_train = np.zeros(N_train)
for i in range(N_train):
    Pw = Power_data[train_index[i]:train_index[i] + lag]
    Sw = Speed_data[train_index[i]:train_index[i] + lag]
    Sdw = dSpeed_data[train_index[i]:train_index[i] + lag]
    Pc, Sc, Sdc = wavedcom(Pw, Sw, Sdw, NL, lag)
    for j in range(NL + 1):
        cA_x_train[i, j * lag:(j + 1) * lag] = Pc[Pc.columns[j]]
        cA_x_train[i, lag * (NL + 1) + j * lag:lag * (NL + 1) + (j + 1) * lag] = Sc[
            Sc.columns[j]]
        cA_x_train[i, 2 * lag * (NL + 1) + j * lag:2 * lag * (NL + 1) + (j + 1) * lag] = Sdc[
            Sdc.columns[j]]
        cA_y_train[i] = Power_data[train_index[i] + lag + slotahead - 1]

cA_x_test = np.zeros((N_test, lag * (NL + 1) * 2))
cA_y_test = np.zeros(N_test)
for i in range(N_test):
    Pw = Power_data[test_index[i]:test_index[i] + lag]
    Sw = Speed_data[test_index[i]:test_index[i] + lag]
    Sdw = dSpeed_data[train_index[i]:train_index[i] + lag]
    Pc, Sc, Sdc = wavedcom(Pw, Sw, Sdw, NL, len(Pw))
    for j in range(NL + 1):
        cA_x_test[i, j * lag:(j + 1) * lag] = Pc[Pc.columns[j]]
        cA_x_test[i, lag * (NL + 1) + j * lag:lag * (NL + 1) + (j + 1) * lag] = Sc[
            Sc.columns[j]]
        cA_x_test[i, 2 * lag * (NL + 1) + j * lag:2 * lag * (NL + 1) + (j + 1) * lag] = Sdc[
            Sdc.columns[j]]
        cA_y_test[i] = Power_data[test_index[i] + lag + slotahead - 1]


# Deep Learning Model
def build_model():
    model = keras.Sequential([
        layers.LSTM(lag, return_sequences=True, input_shape=[None, 1]),
        layers.LSTM(10, return_sequences=True),
        layers.LSTM(10, return_sequences=True),
        layers.LSTM(1),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=100,
                        verbose=0, mode='auto', restore_best_weights=True)


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(cA_x_train, cA_y_train, epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot(), monitor])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

loss, mae, mse = model.evaluate(cA_x_train, cA_y_train, verbose=2)
model.evaluate(cA_x_test, cA_y_test, verbose=2)

model_save_path = f"dataset2/results_power/lstm/models/"
isExist = os.path.exists(model_save_path)
if not isExist:
    os.makedirs(model_save_path)
model.save(f'dataset2/results_power/lstm/models/model_{turbine}_{hourahead}')

Y_test = [Speed_data[test_index[i] + lag + slotahead - 1] for i in range(N_test)]
Y_pred = model.predict(cA_x_test)
y_test = np.array(Y_test)
y_pred = Y_pred.reshape((1, N_test)).flatten().tolist()

# Metrics
MAE.append(np.mean(np.abs(y_test - y_pred)))
RMSE.append(np.sqrt(np.mean(np.power(y_test - y_pred, 2))))

d = {'MAE': MAE, 'RMSE': RMSE}
df1 = pd.DataFrame(d)
df1.to_csv(f'dataset2/results_power/lstm/metrics{turbine}_{hourahead}.csv', index=False)

toc = time.perf_counter()
time_elapsed = (toc - tic)/60
df3 = pd.DataFrame(np.array([time_elapsed]))
df3.to_csv(f'dataset2/results_power/lstm/speed_time_{turbine}_{hourahead}.csv', index=False)

print(f"The computation took {time_elapsed:0.4f} minutes")
