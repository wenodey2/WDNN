# Import Libraries
import pandas as pd
import numpy as np
from pywt import wavedec
import pywt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

tf.keras.backend.clear_session()  # For easy reset of notebook state.

turbine = 150

waveshapes = ['haar', 'db2', 'db3', 'db4', 'db5', 'coif3',
              'rbio1.1', 'rbio1.5', 'rbio2.4', 'rbio4/4']

# Import Data
data = pd.read_csv('dataset_2.csv')
data = data.fillna(0)
Power_data = data[f'Turbine{turbine}_Power']
Speed_data = data[f'Turbine{turbine}_Speed']
dSpeed_data = data[f'Turbine{turbine}_D1Speed']

N = len(data)  # data size, number of samples
NL = [3, 4, 5]  # number of layers for decomposition

mae_all = []
rmse_all = []

for b in range(len(NL)):

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
        for k in range(NL[b] + 1):
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


    mae_b = []
    rmse_b = []

    for w in range(len(waveshapes)):
        waveshape = waveshapes[w]
        mae = []
        rmse = []
        hoursahead = list(range(1, 7))  # number of hours ahead for prediction
        for hourahead in hoursahead:
            slotahead = hourahead  # number of slots ahead for prediction
            lag = 6  # training lag for model. use up to lag slots before as model input

            # Train-Test Split
            N_train = 2160  # train size
            N_test = 2000  # test size
            split = 2160
            train_index = np.random.choice(range(split), N_train, replace=False)
            test_index = range(split, split + N_test)

            # Reshape the training data and test data
            cA_x_train = np.zeros((N_train, lag * (NL[b] + 1) * 3))
            cA_y_train = np.zeros(N_train)
            for i in range(N_train):
                Pw = Power_data[train_index[i]:train_index[i] + lag]
                Sw = Speed_data[train_index[i]:train_index[i] + lag]
                Sdw = dSpeed_data[train_index[i]:train_index[i] + lag]
                Pc, Sc, Sdc = wavedcom(Pw, Sw, Sdw, NL[b], lag)
                for j in range(NL[b] + 1):
                    cA_x_train[i, j * lag:(j + 1) * lag] = Pc[Pc.columns[j]]
                    cA_x_train[i, lag * (NL[b] + 1) + j * lag:lag * (NL[b] + 1) + (j + 1) * lag] = Sc[
                        Sc.columns[j]]
                    cA_x_train[i,
                    2 * lag * (NL[b] + 1) + j * lag:2 * lag * (NL[b] + 1) + (j + 1) * lag] = Sdc[
                        Sdc.columns[j]]
                    cA_y_train[i] = Power_data[train_index[i] + lag + slotahead - 1]

            cA_x_test = np.zeros((N_test, lag * (NL[b] + 1) * 3))
            cA_y_test = np.zeros(N_test)
            for i in range(N_test):
                Pw = Power_data[test_index[i]:test_index[i] + lag]
                Sw = Speed_data[test_index[i]:test_index[i] + lag]
                Sdw = dSpeed_data[test_index[i]:test_index[i] + lag]
                Pc, Sc, Sdc = wavedcom(Pw, Sw, Sdw, NL[b], len(Pw))
                for j in range(NL[b] + 1):
                    cA_x_test[i, j * lag:(j + 1) * lag] = Pc[Pc.columns[j]]
                    cA_x_test[i, lag * (NL[b] + 1) + j * lag:lag * (NL[b] + 1) + (j + 1) * lag] = \
                    Sc[
                        Sc.columns[j]]
                    cA_x_test[i,
                    2 * lag * (NL[b] + 1) + j * lag:2 * lag * (NL[b] + 1) + (j + 1) * lag] = Sdc[
                        Sdc.columns[j]]
                    cA_y_test[i] = Power_data[test_index[i] + lag + slotahead - 1]


            # Deep Learning Model
            def build_model():
                model = keras.Sequential([
                    layers.Dense(lag, activation='relu', input_dim=lag * (NL[b] + 1) * 3),
                    layers.Dense(10, activation='relu'),
                    layers.Dense(10, activation='relu'),
                    layers.Dense(1, activation='sigmoid'),
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

            history = model.fit(cA_x_train, cA_y_train,
                                epochs=EPOCHS, validation_split=0.1, verbose=0,
                                callbacks=[PrintDot(), monitor])
            hist = pd.DataFrame(history.history)
            hist['epoch'] = history.epoch
            hist.tail()

            loss, mae, mse = model.evaluate(cA_x_train, cA_y_train, verbose=2)
            model.evaluate(cA_x_test, cA_y_test, verbose=2)

            Y_test = [Power_data[test_index[i] + lag + slotahead - 1] for i in range(N_test)]
            Y_pred = model.predict(cA_x_test)
            y_test = np.array(Y_test)
            y_pred = Y_pred.reshape((1, N_test)).flatten().tolist()

            # Metrics
            mae.append(np.mean(np.abs(y_test - y_pred)))
            rmse.append(np.sqrt(np.mean(np.power(y_test - y_pred, 2))))

        mae_b.append(mae)
        rmse_b.append(rmse)

    mae_all.append(mae_b)
    rmse_all.append(rmse_b)

d1 = {'h': [1, 2, 3, 4, 5, 6], 'haar': mae_all[0][0], 'db2': mae_all[0][1], 'db3': mae_all[0][2],
      'db4': mae_all[0][3], 'db5': mae_all[0][4], 'coif3': mae_all[0][5], 'rbio1.1': mae_all[0][6],
      'rbio1.5': mae_all[0][7], 'rbio2.4': mae_all[0][8], 'rbio4/4': mae_all[0][9]}

# Metrics
m1 = {'haar': mae_all[0][0], 'db2': mae_all[0][1], 'db3': mae_all[0][2], 'db4': mae_all[0][3],
      'db5': mae_all[0][4], 'coif3': mae_all[0][5], 'rbio1.1': mae_all[0][6],
      'rbio1.5': mae_all[0][7], 'rbio2.4': mae_all[0][8], 'rbio4/4': mae_all[0][9]}
m2 = {'haar': mae_all[1][0], 'db2': mae_all[1][1], 'db3': mae_all[1][2], 'db4': mae_all[1][3],
      'db5': mae_all[1][4], 'coif3': mae_all[1][5], 'rbio1.1': mae_all[1][6],
      'rbio1.5': mae_all[1][7], 'rbio2.4': mae_all[1][8], 'rbio4/4': mae_all[1][9]}
m3 = {'haar': mae_all[2][0], 'db2': mae_all[2][1], 'db3': mae_all[2][2], 'db4': mae_all[2][3],
      'db5': mae_all[2][4], 'coif3': mae_all[2][5], 'rbio1.1': mae_all[2][6],
      'rbio1.5': mae_all[2][7], 'rbio2.4': mae_all[2][8], 'rbio4/4': mae_all[2][9]}
r1 = {'haar': rmse_all[0][0], 'db2': rmse_all[0][1], 'db3': rmse_all[0][2], 'db4': rmse_all[0][3],
      'db5': rmse_all[0][4], 'coif3': rmse_all[0][5], 'rbio1.1': rmse_all[0][6],
      'rbio1.5': rmse_all[0][7], 'rbio2.4': rmse_all[0][8], 'rbio4/4': rmse_all[0][9]}
r2 = {'haar': rmse_all[1][0], 'db2': rmse_all[1][1], 'db3': rmse_all[1][2], 'db4': rmse_all[1][3],
      'db5': rmse_all[1][4], 'coif3': rmse_all[1][5], 'rbio1.1': rmse_all[1][6],
      'rbio1.5': rmse_all[1][7], 'rbio2.4': rmse_all[1][8], 'rbio4/4': rmse_all[1][9]}
r3 = {'haar': rmse_all[2][0], 'db2': rmse_all[2][1], 'db3': rmse_all[2][2], 'db4': rmse_all[2][3],
      'db5': rmse_all[2][4], 'coif3': rmse_all[2][5], 'rbio1.1': rmse_all[2][6],
      'rbio1.5': rmse_all[2][7], 'rbio2.4': rmse_all[2][8], 'rbio4/4': rmse_all[2][9]}

df_m1 = pd.DataFrame(m1)
df_m2 = pd.DataFrame(m2)
df_m3 = pd.DataFrame(m3)
df_r1 = pd.DataFrame(r1)
df_r2 = pd.DataFrame(r2)
df_r3 = pd.DataFrame(r3)

m1_mean = df_m1.mean().mean()
m2_mean = df_m2.mean().mean()
m3_mean = df_m3.mean().mean()

r1_mean = df_r1.mean().mean()
r2_mean = df_r2.mean().mean()
r3_mean = df_r3.mean().mean()

m_rows = pd.DataFrame({'mae_mean': [m1_mean, m2_mean, m3_mean]})
r_rows = pd.DataFrame({'rmse_mean': [r1_mean, r2_mean, r3_mean]})

m = pd.concat([df_m1, df_m2, df_m3])
r = pd.concat([df_r1, df_r2, df_r3])

m.to_csv(f'table1a.csv', index=False)
r.to_csv(f'wdnn_wavelets_mse1.csv', index=False)
m.mean().to_csv(f'table1b.csv', index=True)
r.mean().to_csv(f'wdnn_wavelets_mse2.csv', index=True)
m_rows.mean().to_csv(f'table1c.csv', index=True)
r_rows.mean().to_csv(f'wdnn_wavelets_mse3.csv', index=True)
