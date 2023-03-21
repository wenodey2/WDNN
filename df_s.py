# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
import time


# Deep Forecast Function
def df_function(dataset, num_turbines, start_train, end_train, end_test, wind_cols):
    """
    Function Implenting the Deep-Forecast Algorithm
    :param dataset: 1, 2, or 3
    :param num_turbines: Number of turbines in the dataset
    :param start_train: Index for first training observation
    :param end_train: Index for last training observation
    :param end_test: Index for last testing observation
    :param wind_cols: List of wind speed columns for all turbines
    :return: Write metrics, prediction, and running time data to CSV files.
    """
    np.random.seed(1234)
    start_time = time.perf_counter()
    start_test = end_train + 1
    for h in range(1, 13):

        inputHorizon = 12  # number of time steps as input
        inOutVecDim = num_turbines  # number of stations
        lstmModels = [None for _ in range(6)]

        xTest, yTest = None, None

        file_dataset = f'dataset_{dataset}.csv'
        data_file = pd.read_csv(file_dataset)
        data_speed = data_file.iloc[:, wind_cols]
        winds = data_speed.to_numpy()

        biglist = []
        endTrain = []
        startTest = []
        for i in range(12):
            bb = list(range(start_train - 1, end_train, i + 1))
            cc = list(range(start_test - 1, end_test))
            endTrain.append(len(bb))
            startTest.append(len(bb) + 1)
            dd = [bb, cc]
            dd = [item for sublist in dd for item in sublist]
            biglist.append(dd)

        # Commented out IPython magic to ensure Python compatibility.
        winds = winds[biglist[h - 1], :inOutVecDim]
        means_stds = [0, 0]

        validation_split = 0.05
        batchSize = 3
        activation = ['sigmoid', "tanh", "relu", 'linear']
        activation = activation[2]
        realRun = 1
        # model number : 1   2   3   4   5   6
        epochs, trainDataRate = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[1, 1, 1, 1, 1, 1],
                                                                               0.005]

        def normalize_winds_0_1(winds):
            """normalize based on each station data"""
            # stations = winds.shape[1]
            normal_winds = []
            mins_maxs = []
            windMax = winds.max()
            windMin = winds.min()
            normal_winds = (winds - windMin) / windMax
            mins_maxs = [windMin, windMax]
            return np.array(normal_winds), mins_maxs

        winds, means_stds = normalize_winds_0_1(winds)

        def denormalize(vec):
            res = vec * means_stds[1] + means_stds[0]  # from 0 to 1
            return res

        def loadData_1():
            # for lstm1 output xtrain ytrain
            result = []
            for index in range(len(winds) - inputHorizon):
                result.append(winds[index:index + inputHorizon])
            result = np.array(result)

            trainRow = int(endTrain[h - 1] * trainDataRate)
            X_train = result[:trainRow, :]
            y_train = winds[inputHorizon:trainRow + inputHorizon]
            xTest = result[endTrain[h - 1]:end_test - 13, :]
            yTest = winds[endTrain[h - 1] + inputHorizon:end_test - 13 + inputHorizon]
            predicted = np.zeros_like(yTest)
            return [X_train, y_train, xTest, yTest, predicted]

        def loadData(preXTrain, preYTrain, model):
            # xtrain and ytrain from loadData_1 for lstm2 output: xtrain ytrain
            xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)

            for ind in range(len(preXTrain) - inputHorizon - 1):
                tempInput = preXTrain[ind]
                temp_shape = tempInput.shape
                tempInput = np.reshape(tempInput, (1, temp_shape[0], temp_shape[1]))
                output = model.predict(tempInput)
                tInput = np.reshape(tempInput, temp_shape)
                tempInput = np.vstack((tInput, output))
                tempInput = np.delete(tempInput, 0, axis=0)
                xTrain[ind] = tempInput
                yTrain[ind] = preYTrain[ind + 1]
            return [xTrain, yTrain]

        def buildModelLSTM_1():
            model = Sequential()
            in_nodes = out_nodes = inOutVecDim
            layers = [in_nodes, num_turbines * 2, num_turbines, 32, out_nodes]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM_2():
            model = Sequential()
            layers = [inOutVecDim, 10, num_turbines * 2, 32, inOutVecDim]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM_3():
            model = Sequential()
            layers = [inOutVecDim, num_turbines, num_turbines * 2, 32, inOutVecDim]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM_4():
            model = Sequential()
            layers = [inOutVecDim, num_turbines, num_turbines * 2, num_turbines, inOutVecDim]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=True))
            model.add(LSTM(layers[2], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM_5():
            model = Sequential()
            layers = [inOutVecDim, 30, num_turbines * 2, num_turbines, inOutVecDim]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM_6():
            model = Sequential()
            layers = [inOutVecDim, num_turbines * 2, num_turbines * 2, num_turbines, inOutVecDim]
            model.add(LSTM(input_dim=layers[0], units=layers[1], return_sequences=True))
            model.add(LSTM(layers[2], return_sequences=False))
            model.add(Dense(units=layers[4]))
            model.add(Activation(activation))
            optimizer = keras.optimizers.RMSprop(lr=0.001)
            model.compile(loss="mae", optimizer=optimizer, metrics=['mse'])
            return model

        def buildModelLSTM(lstmModelNum):
            if lstmModelNum == 1:
                return buildModelLSTM_1()
            elif lstmModelNum == 2:
                return buildModelLSTM_2()
            elif lstmModelNum == 3:
                return buildModelLSTM_3()
            elif lstmModelNum == 4:
                return buildModelLSTM_4()
            elif lstmModelNum == 5:
                return buildModelLSTM_5()
            elif lstmModelNum == 6:
                return buildModelLSTM_6()

        def trainLSTM(xTrain, yTrain, lstmModelNum):
            # train first LSTM with inputHorizon number of real input values
            lstmModel = buildModelLSTM(lstmModelNum)
            lstmModel.fit(xTrain, yTrain, batch_size=batchSize,
                          epochs=epochs[lstmModelNum - 1], validation_split=validation_split)
            return lstmModel

        def test():
            ''' calculate the predicted values(predicted) '''
            for ind in range(len(xTest)):
                modelInd = ind % 6
                if modelInd == 0:
                    testInputRaw = xTest[ind]
                    testInputShape = testInputRaw.shape
                    testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
                else:
                    testInputRaw = np.vstack((testInputRaw, predicted[ind - 1]))
                    testInput = np.delete(testInputRaw, 0, axis=0)
                    testInputShape = testInput.shape
                    testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
                predicted[ind] = lstmModels[modelInd].predict(testInput)

            return

        def errorMeasures(denormalYTest, denormalYPredicted):
            mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
            rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
            nrsme_maxMin = 100 * rmse / (denormalYTest.max() - denormalYTest.min())
            nrsme_mean = 100 * rmse / (denormalYTest.mean())
            return mae, rmse, nrsme_maxMin, nrsme_mean

        def drawGraphStation(station, visualise=1, ax=None):
            """draw graph of predicted vs real values"""
            YTest = yTest[:, station]
            denormalYTest = denormalize(YTest)
            denormalPredicted = denormalize(predicted[:, station])
            mae, rmse, nrmse_maxMin, nrmse_mean = errorMeasures(denormalYTest, denormalPredicted)
            print(f'station {station + 1}: MAE = {mae}  RMSE = {rmse}  \
            nrmse_maxMin = {nrmse_maxMin}  nrmse_mean = {nrmse_mean}')
            if visualise:
                if ax is None:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                ax.plot(denormalYTest, label='Real')
                ax.plot(denormalPredicted, label='Predicted', color='red')
                ax.set_xticklabels([0, 100, num_turbines, 300], rotation=40)
            return mae, rmse, nrmse_maxMin, nrmse_mean

        def drawGraphAllStations():
            rows, cols = 4, 4
            maeRmse = np.zeros((rows * cols, 4))
            fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True)
            staInd = 0
            for ax in np.ravel(ax_array):
                maeRmse[staInd] = drawGraphStation(staInd, visualise=1, ax=ax)
                staInd += 1
            plt.xticks([0, 100, num_turbines, 300])
            print(maeRmse.mean(axis=0))
            filename = 'finalEpoch'
            plt.savefig('{}.pgf'.format(filename))
            plt.savefig('{}.pdf'.format(filename))
            plt.show()
            return

        # Training
        xTrain, yTrain, xTest, yTest, predicted = loadData_1()
        print(' Training LSTM 1 ...')
        lstmModels[0] = trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1, 6):
            xTrain, yTrain = loadData(xTrain, yTrain, lstmModels[modelInd - 1])
            print(' Training LSTM %s ...' % (modelInd + 1))
            lstmModels[modelInd] = trainLSTM(xTrain, yTrain, modelInd + 1)

        # Testing
        print('...... TESTING  ...')
        test()

        # Metrics
        mae_list = np.zeros((num_turbines,))
        rmse_list = np.zeros((num_turbines,))
        for station in range(num_turbines):
            YTest1 = yTest[:, station]
            denormalYTest1 = denormalize(YTest1)
            denormalPredicted1 = denormalize(predicted[:, station])
            mae1, rmse1, nrmse_maxMin1, nrmse_mean1 = errorMeasures(denormalYTest1,
                                                                    denormalPredicted1)
            mae_list[station] = mae1
            rmse_list[station] = rmse1
            print(f'station {station + 1}: MAE = {mae1}  RMSE = {rmse1}  \
            nrmse_maxMin = {nrmse_maxMin1}  nrmse_mean = {nrmse_mean1}')

        # Results
        mae_test = np.mean(mae_list)
        rmse_test = np.mean(rmse_list)
        print(f'MAE:  {mae_test:.5f}')
        print(f'RMSE: {rmse_test:.5f}')

        d = {'MAE': [mae_test], 'RMSE': [rmse_test]}
        df = pd.DataFrame.from_dict(d)
        df.to_csv(f'dataset{dataset}/results_df/metrics_{h}.csv', index=False)

        df2 = pd.DataFrame(data=predicted)
        df2.to_csv(f"dataset{dataset}/results_df/df_speed_predicted_{h}.csv", index=False)

        end_time = time.perf_counter()
        time_elapsed = (start_time - end_time) / 60
        df3 = pd.DataFrame(np.array([time_elapsed]))
        df3.to_csv(f'dataset{dataset}/results_df/speed_time_{h}.csv', index=False)

        print(f"Computation time: {time_elapsed} minutes")
        return 0


dat1 = df_function(dataset=1, num_turbines=36, start_train=1, end_train=18293,
                   end_test=26471, wind_cols=range(1, 250, 7))
dat2 = df_function(dataset=2, num_turbines=160, start_train=1, end_train=15592,
                   end_test=22838, wind_cols=range(1, 960, 6))
dat3 = df_function(dataset=3, num_turbines=100, start_train=1, end_train=43824,
                   end_test=52590, wind_cols=range(1, 400, 4))
