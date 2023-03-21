# Libraries
library(DSWE)


### DATASET 1
# Load Data
data = read.csv('dataset_1.csv', header = T)
for (t in 1:36) {
  pred_speed = read.csv(paste0(
    'dataset1/results_speed/ffnn/windspeed_', t, '_1', '.csv'))[13:8212, 1:12]
  for (h in 2:12) {
    pred_speed_x = read.csv(paste0(
      'dataset1/results_speed/ffnn/windspeed_', t, '_', 1, '.csv'))[13:8212, 1:12]
    pred_speed = cbind(pred_speed, pred_speed_x)
  }
  # Prep Data
  train_index =  1:18293
  test_index = 18294:26471
  time_index = 1:26471
  actual_data = data[, c((7*(t-1))+2, (7*(t-1))+4)]
  actual_data[is.na(actual_data[,2]), 2] = rnorm(
    length(which(is.na(actual_data[,2]) == T)),0,1e-8)
  train_data = actual_data[train_index, ]
  metrics_f = matrix(NA, nrow = 12, ncol = 12)
  metrics_y = matrix(NA, nrow = 12, ncol = 12)
  for (h in 1:12) {
    test_data2 = actual_data[test_index - h, ]
    # TempGP Using Persistence data - W
    load(paste0("dataset1/results_df/tempGPObject_", t, ".RData"))
    pred_f9 = predict(tempGPObject, test_data2[, 1])
    rmse_f9 = sqrt(mean((test_data2[, 2] - pred_f9) ^ 2, na.rm = T))
    mae_f9 = mean(abs(test_data2[, 2] - pred_f9), na.rm = T)
    load(paste0("dataset1/results_df/tempGPObject_", t, ".RData"))
    pred_y9 = rep(0, length(pred_f9))
    sixes = length(pred_f9) / 6
    actual_testX = actual_data[test_index, 1, drop = F]  # Actual wind data X
    actual_testY = actual_data[test_index, 2]  # Actual wind data Y
    for (i in 1:sixes) {
      inds = (6 * (i - 1) + 1):(6 * (i - 1) + 6)
      testX_i = as.matrix(test_data2[inds, 1, drop = F])
      testT_i = test_index[inds]
      pred_y9[inds] = predict(tempGPObject, testX_i, testT_i)
      tempGPObject = updateData(tempGPObject,
                                newX = actual_testX[inds, 1],
                                newY = actual_testY[inds],
                                newT = testT_i)
    }
    rmse_y9 = sqrt(mean((test_data2[, 2] - pred_y9) ^ 2, na.rm = T))
    mae_y9 = mean(abs(test_data2[, 2] - pred_y9), na.rm = T)
    metrics_f[h,] = c(mae_f9, rmse_f9)
    metrics_y[h,] = c(mae_y8, rmse_y9)
  }
  colnames(metrics_f) = c('MAE_f_WD', 'RMSE_f_WD')
  colnames(metrics_y) = c('MAE', 'RMSE')
  metrics_1 = metrics_y
  write.csv(metrics_1, paste0('dataset1/results_wdtgp/metrics_', t, '.csv'),
            row.names = F)
}


### DATASET 2
# Load Data
data = read.csv('dataset_2.csv', header = T)
for (t in 1:160) {
  pred_speed = read.csv(paste0(
    'dataset2/results_speed/ffnn/windspeed_', t, '_1', '.csv'))[13:8212, 1:12]
  for (h in 2:12) {
    pred_speed_x = read.csv(paste0(
      'dataset2/results_speed/ffnn/windspeed_', t, '_', 1, '.csv'))[13:8212, 1:12]
    pred_speed = cbind(pred_speed, pred_speed_x)
  }
  train_index =  1:15592
  test_index = 15593:22838
  time_index = 1:22838
  actual_data = data[, c((7*(t-1))+2, (7*(t-1))+4)]
  actual_data[is.na(actual_data[,2]), 2] = rnorm(
    length(which(is.na(actual_data[,2]) == T)),0,1e-8)
  train_data = actual_data[train_index, ]
  metrics_f = matrix(NA, nrow = 12, ncol = 12)
  metrics_y = matrix(NA, nrow = 12, ncol = 12)
  for (h in 1:12) {
    test_data2 = actual_data[test_index - h, ]
    # TempGP Using Persistence data - W
    load(paste0("dataset2/results_df/tempGPObject_", t, ".RData"))
    pred_f9 = predict(tempGPObject, test_data2[, 1])
    rmse_f9 = sqrt(mean((test_data2[, 2] - pred_f9) ^ 2, na.rm = T))
    mae_f9 = mean(abs(test_data2[, 2] - pred_f9), na.rm = T)
    load(paste0("dataset2/results_df/tempGPObject_", t, ".RData"))
    pred_y9 = rep(0, length(pred_f9))
    sixes = length(pred_f9) / 6
    actual_testX = actual_data[test_index, 1, drop = F]  # Actual wind data X
    actual_testY = actual_data[test_index, 2]  # Actual wind data Y
    for (i in 1:sixes) {
      inds = (6 * (i - 1) + 1):(6 * (i - 1) + 6)
      testX_i = as.matrix(test_data2[inds, 1, drop = F])
      testT_i = test_index[inds]
      pred_y9[inds] = predict(tempGPObject, testX_i, testT_i)
      tempGPObject = updateData(tempGPObject,
                                newX = actual_testX[inds, 1],
                                newY = actual_testY[inds],
                                newT = testT_i)
    }
    rmse_y9 = sqrt(mean((test_data2[, 2] - pred_y9) ^ 2, na.rm = T))
    mae_y9 = mean(abs(test_data2[, 2] - pred_y9), na.rm = T)
    metrics_f[h,] = c(mae_f9, rmse_f9)
    metrics_y[h,] = c(mae_y8, rmse_y9)
  }
  colnames(metrics_f) = c('MAE_f_WD', 'RMSE_f_WD')
  colnames(metrics_y) = c('MAE', 'RMSE')
  metrics_2 = metrics_y
  write.csv(metrics_2, paste0('dataset2/results_wdtgp/metrics_', t, '.csv'),
            row.names = F)
}













