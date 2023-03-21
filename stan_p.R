# Libraries
library(DSWE)


### DATASET 1
# Load Data
data = read.csv('dataset1_csv', header = T)
for (t in 1:36) {
  speed_list = NULL
  for (h in 1:12) {
    speed_list[[h]] = read.csv(
      paste0('dataset1/results_stan/pred_speed_', h, '_', t,'.csv.csv'), header = F)
  }
  pred_speed = speed_list[[1]]
  for (h in 2:12) {
    pred_speed = cbind(pred_speed, speed_list[[h]])
  }
  
  # Prep Data
  actual_speed = data[, (7*(t-1))+2, drop = FALSE]
  actual_windpower = data[, (7*(t-1))+4, drop = FALSE]
  
  train_end = 18293
  
  # Predictions
  metrics = matrix(NA, nrow = 12, ncol = 6)
  for (h in 1:12) {
    train_index =  1:(train_end + 105 + h)
    test_index = (train_end + 1 + 106 + h):(24805 + h)
    time_index = 1:(24805 + h)
    actual_data = cbind(actual_speed, actual_windpower)
    actual_data[is.na(actual_data[,2]), 2] = rnorm(
      length(which(is.na(actual_data[,2]) == T)),0,1e-8)
    train_data = actual_data[train_index, ]
    test_data2 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 1]))
    rmse_s = sqrt(mean((test_data2[, 2] - test_data2[, 1]) ^ 2, na.rm = T))
    mae_s = mean(abs(test_data2[, 2] - test_data2[, 1]), na.rm = T) 
    test_data1 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 2]))
    
    # TempGP Using WD - W
    load(paste0("dataset1/results_df/tempGPObject_", t, ".RData"))
    pred_f8 = predict(tempGPObject, test_data1[, 1])
    rmse_f8 = sqrt(mean((test_data1[, 2] - pred_f8) ^ 2, na.rm = T))
    mae_f8 = mean(abs(test_data1[, 2] - pred_f8), na.rm = T)
    load(paste0("dataset1/results_df/tempGPObject_", t, ".RData"))
    pred_y8 = rep(0, length(pred_f8))
    sixes = length(pred_f8) / 6
    actual_testX = actual_data[test_index, 1, drop = F]  # Actual wind data X
    actual_testY = actual_data[test_index, 2]  # Actual wind data Y
    for (i in 1:sixes) {
      inds = (6 * (i - 1) + 1):(6 * (i - 1) + 6)
      testX_i = as.matrix(test_data1[inds, 1, drop = F])
      testT_i = test_index[inds]
      pred_y8[inds] = predict(tempGPObject, testX_i, testT_i)
      tempGPObject = updateData(tempGPObject,
                                newX = actual_testX[inds, 1],
                                newY = actual_testY[inds],
                                newT = testT_i)
    }
    rmse_y8 = sqrt(mean((test_data1[, 2] - pred_y8) ^ 2, na.rm = T))
    mae_y8 = mean(abs(test_data1[, 2] - pred_y8), na.rm = T)
    metrics[h,] = c(mae_s, rmse_s, mae_y8,  rmse_y8)
  }
  colnames(metrics) = c('MAE_Speed', 'RMSE_Speed', 'MAE_Power', 'RMSE_Power')
  write.csv(metrics, paste0('dataset1/results_stan_tgp/metrics_stan_', t, '.csv'), row.names = F)
}


### DATASET 2
# Load Data
data = read.csv('dataset_2.csv', header = T)
for (t in 1:160) {
  speed_list = NULL
  for (h in 1:12) {
    speed_list[[h]] = read.csv(
      paste0('dataset2/results_stan/pred_speed_', h, '_', t,'.csv.csv'), header = F)
  }
  pred_speed = speed_list[[1]]
  for (h in 2:12) {
    pred_speed = cbind(pred_speed, speed_list[[h]])
  }
  
  # Prep Data
  actual_speed = data[, (6*(t-1))+2, drop = FALSE]
  actual_windpower = data[, (6*(t-1))+5, drop = FALSE]
  
  train_end = 15592
  
  # Predictions
  metrics = matrix(NA, nrow = 12, ncol = 6)
  for (h in 1:12) {
    train_index =  1:(train_end + 95 + h)
    test_index = (train_end + 1 + h):(22842 + h)
    time_index = 1:(22843 + h)
    actual_data = cbind(actual_speed, actual_windpower)
    actual_data[is.na(actual_data[,2]), 2] = rnorm(
      length(which(is.na(actual_data[,2]) == T)),0,1e-8)
    train_data = actual_data[train_index, ]
    test_data2 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 1]))
    rmse_s = sqrt(mean((test_data2[, 2] - test_data2[, 1]) ^ 2, na.rm = T))
    mae_s = mean(abs(test_data2[, 2] - test_data2[, 1]), na.rm = T) 
    test_data1 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 2]))
    
    # TempGP Using WD - W
    load(paste0("dataset2/results_df/tempGPObject1_", t, ".RData"))
    pred_f8 = predict(tempGPObject, test_data1[, 1])
    rmse_f8 = sqrt(mean((test_data1[, 2] - pred_f8) ^ 2, na.rm = T))
    mae_f8 = mean(abs(test_data1[, 2] - pred_f8), na.rm = T)
    load(paste0("dataset2/results_df/tempGPObject1_", t, ".RData"))
    pred_y8 = rep(0, length(pred_f8))
    sixes = length(pred_f8) / 6
    actual_testX = actual_data[test_index, 1, drop = F]  # Actual wind data X
    actual_testY = actual_data[test_index, 2]  # Actual wind data Y
    for (i in 1:sixes) {
      inds = (6 * (i - 1) + 1):(6 * (i - 1) + 6)
      testX_i = as.matrix(test_data1[inds, 1, drop = F])
      testT_i = test_index[inds]
      pred_y8[inds] = predict(tempGPObject, testX_i, testT_i)
      tempGPObject = updateData(tempGPObject,
                                newX = actual_testX[inds, 1],
                                newY = actual_testY[inds],
                                newT = testT_i)
    }
    rmse_y8 = sqrt(mean((test_data1[, 2] - pred_y8) ^ 2, na.rm = T))
    mae_y8 = mean(abs(test_data1[, 2] - pred_y8), na.rm = T) 
    metrics[h,] = c(mae_s, rmse_s, mae_y8,  rmse_y8)
  }
  
  colnames(metrics) = c('MAE_Speed', 'RMSE_Speed', 'MAE_Power', 'RMSE_Power')
  write.csv(metrics, paste0('dataset2/results_stan_tgp/metrics_stan_', t, '.csv'), row.names = F)
}
