# Libraries
library(DSWE)


### DATASET 1
# Load and Prep Data I
data = read.csv('dataset_1.csv', header = T)
num_turbines = 160
for (t in 1:num_tubines) {
  actual_speed = data[, (7*(t-1))+2, drop = FALSE]
  actual_windpower = data[, (7*(t-1))+4, drop = FALSE]
  train_start = 1
  train_end = 18293
  test_end = 26471
  train_index = train_start:train_end
  test_index = (train_end + 1):test_end  # (8212)
  time_index = 1:test_end
  actual_data = data[, 2 * t, drop = F]
  trainx = as.matrix(actual_data[train_index, ])
  trainy = as.matrix(actual_windpower[train_index, 1])
  trainy[is.na(trainy)] = rnorm(length(which(is.na(trainy) == T)), 0, 1e-8)
  
  # Create and Save TempGP Object
  tempGPObject = tempGP(trainx, trainy)
  save(tempGPObject,
       file = paste0("dataset1/results_df/tempGPObject1_", t, ".RData"))
  
  # Load and Prep Data II
  speed_list = NULL
  for (h in 1:12) {
    speed_list[[h]] = read.csv(
      paste0('dataset1/results_df/df_speed_predicted_', h, '.csv'), header = T)
  }
  speed_list[[1]] = rbind(
    speed_list[[1]], speed_list[[1]][nrow(speed_list[[1]]), ])
  pred_speed = speed_list[[1]][, t]
  for (h in 2:12) {
    pred_speed = cbind(pred_speed, speed_list[[h]][t])
  }
  
  train_index =  1:(train_start + 12)
  test_index = (train_end + 1 + 12):test_end
  time_index = 1:test_end
  actual_data = cbind(actual_speed, actual_windpower)
  actual_data[is.na(actual_data[, 2]), 2] = rnorm(
    length(which(is.na(actual_data[, 2]) == T)), 0, 1e-8)
  train_data = actual_data[train_index, ]
  
  # Predictions
  metrics = matrix(NA, nrow = 12, ncol = 4)
  
  for (h in 1:12) {
    test_data2 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 1]))
    rmse_s = sqrt(mean((test_data2[, 2] - test_data2[, 1]) ^ 2, na.rm = T))
    mae_s = mean(abs(test_data2[, 2] - test_data2[, 1]), na.rm = T)
    test_data1 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 2]))
    
    # TempGP
    load(paste0("dataset1/results_df/tempGPObject1_", t, ".RData"))
    pred_f8 = predict(tempGPObject, test_data1[, 1])
    rmse_f8 = sqrt(mean((test_data1[, 2] - pred_f8) ^ 2, na.rm = T))
    mae_f8 = mean(abs(test_data1[, 2] - pred_f8), na.rm = T)
    load(paste0("dataset1/results_df/tempGPObject1_", t, ".RData"))
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
    metrics[h,] = c(mae_s, rmse_s, mae_y8, rmse_y8)
  }
  
  colnames(metrics) = c('MAE_Speed', 'RMSE_Speed', 'MAE_Power', 'RMSE_Power')
  write.csv(metrics_f,
            paste0('dataset1/results_df/metrics_df_', t, '.csv'),
            row.names = F)
}


### DATASET 2
# Load and Prep Data I
data = read.csv('dataset_2.csv', header = T)
num_turbines = 160
for (t in 1:num_tubines) {
  actual_speed = data[, (6*(t-1))+2, drop = FALSE]
  actual_windpower = data[, (6*(t-1))+5, drop = FALSE]
  train_start = 1
  train_end = 15592
  test_end = 22838
  train_index = train_start:train_end
  test_index = (train_end + 1):test_end  # (8212)
  time_index = 1:test_end
  actual_data = data[, 2 * t, drop = F]
  trainx = as.matrix(actual_data[train_index, ])
  trainy = as.matrix(actual_windpower[train_index, 1])
  trainy[is.na(trainy)] = rnorm(length(which(is.na(trainy) == T)), 0, 1e-8)
  
  # Create and Save TempGP Object
  tempGPObject = tempGP(trainx, trainy)
  save(tempGPObject,
       file = paste0("dataset2/results_df/tempGPObject1_", t, ".RData"))
  
  # Load and Prep Data II
  speed_list = NULL
  for (h in 1:12) {
    speed_list[[h]] = read.csv(
      paste0('dataset2/results_df/df_speed_predicted_', h, '.csv'), header = T)
  }
  speed_list[[1]] = rbind(
    speed_list[[1]], speed_list[[1]][nrow(speed_list[[1]]), ])
  pred_speed = speed_list[[1]][, t]
  for (h in 2:12) {
    pred_speed = cbind(pred_speed, speed_list[[h]][t])
  }
  
  train_index =  1:(train_start + 12)
  test_index = (train_end + 1 + 12):test_end 
  time_index = 1:test_end
  actual_data = cbind(actual_speed, actual_windpower)
  actual_data[is.na(actual_data[, 2]), 2] = rnorm(
    length(which(is.na(actual_data[, 2]) == T)), 0, 1e-8)
  train_data = actual_data[train_index, ]
  
  # Predictions
  metrics = matrix(NA, nrow = 12, ncol = 4)
  
  for (h in 1:12) {
    test_data2 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 1]))
    rmse_s = sqrt(mean((test_data2[, 2] - test_data2[, 1]) ^ 2, na.rm = T))
    mae_s = mean(abs(test_data2[, 2] - test_data2[, 1]), na.rm = T)
    test_data1 = as.matrix(cbind(pred_speed[, h],  actual_data[test_index, 2]))
    
    # TempGP
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
    metrics[h,] = c(mae_s, rmse_s, mae_y8, rmse_y8)
  }
  
  colnames(metrics) = c('MAE_Speed', 'RMSE_Speed', 'MAE_Power', 'RMSE_Power')
  write.csv(metrics_f,
            paste0('dataset2/results_df/metrics_df_', t, '.csv'),
            row.names = F)
}
