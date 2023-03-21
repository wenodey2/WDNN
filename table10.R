# Persistence Metrics Function
metrics_persistence <-
  function(data, columns, start_index, end_index, n_turbines, max_h) {
    test_index = start_index:end_index
    data = data[, columns]
    data_actual = data[test_index, ]
    mae_mat = matrix(NA, nrow = max_h, ncol = n_turbines)
    rmse_mat = matrix(NA, nrow = max_h, ncol = n_turbines)
    for (h in 1:max_h) {
      data_pred = data[test_index - h, ]
      
      for (t in 1:n_turbines) {
        rmse_mat[h, t] = sqrt(mean((data_actual[, t] - data_pred[, t]) ^ 2, na.rm = T))
        mae_mat[h, t] = mean(abs(data_actual[, t] - data_pred[, t]), na.rm = T)
      }
    }
    metrics = cbind(rowMeans(mae_mat), rowMeans(rmse_mat))
    colnames(metrics) = c('MAE', 'RMSE')
    metrics
  }

# Data Import
data2 = read.csv('dataset_2.csv')

# Power Metrics
metrics_per_p_2 = metrics_persistence(data2, seq(5, 253, 7), 15592, 22838, 160, 12)


### BINNING
# Binning Metrics Function
binning = function(train.y, train.x, bin_width, test.x) {
  library(stats)
  train.y = as.numeric(train.y)
  train.x = as.numeric(train.x)
  test.x = as.numeric(test.x)
  start = 0
  end = round(max(train.x))
  n_bins = round((end - start) / bin_width, 0) + 1
  x_bin = 0
  y_bin = 0
  for (n in 2:n_bins) {
    bin_element = which(train.x > (start + (n - 1) * bin_width) &
                          train.x < (start + n * bin_width))
    x_bin[n] = mean(train.x[bin_element])
    y_bin[n] = mean(train.y[bin_element])
  }
  binned_data = data.frame(x_bin, y_bin)
  binned_data = binned_data[which(is.na(binned_data$y_bin) == F), ]
  splinefit = smooth.spline(x = binned_data$x_bin,
                            y = binned_data$y_bin,
                            all.knots = T)
  y_pred = predict(splinefit, test.x)$y
  return(y_pred)
}

data = read.csv('dataset_2.csv')
train_index =  1:15592
test_index = 15593:22838
power_pred3 = array(NA, c(12, 2, 160))
for (t in 1:160) {
  actual_data = data[, c((6*(t-1))+2, (6*(t-1))+5)]
  train_data = actual_data[train_index, ]
  test_data = actual_data[test_index, ]
  metrics_power3 = matrix(NA, nrow = 12, ncol = 2)
  for (h in 1:12) {
    test_data_pred = actual_data[test_index - h, ]
    # Binning
    bin_pred = binning(train_data[, 2], train_data[, 1], 0.1, test_data_pred[, 1])
    rmse_power3 = sqrt(mean((test_data[, 2] - bin_pred) ^ 2, na.rm = T))
    mae_power3 = mean(abs(test_data[, 2] - bin_pred), na.rm = T)
    # Metrics
    metrics_power3[h, ] = c(mae_power3, rmse_power3)
  }
  power_pred3[, , t] = metrics_power3
}
metrics_bin_p_2 = matrix(NA, nrow = 12, ncol = 2)
colnames(metrics_bin_p_1) = c('MAE', 'RMSE')
for (h in 1:12) {
  metrics_bin_p_2[h, 1] = mean(power_pred3[h, 1, ], na.rm = T)
  metrics_bin_p_2[h, 2] = mean(power_pred3[h, 2, ], na.rm = T)
}



### DEEP FORECAST - TEMPGP
# Deep Forecast - TempGP Metrics Function
metrics_dftgp <- function(dataset, n_turbines, max_h) {
  metrics_array = array(data = NA, dim = c(max_h, 2, n_turbines))
  for (t in 1:n_turbines) {
    dat = read.csv(paste0("dataset", dataset,"/results_df/metrics_df_", t, ".csv"))
    metrics_array[, , t] = as.matrix(dat[, c(3, 4)])
  }
  metrics_array_re = array(data = NA, dim = c(n_turbines, 2, 12))
  for (t in 1:n_turbines) {
    for (h in 1:max_h) {
      metrics_array_re[t, , h] = metrics_array[h, , t]
    }
  }
  metrics = t(colMeans(metrics_array_re))
  colnames(metrics) = c("MAE", "RMSE")
  metrics
}

# Power Metrics
metrics_df_p_2 = metrics_dftgp(1, 160, 12)


### STAN - TEMPGP
# STAN Metrics Function
metrics_stan <- function(dataset, n_turbines, max_h) {
  metrics_array = array(data = NA, dim = c(max_h, 2, n_turbines))
  for (t in 1:n_turbines) {
    dat = read.csv(paste0("dataset", dataset, "/results_stan_tgp/metrics_stan_", t, ".csv"))
    metrics_array[, , t] = as.matrix(dat[, c(3, 4)])
  }
  metrics_array_re = array(data = NA, dim = c(n_turbines, 2, 12))
  for (t in 1:n_turbines) {
    for (h in 1:max_h) {
      metrics_array_re[t, , h] = metrics_array[h, , t]
    }
  }
  metrics = t(colMeans(metrics_array_re))
  colnames(metrics) = c("MAE", "RMSE")
  metrics
}

# Power Metrics
metrics_stan_p_2 = metrics_dftgp(2, 160, 12)


### WDNN - TEMPGP
# WDNN-TGP Metrics Function
metrics_wdtgp <- function(dataset, n_turbines, max_h) {
  metric_data = array(NA, c(max_h, 2, n_turbines))
  for (t in 1:n_turbines) {
    metric_data[ , , t] = as.matrix(
      read.csv(paste0("dataset", dataset, "/results_wdtgp/metrics_", t, ".csv")))
  }
  metric_mat = matrix(NA, nrow = max_h, ncol = 2)
  for (t in 1:n_turbines) {
    for (i in 1:max_h) {
      for (j in 1:2) {
        metric_mat[i, j] = mean(metric_data[i, j, ])
      }
    }
  }
  colnames(metric_mat) = c("MAE", "RMSE")
  metric_mat
}

# Power Metrics
metrics_wdtgp_p_2 = metrics_wdtgp(2, 160, 12)


### WDNN (Power)
wdnn_power = read.csv(paste0('dataset1/results_power/ffnn/metrics_wdnn_p.csv'))
metrics_wdnn_p_2 = wdnn_power[13:24,]


# Table 9 MAE
hours = 1:12
# Table 10 MAE
tab10m = cbind(hours, metrics_per_p_2[,1], metrics_bin_p_2[,1], metrics_df_p_2[,1], 
               metrics_stan_p_2[,1], metrics_wdtgp_p_2[,1], metrics_wdnn_p_2[,1])
colnames(tab10m) = c("h (hours)", "Per", "Bin", "DF-TGP", "STAN-TGP", "WDNN(s)-TGP", "WDNN(p)")

# Table 10 RMSE
tab10r = cbind(hours, metrics_per_p_2[,2], metrics_bin_p_2[,2], metrics_df_p_2[,2], 
               metrics_stan_p_2[,2], metrics_wdtgp_p_2[,2], metrics_wdnn_p_2[,2])
colnames(tab10r) = c("h (hours)", "Per", "Bin", "DF-TGP", "STAN-TGP", "WDNN(s)-TGP", "WDNN(p)")


# Write to File
write.csv(tab10m,'tables/table10.csv', row.names = F)


