rm(list = ls())

### PERSISTENCE
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
data1 = read.csv('dataset_1.csv')
data2 = read.csv('dataset_2.csv')
data3 = read.csv('dataset_3.csv')

# Speed Metrics
metrics_per_s_1 = metrics_persistence(data1, seq(2, 250, 7), 18294, 26471, 36, 12)
metrics_per_s_2 = metrics_persistence(data2, seq(2, 960, 6), 15593, 22838, 160, 12)
metrics_per_s_3 = metrics_persistence(data3, seq(2, 300, 3), 43825, 52590, 100, 12)

# Power Metrics
metrics_per_p_1 = metrics_persistence(data1, seq(4, 252, 7), 18294, 26471, 36, 12)
metrics_per_p_2 = metrics_persistence(data2, seq(5, 963, 6), 15593, 22838, 160, 12)

metrics_per = cbind(metrics_per_s_1, metrics_per_s_2, metrics_per_s_3,
                    metrics_per_p_1, metrics_per_p_2)
colnames(metrics_per) = c('MAE_s_1', 'RMSE_s_1', 'MAE_s_2', 'RMSE_s_2',
                          'MAE_s_3', 'RMSE_s_3', 'MAE_p_1', 'RMSE_p_1', 
                          'MAE_p_2', 'RMSE_p_2')
write.csv(metrics, "metrics_persistence.csv", row.names = F)
