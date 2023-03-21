import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

start_time = time.time()

data_path = 'dataset_3.csv'
meta_path = 'dataset_3_meta.csv'


class NRELDataSimple:
    def __init__(self, folder_path='', file_path=data_path):
        df_wind_speed = pd.read_csv(folder_path + file_path, index_col=0)
        df_meta = pd.read_csv(folder_path + meta_path, index_col=0)
        turbine_idx = df_meta['site_id'].tolist()
        self.df_wind_speed = df_wind_speed.T
        self.df_wind_speed.index = turbine_idx
        self._reformat_1d_to_2d()
        self._scale_data()

    def _reformat_1d_to_2d(self):
        gid_list = list(self.df_wind_speed.index)
        rearrange_list = [sorted(gid_list)[x:x + 10] for x in range(0, len(gid_list), 10)]

        self.pos_list = np.array(rearrange_list).transpose()[::-1].tolist()
        wind_speed_tensors = [torch.tensor(self.df_wind_speed.loc[self.pos_list[i]].to_numpy(),
                                           dtype=torch.float32).unsqueeze(0) for i in range(10)]
        self.wind_speed_tensor = torch.cat(wind_speed_tensors, dim=0)

    def _scale_data(self, lwr=0, upr=40, scale=1.):
        self.wind_speed_tensor = 2 * scale * (self.wind_speed_tensor - lwr) / upr - scale


class NRELDataMgr(NRELDataSimple):
    def __init__(self, folder_path='', file_path=data_path,
                 train_len=39408, val_len=4416,
                 random_state=42, ENC_LEN=3 * 6, DEC_LEN=24 * 6):
        super().__init__(folder_path=folder_path, file_path=file_path)

        self.data = self.wind_speed_tensor
        self.total_len = ENC_LEN + DEC_LEN
        self.enc_len = ENC_LEN
        self.dec_len = DEC_LEN

        self.train_data = self.data[:, :, :train_len]
        self.val_data = self.data[:, :, train_len:(train_len + val_len)]
        self.test_data = self.data[:, :, (train_len + val_len):]


class wpDataset(Dataset):
    def __init__(self, data, ENC_LEN=3 * 6, DEC_LEN=24 * 6, TARGET_TIM=0):
        self.data = data
        self.enc_len = ENC_LEN
        self.total_len = ENC_LEN + DEC_LEN
        self.target_tim = TARGET_TIM

    def __getitem__(self, index):
        one_point = self.data[:, :, index:(index + self.total_len)]
        x = one_point[:, :, :self.enc_len]
        y = one_point[:, :, self.enc_len:]
        x = x.permute(2, 0, 1)
        y = y[:, :, self.target_tim]
        y = y.reshape(-1)

        return x, y

    def __len__(self):
        return self.data.size(2) - self.total_len


class windSpeedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, [3, 3])
        self.maxpool = nn.MaxPool2d([2, 2])
        self.conv2 = nn.Conv2d(20, 50, [3, 3])
        self.conv3 = nn.Conv2d(50, 200, [2, 2])
        self.fc = nn.Linear(200, 200)
        # self.batch_norm = nn.BatchNorm1d(200)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), 200)
        x = self.fc(x)
        # x = self.batch_norm(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(200, 500)
        self.wpcnn = windSpeedCNN()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        swsm_list = []
        for i in range(18):
            swsm_list.append(self.wpcnn(x[:, [i], :, :]).unsqueeze(-1))
        rnn_input = torch.cat(swsm_list, dim=2)
        rnn_input = rnn_input.permute(2, 0, 1)
        output, (hidden, cell_state) = self.rnn(rnn_input)
        hidden = torch.tanh(self.fc1(hidden.squeeze(0))).unsqueeze(0)
        cell_state = torch.tanh(self.fc2(cell_state.squeeze(0))).unsqueeze(0)
        init_y = swsm_list[-1].permute(2, 0, 1)

        return init_y, (hidden, cell_state)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(200, 100)

    def forward(self, init_y, hidden):
        output, (hidden, cell_state) = self.rnn(init_y, hidden)
        return output


class PSTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Linear(100, 100)

    def forward(self, x):
        init_y, hidden = self.encoder(x)
        y_pred = self.decoder(init_y, hidden)
        y_pred = y_pred.squeeze(0)
        y_pred = torch.sigmoid(self.fc(y_pred)) * 2. - 1.
        return y_pred


def cal_loss(y_true, y_pred):
    y_true = (y_true + 1) * 20
    y_pred = (y_pred + 1) * 20
    diff = y_true - y_pred

    l1_x = abs(diff)
    l2_x = diff ** 2
    l1_p = abs(diff).sum() / y_true.sum()
    return l1_x.mean().item(), l2_x.mean().item() ** .5, l1_p.item()


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=.1)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class Trainer:
    def __init__(self, model, data_mgr, optimizer, criterion, SAVE_FILE,
                 TARGET_TIM, BATCH_SIZE, ENC_LEN=3 * 6, DEC_LEN=144):
        self.model = model

        train_dataset = wpDataset(data_mgr.train_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                  TARGET_TIM=TARGET_TIM)
        val_dataset = wpDataset(data_mgr.val_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                TARGET_TIM=TARGET_TIM)
        test_dataset = wpDataset(data_mgr.test_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                 TARGET_TIM=TARGET_TIM)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.optimizer = optimizer
        self.criterion = criterion

        self.SAVE_FILE = SAVE_FILE

    def train(self, epochs):
        early_stopping = EarlyStopping()
        for epoch in range(epochs):
            print(' ')
            print(f"Epoch {epoch + 1} of {epochs}")
            train_loss, train_mae, train_rmse, train_mape = self.fit()
            print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
            print(f'Train RMSE: {train_rmse:.4f}, Train MAPE: {train_mape:.4f}')

            val_loss, val_mae, val_rmse, val_mape = self.validate()
            print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
            print(f'Val RMSE: {val_rmse:.4f}, Val MAPE: {val_mape:.4f}')

            early_stopping(val_loss)
            print(f'Best Val Loss: {early_stopping.best_loss:.4f}')

            if early_stopping.early_stop:
                torch.save(self.model.state_dict(), self.SAVE_FILE + '.pt')
                break
        else:
            torch.save(self.model.state_dict(), self.SAVE_FILE + '.pt')

    def fit(self):
        print('Training')
        self.model.train()
        counter = 0
        running_loss = 0.
        running_mae = 0.
        running_rmse = 0.
        running_mape = 0.
        prog_bar = tqdm(enumerate(self.train_dataloader),
                        total=int(len(self.train_dataset) / self.train_dataloader.batch_size))
        for i, data in prog_bar:
            counter += 1
            x, y_true = data[0], data[1]
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y_true)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            mae, rmse, mape = cal_loss(y_true, y_pred)
            running_mae += mae
            running_rmse += rmse
            running_mape += mape

        train_loss = running_loss / counter
        train_mae = running_mae / counter
        train_rmse = running_rmse / counter
        train_mape = running_mape / counter

        return train_loss, train_mae, train_rmse, train_mape

    def validate(self):
        print('Validating')
        self.model.eval()
        counter = 0
        running_loss = 0.
        running_mae = 0.
        running_rmse = 0.
        running_mape = 0.
        prog_bar = tqdm(enumerate(self.val_dataloader),
                        total=int(len(self.val_dataset) / self.val_dataloader.batch_size))
        with torch.no_grad():
            for i, data in prog_bar:
                counter += 1
                x, y_true = data[0], data[1]
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y_true)
                running_loss += loss.item()

                mae, rmse, mape = cal_loss(y_true, y_pred)
                running_mae += mae
                running_rmse += rmse
                running_mape += mape

            val_loss = running_loss / counter
            val_mae = running_mae / counter
            val_rmse = running_rmse / counter
            val_mape = running_mape / counter

        return val_loss, val_mae, val_rmse, val_mape

    def report_test_error(self):
        print('Calculating Test Error')
        self.model.eval()
        counter = 0
        running_loss = 0.
        running_mae = 0.
        running_rmse = 0.
        running_mape = 0.
        y_pred_all = []
        y_true_all = []
        prog_bar = tqdm(enumerate(self.test_dataloader),
                        total=int(len(self.test_dataset) / self.test_dataloader.batch_size))
        with torch.no_grad():
            for i, data in prog_bar:
                counter += 1
                x, y_true = data[0], data[1]
                y_pred = self.model(x)
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                loss = self.criterion(y_pred, y_true)
                running_loss += loss.item()

                mae, rmse, mape = cal_loss(y_true, y_pred)
                running_mae += mae
                running_rmse += rmse
                running_mape += mape

            test_loss = running_loss / counter
            test_mae = running_mae / counter
            test_rmse = running_rmse / counter
            test_mape = running_mape / counter

        print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}, Test MAPE: {test_mape:.4f}')

        return test_loss, test_mae, test_rmse, test_mape, y_pred_all, y_true_all


BATCH_SIZE = 256
EPOCHS = 30
LR = .001
target_time = [x - 1 for x in range(1, 13)]
out_mae = []
out_rmse = []
out_mape = []
index_list = []

predictions = []
predictions_array = []
actual_values = []
actual_values_array = []

data_mgr = NRELDataMgr(folder_path='')

for t in target_time:
    print(f'Currently on prediction {t + 1}...')
    model = PSTN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)
    print(f'model parameters:{count_parameters(model)}')
    model.apply(init_weights)
    SAVE_FILE = 'dataset3/results_pstn/models/PSTN' + str(t + 1)
    trainer = Trainer(model=model, data_mgr=data_mgr, optimizer=optimizer, criterion=criterion,
                      SAVE_FILE=SAVE_FILE, TARGET_TIM=t, BATCH_SIZE=BATCH_SIZE)
    trainer.train(epochs=EPOCHS)
    loss, mae, rmse, mape, pred, true = trainer.report_test_error()

    out_mae.append(mae)
    out_rmse.append(rmse)
    out_mape.append(mape)
    index_list.append(t + 1)

    predictions.append(torch.cat(pred))
    actual_values.append(torch.cat(true))

    p0 = (predictions[t].detach().cpu().numpy() + 1) * 20
    pd.DataFrame(p0).to_csv(f"dataset3/results_pstn/predictions_{t + 1}.csv", index=False)
    a0 = (actual_values[t].detach().cpu().numpy() + 1) * 20
    pd.DataFrame(a0).to_csv(f"dataset3/results_pstn/actual_values_{t + 1}.csv", index=False)

    outdf = pd.DataFrame({'MAE': out_mae, 'RMSE': out_rmse}, index=index_list)
    outdf.to_csv("dataset3/results_pstn/metrics_pstn_s_3.csv")

print('Finished!')

end_time = time.time()
print(f"{(end_time - start_time) / 3600} hours")
