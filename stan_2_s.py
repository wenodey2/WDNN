import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import sys

DEVICE = 'cuda'

data_path = 'dataset_2.csv'
num_turbines = 100
turbine_no = 1
TRAIN_LEN = 13432
VAL_LEN = 2160

for turbine_no in range(1, num_turbines):

    start_time = time.time()

    class NRELData1d:
        def __init__(self, folder_path='', file_path=data_path):
            df_wind_speed = pd.read_csv(folder_path + file_path, index_col=0).loc[
                            :, [f'Turbine{turbine_no}_Speed']]
            self.df_wind_speed = df_wind_speed.T
            self.wind_speed_tensor = torch.tensor(
                self.df_wind_speed.to_numpy(), dtype=torch.float32)
            self._scale_data()

        def _scale_data(self, lwr=0, upr=40, scale=1.):
            self.wind_speed_tensor = 2 * scale * \
                                     (self.wind_speed_tensor - lwr) / upr - scale


    class DataMgr_Simple(NRELData1d):
        def __init__(self, folder_path='', file_path=data_path,
                     train_len=TRAIN_LEN, val_len=VAL_LEN,
                     ENC_LEN=48, DEC_LEN=12):
            super().__init__(folder_path=folder_path, file_path=file_path)

            self.data = self.wind_speed_tensor.to(DEVICE)
            self.total_len = ENC_LEN + DEC_LEN
            self.enc_len = ENC_LEN
            self.dec_len = DEC_LEN
            self.data = self.data.unsqueeze(-1)
            self.train_data = self.data[:, :TRAIN_LEN, :]
            self.val_data = self.data[:, TRAIN_LEN:(TRAIN_LEN + VAL_LEN), :]
            self.test_data = self.data[:, (TRAIN_LEN + VAL_LEN):, :]


    d_model = 64
    FFN_DIM = 512
    heads = 8
    TIME_STEP = 48
    HIDDEN_SIZE = 32
    INPUT_SIZE = 1
    NUM_TURBINES = 1


    class ScaledDotProductAttention(nn.Module):
        """Scaled dot-product attention mechanism."""

        def __init__(self, attention_dropout=0.0):
            super(ScaledDotProductAttention, self).__init__()
            self.dropout = nn.Dropout(attention_dropout).to(DEVICE)
            self.softmax = nn.Softmax(dim=2).to(DEVICE)

        def forward(self, q, k, v, scale=None, attn_mask=None):
            attention = torch.bmm(q, k.transpose(1, 2))
            if scale:
                attention = attention * scale
            attention = self.softmax(attention)
            attention = self.dropout(attention)
            context = torch.bmm(attention, v)
            return context, attention


    class MultiHeadAttention(nn.Module):

        def __init__(self, model_dim=d_model, num_heads=heads, dropout=0.0):
            super(MultiHeadAttention, self).__init__()

            self.dim_per_head = model_dim // num_heads
            self.num_heads = num_heads
            self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads).to(DEVICE)
            self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads).to(DEVICE)
            self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads).to(DEVICE)

            self.dot_product_attention = ScaledDotProductAttention(dropout)
            self.linear_final = nn.Linear(model_dim, model_dim).to(DEVICE)
            self.dropout = nn.Dropout(dropout).to(DEVICE)

            self.layer_norm = nn.LayerNorm(model_dim).to(DEVICE)

        def forward(self, key, value, query, attn_mask=None):
            # residual connection
            residual = query
            dim_per_head = self.dim_per_head
            num_heads = self.num_heads
            batch_size = key.size(0)

            # linear projection
            key = self.linear_k(key)
            value = self.linear_v(value)
            query = self.linear_q(query)

            # split by heads
            key = key.view(batch_size * num_heads, -1, dim_per_head)
            value = value.view(batch_size * num_heads, -1, dim_per_head)
            query = query.view(batch_size * num_heads, -1, dim_per_head)

            # scaled dot product attention
            scale = (key.size(-1)) ** -0.5
            context, attention = self.dot_product_attention(query, key, value, scale)

            # concat heads
            context = context.view(batch_size, -1, dim_per_head * num_heads)

            # final linear projection
            output = self.linear_final(context)

            # dropout
            output = self.dropout(output)

            # add residual and norm layer
            output = self.layer_norm(residual + output)

            return output, attention


    class PositionalWiseFeedForward(nn.Module):

        def __init__(self, model_dim=d_model, ffn_dim=64, dropout=0.0):
            super(PositionalWiseFeedForward, self).__init__()
            self.w1 = nn.Conv1d(model_dim, ffn_dim, 1).to(DEVICE)
            self.w2 = nn.Conv1d(ffn_dim, model_dim, 1).to(DEVICE)
            self.dropout = nn.Dropout(dropout).to(DEVICE)
            self.layer_norm = nn.LayerNorm(model_dim).to(DEVICE)
            self.relu = nn.ReLU().to(DEVICE)

        def forward(self, x):
            output = x.transpose(1, 2)
            output = self.w2(self.relu(self.w1(output)))
            output = self.dropout(output.transpose(1, 2))

            # add residual and norm layer
            output = self.layer_norm(x + output)
            return output


    class Attn(nn.Module):
        def __init__(self, hidden_size):
            super(Attn, self).__init__()

            self.hidden_size = hidden_size

            self.attn = nn.Linear(self.hidden_size, hidden_size).to(DEVICE)

        def forward(self, hidden, encoder_outputs):
            encoder_outputs = encoder_outputs.transpose(1, 0)
            max_len = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)

            # Create variable to store attention energies
            attn_energies = torch.zeros(batch_size, max_len)  # B x S
            attn_energies = attn_energies.to(DEVICE)

            # For each batch of encoder outputs
            for b in range(batch_size):
                # Calculate energy for each encoder output
                for i in range(max_len):
                    attn_energies[b, i] = self.score(hidden[:, b],
                                                     encoder_outputs[i, b].unsqueeze(0))

            # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

        def score(self, hidden, encoder_output):

            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy


    class EncoderLayer(nn.Module):

        def __init__(self, model_dim=d_model, num_heads=heads, ffn_dim=2048, dropout=0.0):
            super(EncoderLayer, self).__init__()

            self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
            self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

        def forward(self, inputs):
            # self attention
            context, attention = self.attention(inputs, inputs, inputs)

            # feed forward network
            output = self.feed_forward(context)

            return output, attention


    class Encoder(nn.Module):

        def __init__(self, num_layers=1, model_dim=d_model, num_heads=heads,
                     ffn_dim=FFN_DIM, dropout=0.0):
            super(Encoder, self).__init__()
            self.model_dim = model_dim

            self.encoder_layers = nn.ModuleList(
                [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
            self.linear = nn.Linear(1, model_dim).to(DEVICE)
            self.out = nn.Linear(TIME_STEP, 1).to(DEVICE)

            self.relu = nn.ReLU(True).to(DEVICE)

        def forward(self, inputs):
            output = self.linear(inputs)

            for encoder in self.encoder_layers:
                output, attention = encoder(output)

            x = output[:, 0].unsqueeze(1)

            return x


    class Encoder2(nn.Module):
        def __init__(self):
            super(Encoder2, self).__init__()

            self.rnn = nn.RNN(
                input_size=d_model,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                batch_first=True,
            ).to(DEVICE)
            self.out = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

        def forward(self, x, h_state):
            # x (batch, time_step, input_size)
            # h_state (n_layers, batch, hidden_size)
            # r_out (batch, time_step, hidden_size)
            r_out, h_state = self.rnn(x, h_state)
            return r_out, h_state


    class Decoder(nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            self.rnn = nn.RNN(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=1,
                batch_first=True
            ).to(DEVICE)
            self.relu = nn.ReLU().to(DEVICE)
            self.out = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

        def forward(self, input, hidden, ):
            output, hidden = self.rnn(input, hidden)
            out = self.out(output[:, -1, :])
            return out, hidden, output


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.encoder = Encoder()
            self.encoder2 = Encoder2()
            self.decoder = Decoder()
            self.attn = Attn(HIDDEN_SIZE)
            self.concat = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE).to(DEVICE)
            self.out = nn.Linear(HIDDEN_SIZE, 1).to(DEVICE)

        def forward(self, input):
            input_T = input.transpose(1, 2).to(DEVICE)
            batch = input_T.size()[0]
            outputs = torch.zeros(TIME_STEP, batch, d_model).to(DEVICE)
            for k in range(TIME_STEP):
                x = input_T[:, :, k].unsqueeze(2)
                output = self.encoder(x)
                outputs[k] = output.permute(1, 0, 2)

            outputs = outputs.permute(1, 0, 2)

            encoder_output, encoder_hidden = self.encoder2(outputs, None)
            decoder_hidden = encoder_hidden

            temp = torch.zeros_like(input[:, -1].unsqueeze(1)).to(DEVICE)
            out, hidden, decoder_output = self.decoder(temp, decoder_hidden)

            energies = self.attn(hidden, encoder_output)
            context = energies.bmm(encoder_output)
            concat_input = torch.cat((decoder_output.squeeze(), context.squeeze()), dim=1)
            concat_output = torch.tanh(self.concat(concat_input))
            pred = self.out(concat_output)
            pred = pred.squeeze(1)
            return pred


    class wpDataset(Dataset):
        def __init__(self, data, ENC_LEN=48, DEC_LEN=12, target_turbine=0, target_time=0):
            self.data = data
            self.enc_len = ENC_LEN
            self.total_len = ENC_LEN + DEC_LEN
            self.target_time = target_time
            self.target_turbine = target_turbine

        def __getitem__(self, index):
            one_point = self.data[:, index:(index + self.total_len), :]
            x = one_point[:, :self.enc_len, 0]
            y = one_point[:, self.enc_len:self.total_len, 0]

            y = y[self.target_turbine, self.target_time]
            return x, y

        def __len__(self):
            return self.data.size(1) - self.total_len


    def cal_loss(y_true, y_pred):
        y_true = (y_true + 1) / 2
        y_pred = (y_pred + 1) / 2
        diff = y_true - y_pred

        x = diff.detach()
        idx = ~torch.isnan(x)
        mae = abs(x[idx]).mean().item()
        rmse = (x[idx] ** 2).mean().item() ** .5

        return mae, rmse


    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.1)
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
                     BATCH_SIZE, TARGET_TURBINE=0, TARGET_TIME=0, ENC_LEN=48, DEC_LEN=12):
            self.model = model

            train_dataset = wpDataset(data_mgr.train_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                      target_turbine=TARGET_TURBINE, target_time=TARGET_TIME)
            val_dataset = wpDataset(data_mgr.val_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                    target_turbine=TARGET_TURBINE, target_time=TARGET_TIME)
            test_dataset = wpDataset(data_mgr.test_data, ENC_LEN=ENC_LEN, DEC_LEN=DEC_LEN,
                                     target_turbine=TARGET_TURBINE, target_time=TARGET_TIME)
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
                train_loss, train_mae, train_rmse = self.fit()
                print(f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}')
                print(f'Train RMSE: {train_rmse:.4f}')

                val_loss, val_mae, val_rmse = self.validate()
                print(f'Val Loss: {val_loss:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
                print(f'Val RMSE: {val_rmse:.4f}')

                early_stopping(val_loss)
                print(f'Best Val Loss: {early_stopping.best_loss:.4f}')

                # if epoch%2 == 1:
                #     self.report_test_error()
                if early_stopping.early_stop:
                    torch.save(self.model.to('cpu').state_dict(), self.SAVE_FILE + '.pt')
                    break
            else:
                torch.save(self.model.to('cpu').state_dict(), self.SAVE_FILE + '.pt')

        def fit(self):
            print('Training')
            self.model.train()
            counter = 0
            running_loss = 0.
            running_mae = 0.
            running_rmse = 0.
            prog_bar = tqdm(enumerate(self.train_dataloader),
                            total=int(len(self.train_dataset) / self.train_dataloader.batch_size))
            for i, data in prog_bar:
                counter += 1
                self.optimizer.zero_grad()
                x = data[0]
                x = x.permute(0, 2, 1)
                y_pred = self.model(x)
                y_true = data[1]

                idx = ~torch.isnan(y_true)
                loss = self.criterion(y_pred[idx], y_true[idx])
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                mae, rmse = cal_loss(y_true, y_pred)
                running_mae += mae
                running_rmse += rmse

            train_loss = running_loss / counter
            train_mae = running_mae / counter
            train_rmse = running_rmse / counter

            return train_loss, train_mae, train_rmse

        def validate(self):
            print('Validating')
            self.model.eval()
            counter = 0
            running_loss = 0.
            running_mae = 0.
            running_rmse = 0.

            prog_bar = tqdm(enumerate(self.val_dataloader),
                            total=int(len(self.val_dataset) / self.val_dataloader.batch_size))
            with torch.no_grad():
                for i, data in prog_bar:
                    counter += 1
                    x = data[0]
                    x = x.permute(0, 2, 1)
                    y_pred = self.model(x)
                    y_true = data[1]

                    idx = ~torch.isnan(y_true)
                    loss = self.criterion(y_pred[idx], y_true[idx])
                    running_loss += loss.item()

                    mae, rmse = cal_loss(y_true, y_pred)
                    running_mae += mae
                    running_rmse += rmse

                val_loss = running_loss / counter
                val_mae = running_mae / counter
                val_rmse = counter / counter

            return val_loss, val_mae, val_rmse

        def report_test_error(self):
            print(' ')
            print('Calculating Test Error')
            self.model.eval()
            counter = 0
            running_loss = 0.
            running_mae = 0.
            running_rmse = 0.
            y_pred_all = []
            y_true_all = []

            prog_bar = tqdm(enumerate(self.test_dataloader),
                            total=int(len(self.test_dataset) / self.test_dataloader.batch_size))
            with torch.no_grad():
                for i, data in prog_bar:
                    counter += 1
                    x = data[0]
                    x = x.permute(0, 2, 1)
                    y_pred = self.model(x)
                    y_true = data[1]
                    y_pred_all.append(y_pred)
                    y_true_all.append(y_true)

                    idx = ~torch.isnan(y_true)

                    loss = self.criterion(y_pred[idx], y_true[idx])
                    running_loss += loss.item()

                    mae, rmse = cal_loss(y_true, y_pred)
                    running_mae += mae
                    running_rmse += rmse

                test_loss = running_loss / counter
                test_mae = running_mae / counter
                test_rmse = running_rmse / counter

            print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')
            print(f'Test RMSE: {test_rmse:.4f}')

            return test_loss, test_mae, test_rmse, y_pred_all, y_true_all


    BATCH_SIZE = 512
    EPOCHS = 20
    LR = .001

    for hour_ahead in range(1,13):
        data_mgr = DataMgr_Simple()

        tim = [hour_ahead - 1]
        turbine_list = [*range(0, NUM_TURBINES)]
        final_mae = []
        final_rmse = []
        predictions = []
        predictions_array = []
        actual_values = []
        actual_values_array = []

        for t in tim:
            out_mae = []
            out_rmse = []
            count = 0
            for turbine in turbine_list:
                print(f'Currently on turbine {turbine}')
                model = nn.DataParallel(Model())
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                print(f'model parameters:{count_parameters(model)}')
                model.apply(init_weights)
                SAVE_FILE = f'dataset2/results_stan/models/time{str(t + 1)}turbine{str(turbine)}'
                trainer = Trainer(model=model, data_mgr=data_mgr, optimizer=optimizer,
                                  criterion=criterion,
                                  SAVE_FILE=SAVE_FILE, BATCH_SIZE=BATCH_SIZE,
                                  TARGET_TURBINE=turbine, TARGET_TIME=t)
                trainer.train(epochs=EPOCHS)
                trainer.model = trainer.model.to(DEVICE)
                loss, mae, rmse, pred, true = trainer.report_test_error()

                count += 1
                out_mae.append(mae)
                out_rmse.append(rmse)
                running_mae = sum(out_mae) / count
                running_rmse = sum(out_rmse) / count
                predictions.append(torch.cat(pred))
                actual_values.append(torch.cat(true))

                p0 = predictions[turbine].detach().cpu().numpy()
                predictions_array.append(p0.reshape((p0.shape[0], 1)))
                a0 = actual_values[turbine].detach().cpu().numpy()
                actual_values_array.append(a0.reshape((p0.shape[0], 1)))

                print(f'Running MAE: {running_mae:.4f}, Running RMSE: {running_rmse:.4f}')

            out_df = pd.DataFrame({'MAE': out_mae, 'RMSE': out_rmse}, index=turbine_list)
            out_df.to_csv(f'results/metrics_{hour_ahead}_{turbine_no}.csv')
            final_mae.append(sum(out_mae) / len(turbine_list))
            final_rmse.append(sum(out_rmse) / len(turbine_list))

        av_array = (np.concatenate(actual_values_array, axis=1) + 1) * 20
        pv_array = (np.concatenate(predictions_array, axis=1) + 1) * 20

        np.savetxt(f'dataset2/results_stan/actual_speed_{hour_ahead}_{turbine_no}.csv.csv',
                   av_array[48:, :], delimiter=',', comments='')
        np.savetxt(f'dataset2/results_stan/pred_speed_{hour_ahead}_{turbine_no}.csv.csv',
                   pv_array[48:, :], delimiter=',', comments='')

    end_time = time.time()
    print(f"{(end_time - start_time) / 3600} hours")
