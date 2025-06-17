import pandas as pd
import torch
import math
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

from src.utils import resolve_root

resolve_root(2)

from src.dataset import create_sequences
from src.model import LSTMModel
from src.modeling.utils import save_model
from src.config import SEQUENCE_LENGTH, DEVICE

import os


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm_model(data_path, model_path, scaler_path, epochs=10, lr=1e-3):
    # Load data
    df = pd.read_csv(data_path)
    data = df.filter(["Close"])
    data = data[1:].values

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    training_data_len = math.ceil(len(data) * 0.8)

    # Create the training data set
    train_data = scaled[:training_data_len, :]
    # Create the testing data set
    test_data = scaled[training_data_len - 60 :, :]

    # Create sequences
    X, y = create_sequences(train_data, seq_len=SEQUENCE_LENGTH)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Dataloader
    dataset = StockDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = LSTMModel().to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch).squeeze()
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"[{data_path}] Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}"
        )

    # Save
    save_model(model, scaler, model_path, scaler_path)
