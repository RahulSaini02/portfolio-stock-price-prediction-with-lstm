import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, in_features=1, hidden_units=50, out_features=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_units,
            num_layers=2,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc2 = nn.Linear(hidden_units // 2, out_features)
        self.hidden_units = hidden_units

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_units).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_units).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        return self.fc2(out)
