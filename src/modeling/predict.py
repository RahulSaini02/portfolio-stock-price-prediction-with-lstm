import torch
import numpy as np
from src.config import SEQUENCE_LENGTH, DEVICE


def predict_price(model, scaler, data, device: str = "cpu"):
    data = data[["Close"]].values
    data = data[1:]
    scaled_data = scaler.transform(data)
    sequence = np.array([scaled_data[-SEQUENCE_LENGTH:]]).reshape(1, SEQUENCE_LENGTH, 1)
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(sequence_tensor).item()

    return float(scaler.inverse_transform([[pred_scaled]])[0][0])
