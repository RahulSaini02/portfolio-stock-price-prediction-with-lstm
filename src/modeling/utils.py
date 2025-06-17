import torch
import joblib
from src.config import DEVICE


def save_model(model, scaler, model_path, scaler_path):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_class, model_path, scaler_path, device: str = "cpu"):
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler
