from src.utils import resolve_root

resolve_root(2)

import json
from src.modeling.train import train_stock_lstm


with open("portfolio.json") as f:
    portfolio = json.load(f)

for ticker, config in portfolio.items():
    print(f"🔧 Training model for {ticker}")
    print(config)
    train_stock_lstm(
        data_path=config["data_path"],
        model_path=config["model_path"],
        scaler_path=config["scaler_path"],
        epochs=15,
    )
