import json
import pandas as pd
from src.model import LSTMModel
from src.modeling.predict import predict_price
from src.modeling.utils import load_model
from src.config import DEVICE
from datetime import datetime
import os


def generate_signals(config_file="portfolio.json", output_dir="reports/portfolio"):
    with open(config_file) as f:
        portfolio = json.load(f)

    results = {}

    for ticker, meta in portfolio.items():
        df = pd.read_csv(meta["data_path"])
        model, scaler = load_model(
            model_class=LSTMModel,
            model_path=meta["model_path"],
            scaler_path=meta["scaler_path"],
            device=DEVICE,
        )
        last_price = float(df["Close"].iloc[-1])
        predicted_price = float(predict_price(model, scaler, df, DEVICE))

        change_pct = (predicted_price - last_price) / last_price * 100
        signal = "Rise" if change_pct > 0 else "Drop"

        results[ticker] = {
            "Last Price": round(last_price, 2),
            "Predicted Price": round(predicted_price, 2),
            "Change (%)": round(change_pct, 2),
            "Signal": signal,
            "Recommended Weight": meta["weight"],
        }

    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Save JSON
    json_path = os.path.join(output_dir, f"portfolio_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    df = pd.DataFrame.from_dict(results, orient="index")
    csv_path = os.path.join(output_dir, f"portfolio_results_{timestamp}.csv")
    df.to_csv(csv_path)

    print(f"\n✅ Results saved to:\n→ {json_path}\n→ {csv_path}\n")

    return results
