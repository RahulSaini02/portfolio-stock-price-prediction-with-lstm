import json
import yfinance as yf
import os


def build_datasets(config_file="portfolio.json", period="5y", interval="1d"):
    with open(config_file) as f:
        portfolio = json.load(f)

    for ticker, meta in portfolio.items():
        print(f"📥 Downloading {ticker}...")
        df = yf.download(ticker, period=period, interval=interval)
        df.reset_index(inplace=True)

        # Make sure the output path exists
        os.makedirs(os.path.dirname(meta["data_path"]), exist_ok=True)
        df.to_csv(meta["data_path"], index=False)
        print(f"✅ Saved to {meta['data_path']}")


if __name__ == "__main__":
    build_datasets()
