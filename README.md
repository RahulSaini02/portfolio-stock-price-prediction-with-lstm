# 📊 Multi-Stock Portfolio Predictor

<a target="_blank" href="https://rahulsaini.click/">
    <img src="https://img.shields.io/badge/Multi Stock Portfolio-Predictor-B4CDED" alt="Stock Portfolio Price Prediction" />
</a>

**Predict the next-day stock prices of your portfolio using LSTM-based deep learning models.**
Built with Streamlit, PyTorch, and yFinance, this app allows users to:
- Select up to 4 stocks
- Train prediction models
- Visualize historical trends and forecasts
- Get allocation suggestions for smarter investments

---

## 🚀 Features

- 🔍 Fetch live data using `yfinance`
- 🧠 LSTM-based model training and evaluation
- 📈 Visualizations for historical and predicted prices
- 🥧 Suggested portfolio weights based on forecast
- 💾 Save and reload predictions to avoid retraining

---

## 🗂️ Project Structure

```
├── app/
│   ├── app.py                  # Streamlit main app
│   └── utils.py                # Plotting and utilities
├── data/                       # Raw and processed stock data
├── models/                     # Saved LSTM models and scalers
├── src/
│   ├── config.py
│   ├── model.py                # LSTM model definition
│   └── modeling/
│       ├── train.py            # Training logic
│       ├── predict.py          # Prediction logic
│       └── utils.py            # Model loading and helpers
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/RahulSaini02/portfolio-stock-price-prediction-with-lstm.git
cd stock-price-prediction-with-lstm
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app/app.py
```

---

## 🧠 Model

Each stock uses a sequence-based LSTM trained on the last 60 days of closing price data to predict the next day's price.

---

## 🧪 Example Prediction Output

```json
{
  "AAPL": {
    "last": 178.12,
    "pred": 181.45,
    "change": 1.87,
    "trained_on": "2025-06-14"
  }
}
```

---

## 📄 License

[MIT License](LICENSE)
