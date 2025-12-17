# Cryptocurrency Price Prediction System

A production-ready AI system for generating probabilistic price forecasts (1000 simulated paths) for cryptocurrencies (BTC, ETH, SOL, XAU) over a 24-hour horizon.

## 🚀 Features

- **state-of-the-art AI Model**: Enhanced BiLSTM with Attention mechanisms (In Progress).
- **Probabilistic Forecasting**: Generates 7 quantiles (P1-P99) to sample 1000 realistic price paths.
- **Multi-Horizon**: Direct prediction of 288 time steps (24 hours at 5-min intervals).
- **Robust Data Pipeline**: 
    - Fetches real-time 5-min candles from **Binance API**.
    - **Synthetic Data Fallback**: Automatically switches to geometric brownian motion simulation if API is unreachable.
    - **Advanced Feature Engineering**: RSI, MACD, Bollinger Bands, Volatility, Cyclical Time encoding.

## 📂 Project Structure

```
coin-prediction/
├── prediction/
│   ├── data/
│   │   ├── data_fetcher.py          # Data collection (Binance + Synthetic)
│   │   ├── feature_engineering.py   # Technical indicators
│   │   └── data_processor.py        # Normalization & Sequence creation
│   ├── models/                      # (Phase 2)
│   └── visualization/               # (Phase 5)
├── api/                             # (Phase 6)
├── tests/                           # (Phase 7)
└── requirements.txt
```

## 🛠️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/celvios/coin-prediction.git
   cd coin-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # For GPU support (optional):
   # pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

## 📊 Phase 1: Data Pipeline Usage

Phase 1 (Completed) provides the foundational data infrastructure. You can use it to fetch and prepare data for model training.

### 1. Fetching Data
The `DataFetcher` automatically handles API rate limits and switches to synthetic data if needed.

```python
from prediction.data.data_fetcher import BinanceDataFetcher

fetcher = BinanceDataFetcher()
# Fetch last 1000 5-min candles for BTC
df = fetcher.fetch_historical_data("BTC", limit=1000)
print(df.tail())
```

### 2. Processing Data for Training
The `DataProcessor` handles the full pipeline: Fetch -> Feature Engineer -> Normalize -> Sequence.

```python
from prediction.data.data_processor import DataProcessor

# Prepare data for BTC (24h sequence length, 24h prediction horizon)
processor = DataProcessor(asset="BTC", sequence_length=288, prediction_horizon=288)

# Returns PyTorch tensors ready for training
X_train, y_train, X_val, y_val = processor.prepare_data(limit=5000)

print(f"Training Data Shape: {X_train.shape}") 
# Output: (Batch_Size, 288, Num_Features)
```

## 📅 Roadmap

- [x] **Phase 1: Data Pipeline** (Completed) - Real & Synthetic data, Feature Engineering.
- [ ] **Phase 2: Model Development** - Enhanced BiLSTM architecture.
- [ ] **Phase 3: Path Generation** - Probabilistic sampling.
- [ ] **Phase 4: Multi-Asset** - Correlation modeling.
- [ ] **Phase 5: Visualization** - Interactive dashboard.
- [ ] **Phase 6: API Layer** - FastAPI service.
- [ ] **Phase 7: Testing** - Unit & Backtesting.

## 🤝 Contributing

This project is a private development for AI-driven financial forecasting.
