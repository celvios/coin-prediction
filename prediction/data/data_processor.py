import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import our modules
try:
    from prediction.data.data_fetcher import BinanceDataFetcher, SimulationDataFetcher
    from prediction.data.feature_engineering import FeatureEngineer
except ImportError:
    # Handle relative imports for script usage
    import sys
    import os
    sys.path.append(os.getcwd())
    from prediction.data.data_fetcher import BinanceDataFetcher, SimulationDataFetcher
    from prediction.data.feature_engineering import FeatureEngineer

class DataProcessor:
    """
    Orchestrates data fetching, feature engineering, and preprocessing.
    Prepares data for model training (normalization, sequencing).
    """
    
    def __init__(self, asset: str, sequence_length: int = 288, prediction_horizon: int = 288):
        self.asset = asset
        self.sequence_length = sequence_length # Input window (e.g. 24h)
        self.prediction_horizon = prediction_horizon # Output window (e.g. 24h)
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.engineer = FeatureEngineer()
        
    def get_raw_data(self, limit: int = 5000) -> pd.DataFrame:
        """Fetch raw OHLCV data."""
        try:
            fetcher = BinanceDataFetcher()
            df = fetcher.fetch_historical_data(self.asset, limit=limit)
            if df.empty:
                raise Exception("API returned empty data")
        except Exception as e:
            print(f"Using synthetic data for {self.asset} due to: {e}")
            fetcher = SimulationDataFetcher()
            df = fetcher.fetch_historical_data(self.asset, limit=limit)
        return df

    def prepare_data(self, limit: int = 5000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline: Fetch -> Feature Engineering -> Normalize -> Sequence
        Returns: X_train, y_train, X_val, y_val
        """
        # 1. Fetch
        df = self.get_raw_data(limit)
        
        # 2. Features
        df = self.engineer.add_all_features(df)
        
        # Drop NaNs
        df.dropna(inplace=True)
        
        # 3. Select Features
        feature_cols = [
            'close', 'returns', 'log_returns', 'volatility_1h', 'volatility_24h',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'volume_change'
        ]
        
        target_col = 'close'
        
        data = df[feature_cols].values
        targets = df[[target_col]].values # Keep simple 2D for scaling
        
        # 4. Split (80% train, 20% val)
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        train_targets = targets[:train_size]
        val_data = data[train_size:]
        val_targets = targets[train_size:]
        
        # 5. Normalize (Fit on train ONLY to avoid leakage)
        self.feature_scaler.fit(train_data)
        self.target_scaler.fit(train_targets)
        
        X_train_scaled = self.feature_scaler.transform(train_data)
        y_train_scaled = self.target_scaler.transform(train_targets)
        
        X_val_scaled = self.feature_scaler.transform(val_data)
        y_val_scaled = self.target_scaler.transform(val_targets)
        
        # 6. Create Sequences
        X_train, y_train = self._create_sequences(X_train_scaled, y_train_scaled)
        X_val, y_val = self._create_sequences(X_val_scaled, y_val_scaled)
        
        return (
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train), 
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )

    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences.
        Inputs: [t - seq_len : t]
        Targets: [t : t + horizon]
        """
        Xs, ys = [], []
        total_len = len(data)
        
        for i in range(total_len - self.sequence_length - self.prediction_horizon + 1):
            x = data[i : i + self.sequence_length]
            y = targets[i + self.sequence_length : i + self.sequence_length + self.prediction_horizon]
            Xs.append(x)
            ys.append(y)
            
        return np.array(Xs), np.array(ys)
    
    def inverse_transform_targets(self, targets_scaled: torch.Tensor) -> torch.Tensor:
        """Convert scaled predictions back to prices."""
        # Handle 3D tensor [batch, horizon, 1] -> [batch * horizon, 1] -> transform -> reshape
        shape = targets_scaled.shape
        flat = targets_scaled.reshape(-1, 1).cpu().numpy()
        inverse = self.target_scaler.inverse_transform(flat)
        return torch.FloatTensor(inverse).reshape(shape)

if __name__ == "__main__":
    # Test Processor
    processor = DataProcessor("BTC")
    X_train, y_train, X_val, y_val = processor.prepare_data(limit=1000)
    
    print("\n--- Data Processor Output ---")
    print(f"X_train shape: {X_train.shape} (Batch, SeqLen, Features)")
    print(f"y_train shape: {y_train.shape} (Batch, Horizon, Targets)")
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")
    
    # Check normalization stats
    print(f"Train Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
