import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Generates technical indicators and time-based features from OHLCV data.
    """
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical and time features to the dataframe."""
        df = df.copy()
        
        # 1. Returns and Log Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Volatility (Rolling Standard Deviation) - 1 hour and 24 hours
        # 5 minute intervals: 1h = 12 steps, 24h = 288 steps
        df['volatility_1h'] = df['returns'].rolling(window=12).std()
        df['volatility_24h'] = df['returns'].rolling(window=288).std()
        
        # 3. RSI (Relative Strength Index) - 14 periods (14 * 5m = 70m)
        df['rsi'] = self._calculate_rsi(df['close'], window=14)
        
        # 4. MACD (Moving Average Convergence Divergence)
        # Fast=12, Slow=26, Signal=9
        macd, signal, hist = self._calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # 5. Bollinger Bands (20 periods, 2 std dev)
        upper, middle, lower = self._calculate_bollinger_bands(df['close'], window=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # 6. Time Features
        # Sin/Cos transformation for cyclical time features
        # Hour of day (0-23)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week (0-6)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 7. Volume Change
        df['volume_change'] = df['volume'].pct_change()
        
        # Fill NaNs created by rolling windows
        df.fillna(0, inplace=True)
        
        # Drop initial rows where indicators might be unstable if preferred, 
        # but for now we fill 0 or backward fill. 
        # Better to drop if training, but for inference we might need to be careful.
        # We'll drop the first 288 rows (24h) in usage or just handle them.
        
        return df

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: int = 2):
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

if __name__ == "__main__":
    # Test feature generation
    # Mock data or import fetcher
    import sys
    import os
    sys.path.append(os.getcwd())
    
    from prediction.data.data_fetcher import BinanceDataFetcher
    
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_historical_data("BTC", limit=500)
    
    engineer = FeatureEngineer()
    df_features = engineer.add_all_features(df)
    
    print("\n--- Features Generated ---")
    print(df_features[['close', 'rsi', 'macd', 'volatility_1h', 'hour_sin']].tail())
