import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple

class DataFetcher:
    """
    Fetches historical price data from CoinGecko API.
    Handles rate limiting and data formatting.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Map symbols to CoinGecko IDs
    ASSET_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XAU": "tether-gold" # Using Tether Gold as proxy for XAU on CoinGecko if XAU not available directly as a crypto, or pax-gold. 
                             # Real XAU data might strictly require a different commodity API, but for "coin" prediction, PAXG or XAUT is standard on-chain.
                             # Let's standardize on 'bitcoin' etc.
    }
    
    def __init__(self):
        self.session = requests.Session()
        # Basic rate limiting: free tier allows ~10-30 calls/min
        self.last_call_time = 0
        self.min_interval = 2.0  # seconds between calls safety buffer

    def _wait_for_rate_limit(self):
        """Ensure we don't hit rate limits"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()

    def fetch_historical_data(self, asset: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch OHLCV data for the specified asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            days: Number of days of history to fetch (1, 7, 14, 30, 90, 180, 365, max)
        
        Returns:
            pd.DataFrame with columns [timestamp, open, high, low, close]
            indexed by timestamp.
        """
        if asset not in self.ASSET_MAP:
            raise ValueError(f"Unsupported asset: {asset}")
            
        coin_id = self.ASSET_MAP[asset]
        
        # CoinGecko /coins/{id}/ohlc endpoint
        # vs allows 'usd'
        # days options: 1, 7, 14, 30, 90, 180, 365, max
        url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
        params = {
            "vs_currency": "usd",
            "days": days
        }
        
        print(f"Fetching data for {asset} ({coin_id})...")
        
        try:
            self._wait_for_rate_limit()
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"Warning: No data returned for {asset}")
                return pd.DataFrame()

            # CoinGecko returns [time, open, high, low, close]
            # Time is ms timestamp
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            
            # Sort just in case
            df.sort_index(inplace=True)
            
            # Add volume? OHLC endpoint doesn't always provide volume consistently on free tier for this specific call layout, 
            # but let's check market_chart endpoint if we strictly need volume.
            # market_chart gives prices, market_caps, total_volumes arrays.
            # For Phase 1 simplified, OHLC is critical. Volume is "nice to have" for features.
            # Let's switch to market_chart range if we need volume, or stick to OHLC.
            # market_chart is better for granularity control sometimes.
            # OHLC auto-adjusts granularity based on 'days'.
            # days=90 -> 4-day granularity? No, usually 4-hourly or daily.
            # For 5-minute predictions, we need high resolution. 
            # CoinGecko Free API has limits on granularity. 
            # < 1 day = 30 min?
            # actually max resolution is usually hourly on free tier for > 90 days.
            # For 5 minute intervals, we really need the last 24h to be accurate, 
            # or we need a better data source like Binance public API for candle data (klines).
            
            # IMPORTANT: For 5-minute interval predictions, CoinGecko free might be too coarse (hourly/daily for long ranges).
            # Binance offers 5m candles freely.
            
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limit hit. Waiting...")
                time.sleep(60) 
                return self.fetch_historical_data(asset, days)
            else:
                print(f"Error fetching data: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

class BinanceDataFetcher:
    """
    Alternative specific for Crypto to get higher resolution (5-min candles).
    No API key needed for public data usually.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    ASSET_MAP = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XAU": "PAXGUSDT" # Standardize XAU as PAX Gold (gold-backed token)
    }
    
    def fetch_historical_data(self, asset: str, interval: str = "5m", limit: int = 1000) -> pd.DataFrame:
        """
        Fetch klines from Binance.
        interval: 1m, 3m, 5m, 15m, 30m, 1h, etc.
        """
        symbol = self.ASSET_MAP.get(asset)
        if not symbol:
             # Fallback or error
             if asset == "XAU": 
                 # Binance might not have XAU direct, maybe PAXGUSDT
                 symbol = "PAXGUSDT"
             else:
                 raise ValueError(f"Unknown asset for Binance: {asset}")

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit # Max 1000
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            
            # [
            #   [
            #     1499040000000,      // Open time
            #     "0.01634790",       // Open
            #     "0.80000000",       // High
            #     "0.01575800",       // Low
            #     "0.01577100",       // Close
            #     "148976.11427815",  // Volume
            #     ...
            #   ]
            # ]
            
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
                
            df.set_index("timestamp", inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching from Binance: {e}")
            return pd.DataFrame()

class SimulationDataFetcher:
    """
    Generates synthetic OHLCV data using Geometric Brownian Motion.
    Useful for development when API is unavailable.
    """
    
    def fetch_historical_data(self, asset: str, limit: int = 1000) -> pd.DataFrame:
        print(f"Generating synthetic data for {asset}...")
        
        # Parameters
        if asset == "BTC":
            start_price = 100000.0
            volatility = 0.02
        elif asset == "ETH":
            start_price = 4000.0
            volatility = 0.03
        elif asset == "SOL":
            start_price = 150.0
            volatility = 0.04
        else: # XAU
            start_price = 2500.0
            volatility = 0.01

        # Time index (5 min intervals, ending now)
        end_time = datetime.now(timezone.utc)
        # start_time is calculated from end, periods and freq by pandas
        timestamps = pd.date_range(end=end_time, freq="5min", periods=limit)
        
        # Generate random returns
        # GBM: P_t = P_{t-1} * exp((mu - sigma^2/2)dt + sigma * sqrt(dt) * Z)
        # Simplified: random walk
        dt = 5 / (24 * 60) # 5 mins in days
        sigma = volatility
        mu = 0.0001
        
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), limit)
        price_path = start_price * np.cumprod(1 + returns)
        
        # Create OHLC
        # Open = Previous Close (approx)
        # Close = price_path
        # High = max(Open, Close) + noise
        # Low = min(Open, Close) - noise
        
        opens = np.roll(price_path, 1)
        opens[0] = start_price
        
        closes = price_path
        
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.001, limit)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.001, limit)))
        
        volumes = np.random.lognormal(10, 1, limit) * (start_price / 1000)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        df.set_index("timestamp", inplace=True)
        return df

if __name__ == "__main__":
    import numpy as np # Needed for simulation
    
    # Try fetching real data, fall back to synthetic
    try:
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_historical_data("BTC", limit=50)
        if df.empty:
            raise Exception("Empty dataframe returned")
        print("\n--- BTC Real Data (Binance) ---")
    except Exception as e:
        print(f"API failed ({e}), using synthetic data.")
        fetcher = SimulationDataFetcher()
        df = fetcher.fetch_historical_data("BTC", limit=50)
        print("\n--- BTC Synthetic Data ---")
        
    print(df.tail())

