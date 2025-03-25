import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional

class DataHandler:
    """
    Class for handling data operations, including fetching, cleaning, and feature engineering
    """
    
    def __init__(self):
        """Initialize the DataHandler class"""
        self.ticker_mapping = {
            'Bitcoin': 'BTC-USD',
            'Gold': 'GLD',
            'S&P 500': 'SPY',
            'Real Estate': 'VNQ',
            'Utilities': 'XLU',
            'Technology': 'XLK',
            'Energy': 'XLE',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Materials': 'XLB',
            'Industrials': 'XLI',
            'IHSG': '^JKSE',
            'BBRI': 'BBRI.JK',
            'BBCA': 'BBCA.JK',
            'ASII': 'ASII.JK'
        }

    def fetch_data(self, assets: List[str], start_date, end_date) -> pd.DataFrame:
        """
        Fetch historical price data for selected assets
        
        Parameters:
        -----------
        assets : List[str]
            List of asset names to fetch
        start_date : datetime.date
            Start date for data fetching
        end_date : datetime.date
            End date for data fetching
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing adjusted close prices for selected assets
        """
        tickers = [self.ticker_mapping.get(asset, asset) for asset in assets]
        
        # For debugging
        print(f"Fetching data for tickers: {tickers}")
        
        # Handle single ticker case
        if len(tickers) == 1:
            # Fetch data using yfinance for single ticker
            ticker_obj = yf.Ticker(tickers[0])
            data = ticker_obj.history(start=start_date, end=end_date)
            prices_df = pd.DataFrame({tickers[0]: data['Close']})
        else:
            # Fetch data using yfinance for multiple tickers
            data = yf.download(tickers, start=start_date, end=end_date)
            
            # Check if the data has a MultiIndex structure
            if isinstance(data.columns, pd.MultiIndex):
                # Try to extract the Close prices first, fallback to Adj Close if available
                if 'Close' in data.columns.levels[0]:
                    prices_df = data['Close']
                elif 'Adj Close' in data.columns.levels[0]:
                    prices_df = data['Adj Close']
                else:
                    # Fallback to the first available price column
                    prices_df = data[data.columns.levels[0][0]]
                    print(f"Warning: Using {data.columns.levels[0][0]} prices instead of Close/Adj Close")
            else:
                # Handle the case when there's no MultiIndex (usually happens with single ticker)
                if 'Close' in data.columns:
                    prices_df = pd.DataFrame({tickers[0]: data['Close']})
                elif 'Adj Close' in data.columns:
                    prices_df = pd.DataFrame({tickers[0]: data['Adj Close']})
                else:
                    raise ValueError(f"Neither 'Close' nor 'Adj Close' found in columns: {data.columns}")
        
        # Print the columns for debugging
        print(f"Data columns: {data.columns}")
        print(f"Prices DataFrame shape: {prices_df.shape}")
        
        # Map ticker symbols back to asset names
        reverse_mapping = {v: k for k, v in self.ticker_mapping.items()}
        
        # Handle the case when prices_df might be a Series (single asset)
        if isinstance(prices_df, pd.Series):
            prices_df = pd.DataFrame(prices_df)
            prices_df.columns = [reverse_mapping.get(tickers[0], tickers[0])]
        else:
            prices_df.columns = [reverse_mapping.get(col, col) for col in prices_df.columns]
        
        # Forward fill missing values (for non-trading days)
        prices_df = prices_df.ffill()
        
        # Drop any remaining NaN values
        prices_df = prices_df.dropna()
        
        return prices_df
    
    def calculate_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame containing price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing log returns
        """
        # Calculate log returns
        log_returns = np.log(prices_df) - np.log(prices_df.shift(1))
        
        # Drop first row with NaN values
        log_returns = log_returns.dropna()
        
        return log_returns
    
    def engineer_features(self, prices_df: pd.DataFrame, returns_df: pd.DataFrame, option: str = 'Basic') -> pd.DataFrame:
        """
        Engineer features for regime prediction
        
        Parameters:
        -----------
        prices_df : pd.DataFrame
            DataFrame containing price data
        returns_df : pd.DataFrame
            DataFrame containing returns data
        option : str
            Feature engineering complexity level ('Basic', 'Advanced', or 'Custom')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing engineered features
        """
        features = pd.DataFrame(index=returns_df.index)
        
        # Basic features
        if option in ['Basic', 'Advanced', 'Custom']:
            # Add rolling return features
            for window in [5, 10, 21, 63]:
                for col in returns_df.columns:
                    # Rolling mean returns
                    features[f'{col}_return_{window}d'] = returns_df[col].rolling(window=window).mean()
                    # Rolling volatility
                    features[f'{col}_vol_{window}d'] = returns_df[col].rolling(window=window).std()
            
            # Add price-based features
            for window in [21, 63, 126]:
                for col in prices_df.columns:
                    # Moving average
                    features[f'{col}_ma_{window}d'] = prices_df[col].rolling(window=window).mean() / prices_df[col] - 1
            
            # Add cross-asset correlation features
            if len(returns_df.columns) > 1:
                assets = returns_df.columns
                for i, asset1 in enumerate(assets):
                    for asset2 in assets[i+1:]:
                        # 30-day rolling correlation
                        rolling_corr = returns_df[asset1].rolling(window=30).corr(returns_df[asset2])
                        features[f'corr_{asset1}_{asset2}_30d'] = rolling_corr
        
        # Advanced features
        if option in ['Advanced', 'Custom']:
            # Add technical indicators
            for col in prices_df.columns:
                # RSI (Relative Strength Index)
                delta = prices_df[col].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                features[f'{col}_rsi_14d'] = 100 - (100 / (1 + rs))
                
                # MACD components
                ema12 = prices_df[col].ewm(span=12, adjust=False).mean()
                ema26 = prices_df[col].ewm(span=26, adjust=False).mean()
                features[f'{col}_macd'] = ema12 - ema26
                features[f'{col}_macd_signal'] = features[f'{col}_macd'].ewm(span=9, adjust=False).mean()
                
                # Bollinger Bands
                rolling_mean = prices_df[col].rolling(window=20).mean()
                rolling_std = prices_df[col].rolling(window=20).std()
                features[f'{col}_bb_upper'] = rolling_mean + (rolling_std * 2)
                features[f'{col}_bb_lower'] = rolling_mean - (rolling_std * 2)
                features[f'{col}_bb_position'] = (prices_df[col] - rolling_mean) / (2 * rolling_std)
            
            # Add market breadth indicators if we have equity indices
            equity_indices = ['S&P 500', 'IHSG']
            available_indices = [idx for idx in equity_indices if idx in prices_df.columns]
            
            if available_indices:
                # Use the first available index as a proxy for market returns
                market_col = available_indices[0]
                market_returns = returns_df[market_col]
                
                # Up/Down days ratio
                features['up_down_ratio_10d'] = (
                    market_returns.rolling(window=10).apply(lambda x: sum(x > 0) / sum(x < 0) if sum(x < 0) > 0 else sum(x > 0))
                )
                
                # Market momentum
                features['market_momentum'] = market_returns.rolling(window=20).sum()
        
        # Custom features
        if option == 'Custom':
            # Example of additional custom features
            # Regime transition probability estimation
            for col in returns_df.columns:
                # High volatility regime estimation using EWMA volatility
                ewma_vol = returns_df[col].ewm(span=21).std()
                vol_percentile = ewma_vol.rolling(window=252).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                features[f'{col}_vol_regime_prob'] = vol_percentile
                
                # Trend strength 
                trend_strength = abs(
                    prices_df[col].rolling(window=63).mean() / 
                    prices_df[col].rolling(window=252).mean() - 1
                )
                features[f'{col}_trend_strength'] = trend_strength
            
            # Economic indicators could be added here if data is available
            # For now, we'll add a few more complex features from the existing data
            
            # Cross-asset volatility ratios
            if len(returns_df.columns) > 1:
                assets = returns_df.columns
                for i, asset1 in enumerate(assets):
                    for asset2 in assets[i+1:]:
                        # Relative volatility
                        vol1 = returns_df[asset1].rolling(window=21).std()
                        vol2 = returns_df[asset2].rolling(window=21).std()
                        features[f'{asset1}_{asset2}_vol_ratio'] = vol1 / vol2
            
            # Asymmetric volatility features
            for col in returns_df.columns:
                up_returns = returns_df[col].copy()
                down_returns = returns_df[col].copy()
                
                up_returns[up_returns < 0] = 0
                down_returns[down_returns > 0] = 0
                
                up_vol = up_returns.rolling(window=21).std()
                down_vol = down_returns.abs().rolling(window=21).std()
                
                features[f'{col}_up_down_vol_ratio'] = up_vol / down_vol
        
        # Drop NaN values
        features = features.dropna()
        
        return features
