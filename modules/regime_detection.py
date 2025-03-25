import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import statsmodels.api as sm

class RegimeDetector:
    """
    Class for market regime detection using KAMA+MSR (Kaufman's Adaptive Moving Average + Markov-Switching Regression)
    """
    
    def __init__(self, kama_period: int = 21, ms_states: int = 2):
        """
        Initialize the RegimeDetector class
        
        Parameters:
        -----------
        kama_period : int
            Period for Kaufman's Adaptive Moving Average
        ms_states : int
            Number of states for Markov-Switching Regression model
        """
        self.kama_period = kama_period
        self.ms_states = ms_states
        self.regime_names = {
            0: 'Bear Market',
            1: 'Bull Market',
            2: 'Bear-to-Bull Transition',
            3: 'Bull-to-Bear Transition'
        }
        self.models = {}
        
    def calculate_kama(self, series: pd.Series) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average for a time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series data for which to calculate KAMA
            
        Returns:
        --------
        pd.Series
            Calculated KAMA values
        """
        er_period = 10
        fast_alpha = 2/(2+1)
        slow_alpha = 2/(30+1)
        
        # Calculate absolute price change (direction)
        change = abs(series.diff())
        
        # Calculate volatility (noise)
        volatility = change.rolling(window=er_period).sum()
        
        # Avoid division by zero
        volatility = volatility.replace(0, np.nan)
        
        # Calculate efficiency ratio
        er = change / volatility
        
        # Calculate smoothing constant
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Calculate KAMA
        kama = pd.Series(index=series.index, dtype='float64')
        
        # Set first value
        kama.iloc[0] = series.iloc[0]
        
        # Calculate KAMA values
        for i in range(1, len(series)):
            if pd.notna(sc.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
            else:
                kama.iloc[i] = kama.iloc[i-1]
        
        return kama
    
    def fit_markov_model(self, returns: pd.Series) -> MarkovRegression:
        """
        Fit a Markov-Switching Regression model to returns data
        
        Parameters:
        -----------
        returns : pd.Series
            Asset returns for which to fit the model
            
        Returns:
        --------
        MarkovRegression
            Fitted Markov-Switching Regression model
        """
        # Add a constant for the regression
        X = sm.add_constant(np.ones_like(returns))
        
        # Fit the model
        model = MarkovRegression(
            returns.values, 
            k_regimes=self.ms_states, 
            trend='c', 
            switching_variance=True
        )
        result = model.fit()
        
        return result
    
    def detect_regimes(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes using KAMA+MSR approach
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame containing asset returns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing detected regimes for each asset
        """
        regimes_df = pd.DataFrame(index=returns_df.index)
        
        for col in returns_df.columns:
            print(f"Detecting regimes for {col}...")
            
            # Step 1: Calculate KAMA
            kama = self.calculate_kama(returns_df[col].cumsum())
            
            # Step 2: Calculate KAMA returns
            kama_returns = kama.pct_change().dropna()
            
            # Step 3: Fit Markov-Switching Regression model to KAMA returns
            try:
                markov_model = self.fit_markov_model(kama_returns)
                self.models[col] = markov_model
                
                # Step 4: Extract smoothed probabilities
                smoothed_probs = markov_model.smoothed_marginal_probabilities
                
                # Print lengths for debugging
                print(f"Length of returns_df: {len(returns_df)}")
                print(f"Length of kama: {len(kama)}")
                print(f"Length of kama_returns: {len(kama_returns)}")
                print(f"Length of smoothed_probs: {len(smoothed_probs)}")
                
                # Step 5: Determine the primary regime (state with highest probability)
                if self.ms_states == 2:
                    # For 2-state model: classify as 0 (bear) or 1 (bull)
                    primary_regime = smoothed_probs[:, 1] > 0.5
                    
                    # Also detect transition regimes
                    # Transitions occur when the probability is close to 0.5 (uncertainty)
                    transition_threshold = 0.3
                    
                    # Convert numpy array to pandas Series for using diff()
                    prob_series = pd.Series(smoothed_probs[:, 1])
                    prob_diff = prob_series.diff()
                    
                    bear_to_bull = (smoothed_probs[:, 1] > 0.5 - transition_threshold) & (smoothed_probs[:, 1] < 0.5 + transition_threshold) & (prob_diff > 0)
                    bull_to_bear = (smoothed_probs[:, 1] > 0.5 - transition_threshold) & (smoothed_probs[:, 1] < 0.5 + transition_threshold) & (prob_diff < 0)
                    
                    # Create a Series with proper index
                    regime_series = pd.Series(np.zeros(len(kama_returns)), index=kama_returns.index)
                    
                    # Set regimes in the Series
                    regime_series[primary_regime] = 1  # Bull market
                    regime_series[bear_to_bull] = 2    # Bear-to-Bull Transition
                    regime_series[bull_to_bear] = 3    # Bull-to-Bear Transition
                    
                    # Reindex to match returns_df
                    # Use forward fill to propagate regimes to dates not in the model
                    full_regime_series = regime_series.reindex(returns_df.index, method='ffill')
                    
                    # Fill any initial NaN values with 0 (bear market)
                    full_regime_series = full_regime_series.fillna(0)
                    
                    # Assign to regimes_df
                    regimes_df[col] = full_regime_series
                else:
                    # For models with more than 2 states: classify based on highest probability
                    primary_regime = np.argmax(smoothed_probs, axis=1)
                    
                    # Create a Series with proper index
                    regime_series = pd.Series(primary_regime, index=kama_returns.index)
                    
                    # Reindex to match returns_df
                    full_regime_series = regime_series.reindex(returns_df.index, method='ffill').fillna(0)
                    
                    # Assign to regimes_df
                    regimes_df[col] = full_regime_series
            except Exception as e:
                print(f"Error detecting regimes for {col}: {str(e)}")
                print(f"Using simple volatility-based regime detection as fallback")
                
                # Fallback to a simple volatility-based regime detection
                rolling_vol = returns_df[col].rolling(window=21).std() * np.sqrt(252)
                high_vol_threshold = rolling_vol.quantile(0.7)
                low_vol_threshold = rolling_vol.quantile(0.3)
                
                regimes = np.ones(len(returns_df))  # Default: normal market (1)
                regimes[rolling_vol > high_vol_threshold] = 0  # High volatility: bear market (0)
                regimes[rolling_vol < low_vol_threshold] = 1  # Low volatility: bull market (1)
                
                # Add transitions
                regimes_shifted = np.roll(regimes, 1)  # Shift by 1 position
                regimes_shifted[0] = regimes[0]  # Set first value to avoid artifacts
                
                bear_to_bull = (regimes_shifted == 0) & (regimes == 1)
                bull_to_bear = (regimes_shifted == 1) & (regimes == 0)
                
                regimes[bear_to_bull] = 2  # Bear-to-Bull Transition
                regimes[bull_to_bear] = 3  # Bull-to-Bear Transition
                
                regimes_df[col] = regimes
        
        # Calculate a "market regime" as the mode of all asset regimes
        # This assumes all assets have the same regime structure
        regimes_df['market'] = regimes_df.mode(axis=1)[0]
        
        return regimes_df
    
    def get_regime_statistics(self, returns_df: pd.DataFrame, regimes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each detected regime
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame containing asset returns
        regimes_df : pd.DataFrame
            DataFrame containing detected regimes
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing statistics for each regime
        """
        stats = []
        
        # Use the market regime for statistics
        market_regime = regimes_df['market']
        
        for regime in sorted(market_regime.unique()):
            regime_mask = market_regime == regime
            regime_stats = {}
            
            # Skip regimes with very few observations
            if sum(regime_mask) < 5:
                continue
                
            regime_stats['Regime'] = self.regime_names.get(regime, f'Regime {regime}')
            regime_stats['Days'] = sum(regime_mask)
            regime_stats['Frequency'] = sum(regime_mask) / len(market_regime)
            
            # Calculate return statistics for each asset in this regime
            for col in returns_df.columns:
                regime_returns = returns_df.loc[regime_mask, col]
                
                regime_stats[f'{col} Avg. Return'] = regime_returns.mean() * 252  # Annualized
                regime_stats[f'{col} Volatility'] = regime_returns.std() * np.sqrt(252)  # Annualized
                regime_stats[f'{col} Sharpe'] = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                regime_stats[f'{col} Skew'] = regime_returns.skew()
                regime_stats[f'{col} Kurtosis'] = regime_returns.kurtosis()
            
            stats.append(regime_stats)
        
        return pd.DataFrame(stats)
    
    def get_current_regime(self, regimes_df: pd.DataFrame) -> dict:
        """
        Get the current market regime
        
        Parameters:
        -----------
        regimes_df : pd.DataFrame
            DataFrame containing detected regimes
            
        Returns:
        --------
        dict
            Dictionary containing current regime information
        """
        current_regimes = regimes_df.iloc[-1].to_dict()
        
        # Convert numeric regimes to names
        named_regimes = {}
        for asset, regime in current_regimes.items():
            named_regimes[asset] = self.regime_names.get(regime, f'Regime {regime}')
        
        return named_regimes
