import pandas as pd
import numpy as np
from typing import Dict, Union

class PerformanceAnalyzer:
    """
    Class for calculating and analyzing portfolio performance metrics
    """
    
    def __init__(self):
        """Initialize the PerformanceAnalyzer class"""
        self.risk_free_rate = 0.02 / 252  # Daily risk-free rate (2% annual)
    
    def calculate_portfolio_returns(self, returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns over time
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame containing asset returns
        weights_df : pd.DataFrame
            DataFrame containing portfolio weights
            
        Returns:
        --------
        pd.Series
            Series containing portfolio returns
        """
        # Align the index of returns and weights
        aligned_returns = returns_df.reindex(weights_df.index)
        aligned_weights = weights_df.reindex(returns_df.index)
        
        # Fill any missing values with the previous weights
        aligned_weights = aligned_weights.ffill()
        
        # Calculate portfolio returns: weight * return for each asset, summed across assets
        # We need to shift weights by 1 period to avoid look-ahead bias
        portfolio_returns = (aligned_returns * aligned_weights.shift(1)).sum(axis=1)
        
        # The first entry will be NaN, drop it
        portfolio_returns = portfolio_returns.dropna()
        
        return portfolio_returns
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns
        
        Parameters:
        -----------
        returns : pd.Series
            Series containing returns
            
        Returns:
        --------
        pd.Series
            Series containing cumulative returns
        """
        return (1 + returns).cumprod() - 1
    
    def calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series
        
        Parameters:
        -----------
        returns : pd.Series
            Series containing returns
            
        Returns:
        --------
        pd.Series
            Series containing drawdowns
        """
        # Calculate cumulative returns
        cumulative_returns = self.calculate_cumulative_returns(returns)
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        
        return drawdown
    
    def calculate_performance_metrics(self, portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for portfolio and benchmark
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series containing portfolio returns
        benchmark_returns : pd.Series, optional
            Series containing benchmark returns
            
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Dictionary containing performance metrics for portfolio and benchmark
        """
        metrics = {
            'portfolio': {},
            'benchmark': {}
        }
        
        # Calculate portfolio metrics
        metrics['portfolio'] = self._calculate_metrics(portfolio_returns)
        
        # Calculate benchmark metrics if provided
        if benchmark_returns is not None:
            metrics['benchmark'] = self._calculate_metrics(benchmark_returns)
            
            # Calculate relative metrics
            metrics['portfolio']['Alpha'] = self._calculate_alpha(
                portfolio_returns, benchmark_returns
            )
            metrics['portfolio']['Beta'] = self._calculate_beta(
                portfolio_returns, benchmark_returns
            )
            metrics['portfolio']['Information Ratio'] = self._calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )
        
        return metrics
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics for a return series
        
        Parameters:
        -----------
        returns : pd.Series
            Series containing returns
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing performance metrics
        """
        metrics = {}
        
        # Annualized return
        metrics['Annualized Return'] = returns.mean() * 252
        
        # Annualized volatility
        metrics['Annualized Volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate
        metrics['Sharpe Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics['Sortino Ratio'] = excess_returns.mean() * 252 / downside_deviation
        else:
            metrics['Sortino Ratio'] = np.inf
        
        # Maximum drawdown
        drawdown = self.calculate_drawdown(returns)
        metrics['Maximum Drawdown'] = drawdown.min()
        
        # Calmar ratio
        if metrics['Maximum Drawdown'] < 0:
            metrics['Calmar Ratio'] = metrics['Annualized Return'] / abs(metrics['Maximum Drawdown'])
        else:
            metrics['Calmar Ratio'] = np.inf
        
        # Skewness
        metrics['Skewness'] = returns.skew()
        
        # Kurtosis
        metrics['Kurtosis'] = returns.kurtosis()
        
        # Value at Risk (95%)
        metrics['VaR (95%)'] = returns.quantile(0.05)
        
        # Conditional VaR (95%)
        metrics['CVaR (95%)'] = returns[returns <= metrics['VaR (95%)']].mean()
        
        # Positive periods
        metrics['Positive Periods'] = (returns > 0).sum() / len(returns)
        
        return metrics
    
    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Jensen's Alpha
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series containing portfolio returns
        benchmark_returns : pd.Series
            Series containing benchmark returns
            
        Returns:
        --------
        float
            Jensen's Alpha
        """
        beta = self._calculate_beta(portfolio_returns, benchmark_returns)
        
        # Annualized alpha
        alpha = (portfolio_returns.mean() - self.risk_free_rate) - beta * (benchmark_returns.mean() - self.risk_free_rate)
        
        return alpha * 252
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio Beta
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series containing portfolio returns
        benchmark_returns : pd.Series
            Series containing benchmark returns
            
        Returns:
        --------
        float
            Portfolio Beta
        """
        # Align indices
        aligned_returns = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        })
        aligned_returns = aligned_returns.dropna()
        
        # Calculate covariance and variance
        covariance = aligned_returns['portfolio'].cov(aligned_returns['benchmark'])
        variance = aligned_returns['benchmark'].var()
        
        if variance == 0:
            return 0
        
        return covariance / variance
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Series containing portfolio returns
        benchmark_returns : pd.Series
            Series containing benchmark returns
            
        Returns:
        --------
        float
            Information Ratio
        """
        # Align indices
        aligned_returns = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        })
        aligned_returns = aligned_returns.dropna()
        
        # Calculate tracking error
        tracking_error = (aligned_returns['portfolio'] - aligned_returns['benchmark']).std()
        
        if tracking_error == 0:
            return 0
        
        # Calculate information ratio
        active_return = aligned_returns['portfolio'].mean() - aligned_returns['benchmark'].mean()
        
        return active_return / tracking_error * np.sqrt(252)
