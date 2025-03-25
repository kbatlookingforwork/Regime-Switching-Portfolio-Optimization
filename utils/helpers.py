import pandas as pd
import numpy as np

def load_asset_classes():
    """
    Load available asset classes for the application
    
    Returns:
    --------
    list
        List of available asset classes
    """
    return [
        'Bitcoin',
        'Gold',
        'S&P 500',
        'Real Estate',
        'Utilities',
        'Technology',
        'Energy',
        'Financials',
        'Healthcare',
        'Materials',
        'Industrials',
        'IHSG',
        'BBRI',
        'BBCA',
        'ASII'
    ]

def format_asset_returns_summary(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format asset returns summary for display
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame containing asset returns
        
    Returns:
    --------
    pd.DataFrame
        Formatted summary of asset returns
    """
    summary_dict = {}
    
    for col in returns_df.columns:
        summary_dict[col] = {
            'Annualized Return': returns_df[col].mean() * 252,
            'Annualized Volatility': returns_df[col].std() * np.sqrt(252),
            'Sharpe Ratio': returns_df[col].mean() / returns_df[col].std() * np.sqrt(252),
            'Max Drawdown': get_max_drawdown(returns_df[col]),
            'Skewness': returns_df[col].skew(),
            'Kurtosis': returns_df[col].kurtosis()
        }
    
    summary_df = pd.DataFrame(summary_dict).T
    summary_df = summary_df.round(4)
    
    return summary_df

def get_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown for a return series
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing returns
        
    Returns:
    --------
    float
        Maximum drawdown
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cum_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    return drawdown.min()

def calculate_rolling_sharpe(returns: pd.Series, window: int = 252, risk_free_rate: float = 0.02/252) -> pd.Series:
    """
    Calculate rolling Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing returns
    window : int
        Rolling window size
    risk_free_rate : float
        Daily risk-free rate
        
    Returns:
    --------
    pd.Series
        Series containing rolling Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    
    # Annualize
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
    
    return rolling_sharpe

def calculate_calmar_ratio(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Calculate rolling Calmar ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series containing returns
    window : int
        Rolling window size
        
    Returns:
    --------
    pd.Series
        Series containing rolling Calmar ratio
    """
    # Calculate rolling annualized return
    rolling_return = returns.rolling(window=window).mean() * 252
    
    # Calculate rolling maximum drawdown
    rolling_max_dd = returns.rolling(window=window).apply(
        lambda x: get_max_drawdown(x)
    )
    
    # Avoid division by zero
    rolling_max_dd = rolling_max_dd.replace(0, np.nan)
    
    # Calculate Calmar ratio
    calmar_ratio = rolling_return / abs(rolling_max_dd)
    
    return calmar_ratio

def create_portfolio_report(returns_df: pd.DataFrame, weights_df: pd.DataFrame, benchmark_returns: pd.Series = None) -> pd.DataFrame:
    """
    Create a comprehensive portfolio performance report
    
    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame containing asset returns
    weights_df : pd.DataFrame
        DataFrame containing portfolio weights
    benchmark_returns : pd.Series, optional
        Series containing benchmark returns
        
    Returns:
    --------
    pd.DataFrame
        Portfolio performance report
    """
    from modules.performance_metrics import PerformanceAnalyzer
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer()
    
    # Calculate portfolio returns
    portfolio_returns = analyzer.calculate_portfolio_returns(returns_df, weights_df)
    
    # Calculate performance metrics
    metrics = analyzer.calculate_performance_metrics(portfolio_returns, benchmark_returns)
    
    # Create report DataFrame
    report = pd.DataFrame({
        'Portfolio': metrics['portfolio'],
        'Benchmark': metrics.get('benchmark', {})
    })
    
    return report
