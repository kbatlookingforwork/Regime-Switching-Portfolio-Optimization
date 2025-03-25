import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import seaborn as sns

def plot_regime_transitions(prices_df: pd.DataFrame, regimes_df: pd.DataFrame):
    """
    Plot asset prices with regime transitions
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        DataFrame containing asset prices
    regimes_df : pd.DataFrame
        DataFrame containing regime labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use the first asset and market regime for visualization
    first_asset = prices_df.columns[0]
    asset_prices = prices_df[first_asset]
    regime_series = regimes_df['market']
    
    # Define colors for each regime
    regime_colors = {
        0: '#ffcccc',  # Light red for Bear Market
        1: '#ccffcc',  # Light green for Bull Market
        2: '#ffffcc',  # Light yellow for Bear-to-Bull Transition
        3: '#ffddaa'   # Light orange for Bull-to-Bear Transition
    }
    
    # Define labels for each regime
    regime_labels = {
        0: 'Bear Market',
        1: 'Bull Market',
        2: 'Bear-to-Bull Transition',
        3: 'Bull-to-Bear Transition'
    }
    
    # Plot price series
    ax.plot(asset_prices.index, asset_prices, 'k-', linewidth=1.5, label=first_asset)
    
    # Shade background according to regime
    prev_idx = asset_prices.index[0]
    prev_regime = regime_series.iloc[0]
    
    for idx, regime in zip(regime_series.index[1:], regime_series.iloc[1:]):
        if regime != prev_regime:
            # Shade the previous regime
            ax.axvspan(prev_idx, idx, alpha=0.3, color=regime_colors.get(prev_regime, 'white'))
            
            # Mark the transition with a vertical line
            ax.axvline(idx, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Update previous values
            prev_idx = idx
            prev_regime = regime
    
    # Shade the last regime
    ax.axvspan(prev_idx, asset_prices.index[-1], alpha=0.3, color=regime_colors.get(prev_regime, 'white'))
    
    # Create a custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors[regime], edgecolor='gray', alpha=0.3, label=label)
        for regime, label in regime_labels.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Set plot title and labels
    ax.set_title(f'Price Evolution with Market Regimes: {first_asset}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance: pd.Series, feature_names: list, top_n: int = 20):
    """
    Plot feature importance from the Random Forest model
    
    Parameters:
    -----------
    feature_importance : pd.Series
        Series containing feature importance scores
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Take top N features
    top_features = feature_importance.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar plot
    top_features.plot(kind='barh', ax=ax)
    
    # Set plot title and labels
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    return fig

def plot_portfolio_performance(portfolio_returns: pd.Series, benchmark_returns: pd.Series = None):
    """
    Plot portfolio cumulative performance vs. benchmark
    
    Parameters:
    -----------
    portfolio_returns : pd.Series
        Series containing portfolio returns
    benchmark_returns : pd.Series, optional
        Series containing benchmark returns
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Calculate cumulative returns
    portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot portfolio returns
    ax.plot(portfolio_cum_returns.index, portfolio_cum_returns, 'b-', linewidth=2, label='Portfolio')
    
    # Plot benchmark returns if provided
    if benchmark_returns is not None:
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        ax.plot(benchmark_cum_returns.index, benchmark_cum_returns, 'r--', linewidth=1.5, label='Benchmark')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Set plot title and labels
    ax.set_title('Cumulative Portfolio Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_regime_allocation_heatmap(weights_df: pd.DataFrame, regimes_df: pd.DataFrame):
    """
    Plot portfolio allocation heatmap by regime
    
    Parameters:
    -----------
    weights_df : pd.DataFrame
        DataFrame containing portfolio weights
    regimes_df : pd.DataFrame
        DataFrame containing regime labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Regime labels
    regime_labels = {
        0: 'Bear Market',
        1: 'Bull Market',
        2: 'Bear-to-Bull Transition',
        3: 'Bull-to-Bear Transition'
    }
    
    # Group weights by regime
    regimes = regimes_df['market'].reindex(weights_df.index)
    regime_weights = {}
    
    for regime in sorted(regimes.unique()):
        regime_mask = regimes == regime
        if sum(regime_mask) > 0:
            regime_weights[regime_labels.get(regime, f'Regime {regime}')] = weights_df.loc[regime_mask].mean()
    
    # Create DataFrame from dict
    regime_allocation_df = pd.DataFrame(regime_weights).T
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, len(regime_allocation_df) * 1.2))
    
    sns.heatmap(regime_allocation_df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=.5, ax=ax)
    
    # Set plot title and labels
    ax.set_title('Average Asset Allocation by Market Regime')
    ax.set_ylabel('Regime')
    ax.set_xlabel('Asset')
    
    plt.tight_layout()
    return fig
