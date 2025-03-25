import pandas as pd
import numpy as np
import cvxpy as cp
from typing import List, Dict, Tuple

class PortfolioOptimizer:
    """
    Class for portfolio optimization using Model Predictive Control (MPC)
    with regime-switching considerations
    """
    
    def __init__(self, risk_aversion: float = 5.0, transaction_cost: float = 0.001, 
                 max_allocation: float = 0.4):
        """
        Initialize the PortfolioOptimizer class
        
        Parameters:
        -----------
        risk_aversion : float
            Risk aversion parameter (higher values = more risk-averse)
        transaction_cost : float
            Transaction cost as a fraction of traded amount
        max_allocation : float
            Maximum allocation to any single asset (0.4 = 40%)
        """
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        self.max_allocation = max_allocation
        
        # Regime-specific parameters
        self.regime_risk_multipliers = {
            0: 2.0,    # Bear Market - increase risk aversion
            1: 0.8,    # Bull Market - decrease risk aversion
            2: 1.0,    # Bear-to-Bull Transition - neutral
            3: 1.5     # Bull-to-Bear Transition - moderate increase in risk aversion
        }
    
    def estimate_moments(self, returns_df: pd.DataFrame, regimes_df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate mean returns and covariance matrices for each regime
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame containing asset returns
        regimes_df : pd.DataFrame
            DataFrame containing regimes
            
        Returns:
        --------
        Dict[int, Tuple[np.ndarray, np.ndarray]]
            Dictionary mapping regime to (mean, covariance) tuples
        """
        moments = {}
        
        # For each regime, estimate the mean and covariance
        market_regime = regimes_df['market']
        
        for regime in sorted(market_regime.unique()):
            regime_mask = market_regime == regime
            
            # Skip regimes with too few observations
            if sum(regime_mask) < 10:
                continue
                
            regime_returns = returns_df.loc[regime_mask]
            
            # Estimate annualized mean returns (252 trading days)
            mean_returns = regime_returns.mean() * 252
            
            # Estimate annualized covariance matrix
            cov_matrix = regime_returns.cov() * 252
            
            moments[regime] = (mean_returns.values, cov_matrix.values)
        
        # If a regime has too few observations, use the overall estimates
        overall_mean = returns_df.mean() * 252
        overall_cov = returns_df.cov() * 252
        
        for regime in range(4):  # Assuming 4 possible regimes
            if regime not in moments:
                moments[regime] = (overall_mean.values, overall_cov.values)
        
        return moments
    
    def optimize_single_period(self, current_weights: np.ndarray, 
                               expected_returns: np.ndarray, 
                               covariance_matrix: np.ndarray,
                               current_regime: int) -> np.ndarray:
        """
        Optimize portfolio weights for a single period
        
        Parameters:
        -----------
        current_weights : np.ndarray
            Current portfolio weights
        expected_returns : np.ndarray
            Expected returns for each asset
        covariance_matrix : np.ndarray
            Covariance matrix of returns
        current_regime : int
            Current market regime
            
        Returns:
        --------
        np.ndarray
            Optimized portfolio weights
        """
        n_assets = len(current_weights)
        
        # Adjust risk aversion based on regime
        regime_risk_aversion = self.risk_aversion * self.regime_risk_multipliers.get(current_regime, 1.0)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Define objective function
        returns = weights @ expected_returns
        risk = cp.quad_form(weights, covariance_matrix)
        trade_size = cp.sum(cp.abs(weights - current_weights))
        transaction_cost = self.transaction_cost * trade_size
        
        # Mean-variance utility with transaction costs
        objective = cp.Maximize(returns - regime_risk_aversion * risk - transaction_cost)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Fully invested
            weights >= 0,         # Long-only constraint
            weights <= self.max_allocation  # Maximum allocation constraint
        ]
        
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Check if the problem was solved successfully
        if problem.status != 'optimal':
            # If optimization fails, return the current weights
            return current_weights
        
        return weights.value
    
    def optimize_portfolio(self, returns_df: pd.DataFrame, regimes_df: pd.DataFrame, 
                           predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize portfolio weights over time using regime-switching MPC
        
        Parameters:
        -----------
        returns_df : pd.DataFrame
            DataFrame containing asset returns
        regimes_df : pd.DataFrame
            DataFrame containing historical regimes
        predictions_df : pd.DataFrame
            DataFrame containing regime predictions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing optimized portfolio weights over time
        """
        # Estimate moments for each regime
        regime_moments = self.estimate_moments(returns_df, regimes_df)
        
        # Initialize with equal weights
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        # DataFrame to store weights over time
        weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        
        # Initial weights
        weights_df.iloc[0] = weights
        
        # Loop through time periods
        for t in range(1, len(returns_df)):
            if t < len(predictions_df):
                # Get the predicted regime for this time period
                predicted_regime = predictions_df.iloc[t]['predicted_regime']
                
                # Get the moments for the predicted regime
                expected_returns, covariance_matrix = regime_moments[predicted_regime]
                
                # Current weights are the previous period's weights
                current_weights = weights_df.iloc[t-1].values
                
                # Optimize for this period
                new_weights = self.optimize_single_period(
                    current_weights, 
                    expected_returns, 
                    covariance_matrix,
                    predicted_regime
                )
                
                # Store the new weights
                weights_df.iloc[t] = new_weights
            else:
                # If we don't have predictions for this period, use the last weights
                weights_df.iloc[t] = weights_df.iloc[t-1]
        
        return weights_df
    
    def calculate_turnover(self, weights_df: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio turnover over time
        
        Parameters:
        -----------
        weights_df : pd.DataFrame
            DataFrame containing portfolio weights over time
            
        Returns:
        --------
        pd.Series
            Series containing portfolio turnover over time
        """
        # Calculate absolute weight changes
        weight_changes = weights_df.diff().abs().sum(axis=1)
        
        # First entry is NaN, set it to 0
        weight_changes.iloc[0] = 0
        
        return weight_changes
