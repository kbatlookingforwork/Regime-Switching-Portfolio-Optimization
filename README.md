
# Regime-Switching Portfolio Optimization
![Uploading image.png…]()

This application implements a data-driven framework for regime-switching portfolio optimization using a combination of machine learning and traditional financial approaches. The idea of regime-switching portfolio optimization originates from a [dissertation by P. Pomorski](https://www.nber.org/papers/w17703) that explores how asset returns behave differently across different market regimes.

## Features

- **Market Regime Detection**: Uses KAMA (Kaufman's Adaptive Moving Average) and Markov-Switching Regression to identify market regimes
- **Regime Prediction**: Employs Random Forest for predicting future market regimes
- **Portfolio Optimization**: Implements Model Predictive Control (MPC) with regime-switching considerations
- **Performance Analysis**: Calculates and visualizes key performance metrics

## Project Structure

```
├── modules/
│   ├── data_handler.py         # Data operations and fetching
│   ├── regime_detection.py     # Market regime detection
│   ├── regime_prediction.py    # Regime prediction
│   ├── portfolio_optimization.py # Portfolio optimization
│   └── performance_metrics.py  # Performance metrics
├── utils/
│   ├── helpers.py             # Helper functions
│   └── visualizations.py      # Visualization functions
└── app.py                     # Main Streamlit application
```

## Key Components

1. **Regime Detection**
   - Bull/Bear market identification
   - Transition regime detection
   - Regime statistics calculation

2. **Regime Prediction**
   - Feature engineering
   - Random Forest classification
   - Cross-validation and hyperparameter tuning

3. **Portfolio Optimization**
   - Regime-specific risk adjustments
   - Transaction cost consideration
   - Maximum allocation constraints

4. **Performance Analysis**
   - Return metrics
   - Risk metrics
   - Regime-specific analysis

## Usage

The application runs on Streamlit and provides an interactive interface for:
- Selecting asset classes
- Configuring model parameters
- Visualizing regime transitions
- Analyzing portfolio performance

## Detailed Methodologies

#### KAMA Calculation
```python
ER = |Price Change| / Volatility
SC = [ER × (Fast_alpha - Slow_alpha) + Slow_alpha]²
KAMA = KAMA_previous + SC × (Price - KAMA_previous)
```
where:
- ER = Efficiency Ratio
- SC = Smoothing Constant
- Fast_alpha = 2/(2+1)
- Slow_alpha = 2/(30+1)

#### Regime Classification
- **Bear Market (0)**: High volatility, negative trend
- **Bull Market (1)**: Low volatility, positive trend
- **Bear-to-Bull Transition (2)**: Regime probability near 0.5 with positive momentum
- **Bull-to-Bear Transition (3)**: Regime probability near 0.5 with negative momentum

### 2. Regime Prediction
Uses Random Forest classification with time-series specific considerations:

#### Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volatility measures
- Cross-asset correlation features
- Regime transition indicators

#### Model Training
- Time-series cross-validation
- Hyperparameter optimization via GridSearchCV
- Feature importance analysis
- Performance metrics:
  ```
  Accuracy = Correct Predictions / Total Predictions
  Precision = True Positives / (True Positives + False Positives)
  Recall = True Positives / (True Positives + False Negatives)
  F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
  ```

### 3. Portfolio Optimization
Implements Model Predictive Control (MPC) with regime-switching considerations:

#### Objective Function
```
Maximize: Returns - λ₍ᵣₑᵧᵢₘₑ₎ × Risk - Transaction_Costs

where:
λ₍ᵣₑᵧᵢₘₑ₎ = Base_Risk_Aversion × Regime_Multiplier
```

#### Regime-Specific Risk Multipliers
- Bear Market: 2.0x risk aversion
- Bull Market: 0.8x risk aversion
- Bear-to-Bull Transition: 1.0x risk aversion
- Bull-to-Bear Transition: 1.5x risk aversion

#### Constraints
- Sum of weights = 1 (fully invested)
- 0 ≤ weight ≤ max_allocation (long-only, concentration limits)
- Transaction costs = cost_factor × Σ|weight_changes|

### 4. Performance Analysis
Comprehensive performance metrics calculation:

#### Return Metrics
- Annualized Return = Mean_Daily_Return × 252
- Cumulative Return = Π(1 + r_t) - 1

#### Risk Metrics
- Annualized Volatility = Daily_Std × √252
- Sharpe Ratio = (Ann_Return - Risk_Free_Rate) / Ann_Volatility
- Sortino Ratio = Ann_Return / Downside_Volatility
- Maximum Drawdown = min((Port_Value - Peak_Value) / Peak_Value)

#### Regime-Specific Analysis
- Conditional performance metrics per regime
- Transition analysis
- Regime-specific risk-adjusted returns

#### Advanced Metrics
- Information Ratio = Active_Return / Tracking_Error
- Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
- Alpha = Portfolio_Return - (Risk_Free_Rate + Beta × Market_Premium)

## References

[1] P. Pomorski, "Economic Scenario Generation Using Regime-Switching Models," Ph.D. dissertation, Department of Economics, Warsaw School of Economics, Poland, 2023.
