import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class RegimePredictor:
    """
    Class for predicting market regimes using Random Forest
    """
    
    def __init__(self, n_estimators: int = 100, horizon: int = 5):
        """
        Initialize the RegimePredictor class
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        horizon : int
            Prediction horizon in days
        """
        self.n_estimators = n_estimators
        self.horizon = horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_data(self, features_df: pd.DataFrame, regimes_df: pd.DataFrame) -> tuple:
        """
        Prepare data for model training
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features
        regimes_df : pd.DataFrame
            DataFrame containing regime labels
            
        Returns:
        --------
        tuple
            Tuple containing (X, y, indices) where X is the feature matrix,
            y is the target vector, and indices are the corresponding datetime indices
        """
        # Shift regime labels to create future prediction targets
        target_col = 'market'  # Use market regime as target
        target_series = regimes_df[target_col].shift(-self.horizon)
        
        # Align features and target
        common_index = features_df.index.intersection(target_series.dropna().index)
        X = features_df.loc[common_index]
        y = target_series.loc[common_index]
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, common_index
    
    def train_model(self, features_df: pd.DataFrame, regimes_df: pd.DataFrame, 
                    tune_hyperparams: bool = False) -> tuple:
        """
        Train the Random Forest model for regime prediction
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features
        regimes_df : pd.DataFrame
            DataFrame containing regime labels
        tune_hyperparams : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        tuple
            Tuple containing training and testing results dictionaries
        """
        # Prepare data
        X, y, indices = self.prepare_data(features_df, regimes_df)
        
        # Split data into training and testing sets (time-aware split)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best parameters
            best_params = grid_search.best_params_
            self.model = RandomForestClassifier(random_state=42, **best_params)
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=42
            )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        train_results = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average='weighted'),
            'recall': recall_score(y_train, y_train_pred, average='weighted'),
            'f1': f1_score(y_train, y_train_pred, average='weighted'),
            'predictions': pd.Series(y_train_pred, index=train_indices)
        }
        
        # Evaluate on testing set
        y_test_pred = self.model.predict(X_test)
        test_results = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average='weighted'),
            'recall': recall_score(y_test, y_test_pred, average='weighted'),
            'f1': f1_score(y_test, y_test_pred, average='weighted'),
            'predictions': pd.Series(y_test_pred, index=test_indices),
            'report': classification_report(y_test, y_test_pred)
        }
        
        return train_results, test_results
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make regime predictions using the trained model
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing regime predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        # Create a DataFrame with predictions and probabilities
        predictions_df = pd.DataFrame(index=features_df.index)
        predictions_df['predicted_regime'] = predictions
        
        # Add probability columns for each regime
        for i, regime in enumerate(self.model.classes_):
            predictions_df[f'prob_regime_{regime}'] = probabilities[:, i]
        
        # Add prediction confidence (probability of the predicted class)
        predictions_df['confidence'] = np.max(probabilities, axis=1)
        
        # Add regime names based on numeric codes
        regime_names = {
            0: 'Bear Market',
            1: 'Bull Market',
            2: 'Bear-to-Bull Transition',
            3: 'Bull-to-Bear Transition'
        }
        
        predictions_df['regime_name'] = predictions_df['predicted_regime'].map(
            lambda x: regime_names.get(x, f'Regime {x}')
        )
        
        return predictions_df
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from the trained model
        
        Returns:
        --------
        pd.Series
            Series containing feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
