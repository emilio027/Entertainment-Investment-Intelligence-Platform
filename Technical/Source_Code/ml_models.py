#!/usr/bin/env python3
"""
Entertainment Investment Intelligence Platform - ML Models Module
Advanced Machine Learning Models for Box Office Prediction and Investment Analytics

Author: Emilio Cardenas
Institution: MIT PhD AI Automation | Harvard MBA
Version: 2.0.0 Enterprise
License: Proprietary
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

# Advanced ML Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Advanced ML libraries (XGBoost, LightGBM, CatBoost) not available")

# Deep Learning (Optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logging.warning("TensorFlow not available - deep learning models disabled")

# Model Interpretability
try:
    import shap
    import lime
    import lime.tabular
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    logging.warning("Model interpretability libraries not available")


class MLModelManager:
    """
    Comprehensive machine learning model manager for entertainment investment analytics.
    Provides ensemble learning, hyperparameter optimization, and model interpretability.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the ML Model Manager."""
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Model storage
        self.models = {}
        self.pipelines = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Best model tracking
        self.best_model = None
        self.best_score = -np.inf
        
        # Feature importance
        self.feature_importance = {}
        
        self.logger.info("ML Model Manager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('MLModelManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_model_configurations(self) -> Dict[str, Dict]:
        """Create comprehensive model configurations for ensemble learning."""
        base_config = {
            'random_state': self.random_state,
            'n_jobs': -1 if hasattr(self, 'use_parallel') and self.use_parallel else 1
        }
        
        configurations = {
            # Linear Models
            'linear_regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'ridge': {
                'model': Ridge(**base_config),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            },
            'lasso': {
                'model': Lasso(**base_config),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'max_iter': [1000, 2000]
                }
            },
            'elastic_net': {
                'model': ElasticNet(**base_config),
                'params': {
                    'alpha': [0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
                    'max_iter': [1000, 2000]
                }
            },
            
            # Tree-Based Models
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=self.random_state),
                'params': {
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor(**base_config),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            
            # Support Vector Machines
            'svr': {
                'model': SVR(),
                'params': {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            
            # Neural Networks
            'mlp': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Add advanced ML models if available
        if ADVANCED_ML_AVAILABLE:
            configurations.update({
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMRegressor(random_state=self.random_state, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'catboost': {
                    'model': cb.CatBoostRegressor(random_state=self.random_state, verbose=False),
                    'params': {
                        'iterations': [100, 200, 300],
                        'depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'l2_leaf_reg': [1, 3, 5]
                    }
                }
            })
        
        return configurations
    
    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_val: pd.DataFrame = None, y_val: pd.Series = None,
                              cv_folds: int = 5) -> Dict[str, Dict]:
        """Train individual models with hyperparameter optimization."""
        self.logger.info("Training individual models with hyperparameter optimization...")
        
        # Get model configurations
        model_configs = self.create_model_configurations()
        results = {}
        
        # Create time series cross-validation if data has temporal structure
        if hasattr(X_train, 'index') and pd.api.types.is_datetime64_any_dtype(X_train.index):
            cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv_strategy = cv_folds
        
        for model_name, config in model_configs.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config['model'])
                ])
                
                # Update parameter names for pipeline
                param_grid = {}
                for param, values in config['params'].items():
                    param_grid[f'model__{param}'] = values
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_strategy,
                    scoring='neg_mean_squared_error',
                    n_jobs=1,  # Avoid nested parallelism
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    y_pred = best_model.predict(X_val)
                    mse = mean_squared_error(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                else:
                    # Use cross-validation scores
                    cv_scores = cross_val_score(best_model, X_train, y_train, 
                                              cv=cv_strategy, scoring='neg_mean_squared_error')
                    mse = -cv_scores.mean()
                    mae = None  # Not available from CV
                    r2 = None   # Not available from CV
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                # Update best model
                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = best_model
                
                # Store in class attributes
                self.models[model_name] = best_model
                self.performance_metrics[model_name] = results[model_name]
                
                self.logger.info(f"{model_name} - CV Score: {grid_search.best_score_:.4f}, "
                               f"RMSE: {np.sqrt(mse):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.logger.info(f"Individual model training completed. Best model: {self.get_best_model_name()}")
        return results
    
    def create_ensemble_model(self, model_weights: Dict[str, float] = None) -> VotingRegressor:
        """Create ensemble model from trained individual models."""
        if not self.models:
            raise ValueError("No models trained yet. Train individual models first.")
        
        self.logger.info("Creating ensemble model...")
        
        # Use all models if no weights specified
        if model_weights is None:
            # Weight by performance (inverse of MSE)
            weights = []
            estimators = []
            for name, model in self.models.items():
                if name in self.performance_metrics and 'mse' in self.performance_metrics[name]:
                    mse = self.performance_metrics[name]['mse']
                    weight = 1.0 / (mse + 1e-6)  # Add small epsilon to avoid division by zero
                    weights.append(weight)
                    estimators.append((name, model))
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        else:
            estimators = [(name, model) for name, model in self.models.items() 
                         if name in model_weights]
            weights = [model_weights[name] for name, _ in estimators]
        
        # Create voting regressor
        ensemble = VotingRegressor(estimators=estimators, weights=weights)
        
        self.models['ensemble'] = ensemble
        self.logger.info(f"Ensemble created with {len(estimators)} models")
        
        return ensemble
    
    def create_deep_learning_model(self, input_dim: int, hidden_layers: List[int] = None,
                                 dropout_rate: float = 0.3, learning_rate: float = 0.001) -> Any:
        """Create deep learning model using TensorFlow/Keras."""
        if not DEEP_LEARNING_AVAILABLE:
            self.logger.warning("TensorFlow not available - skipping deep learning model")
            return None
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        self.logger.info(f"Creating deep learning model with architecture: {hidden_layers}")
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_deep_learning_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame = None, y_val: pd.Series = None,
                                epochs: int = 100, batch_size: int = 32) -> Any:
        """Train deep learning model."""
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        self.logger.info("Training deep learning model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            # Use 20% of training data for validation
            X_train_scaled, X_val_scaled, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=self.random_state
            )
            validation_data = (X_val_scaled, y_val_split)
            y_train = y_train_split
        
        # Create model
        model = self.create_deep_learning_model(input_dim=X_train_scaled.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Store model and scaler
        self.models['deep_learning'] = model
        self.scalers['deep_learning'] = scaler
        
        # Evaluate
        if X_val is not None:
            y_pred = model.predict(X_val_scaled).flatten()
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
        else:
            y_pred = model.predict(X_val_scaled).flatten()
            mse = mean_squared_error(y_val_split, y_pred)
            mae = mean_absolute_error(y_val_split, y_pred)
            r2 = r2_score(y_val_split, y_pred)
        
        self.performance_metrics['deep_learning'] = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'history': history
        }
        
        self.logger.info(f"Deep learning model - RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")
        
        return model
    
    def get_feature_importance(self, model_name: str = None, X: pd.DataFrame = None) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name is None:
            model_name = self.get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Extract model from pipeline if needed
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model
        
        # Get feature importance based on model type
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            importances = np.abs(actual_model.coef_)
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return None
        
        # Create feature importance DataFrame
        if X is not None:
            feature_names = X.columns
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_prediction(self, X_instance: pd.DataFrame, model_name: str = None) -> Dict:
        """Explain individual prediction using SHAP or LIME."""
        if not INTERPRETABILITY_AVAILABLE:
            self.logger.warning("Interpretability libraries not available")
            return {}
        
        if model_name is None:
            model_name = self.get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Try SHAP first
            if hasattr(model, 'predict'):
                explainer = shap.Explainer(model)
                shap_values = explainer(X_instance)
                
                return {
                    'method': 'SHAP',
                    'shap_values': shap_values.values[0],
                    'feature_names': X_instance.columns.tolist(),
                    'base_value': shap_values.base_values[0],
                    'prediction': model.predict(X_instance)[0]
                }
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {str(e)}")
        
        # Fallback to LIME
        try:
            explainer = lime.tabular.LimeTabularExplainer(
                X_instance.values,
                feature_names=X_instance.columns.tolist(),
                mode='regression'
            )
            
            explanation = explainer.explain_instance(
                X_instance.iloc[0].values,
                model.predict,
                num_features=min(10, len(X_instance.columns))
            )
            
            return {
                'method': 'LIME',
                'explanation': explanation,
                'prediction': model.predict(X_instance)[0]
            }
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {str(e)}")
            return {}
    
    def get_best_model_name(self) -> str:
        """Get the name of the best performing model."""
        if not self.performance_metrics:
            return None
        
        best_name = min(self.performance_metrics.keys(), 
                       key=lambda x: self.performance_metrics[x].get('mse', float('inf')))
        return best_name
    
    def predict(self, X: pd.DataFrame, model_name: str = None, 
                return_confidence: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using specified model or best model."""
        if model_name is None:
            model_name = self.get_best_model_name()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Handle deep learning models
        if model_name == 'deep_learning' and model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            predictions = model.predict(X_scaled).flatten()
        else:
            predictions = model.predict(X)
        
        if return_confidence:
            # Simple confidence estimation based on prediction variance
            confidence = np.ones_like(predictions) * 0.95  # Default confidence
            return predictions, confidence
        
        return predictions
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models to disk."""
        self.logger.info(f"Saving models to {filepath}")
        
        model_data = {
            'models': self.models,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'best_model_name': self.get_best_model_name(),
            'scalers': self.scalers
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info("Models saved successfully")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        self.logger.info(f"Loading models from {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.performance_metrics = model_data['performance_metrics']
        self.feature_importance = model_data.get('feature_importance', {})
        self.scalers = model_data.get('scalers', {})
        
        self.logger.info("Models loaded successfully")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models."""
        if not self.performance_metrics:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'Model': model_name,
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'R²': metrics.get('r2', np.nan),
                'CV_Score': metrics.get('cv_score', np.nan)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE') if 'RMSE' in summary_df.columns else summary_df
        
        return summary_df


def main():
    """Main function demonstrating ML Model Manager usage."""
    print("=" * 80)
    print("Entertainment Investment Intelligence Platform")
    print("Machine Learning Models Module")
    print("Author: Emilio Cardenas | MIT PhD AI Automation | Harvard MBA")
    print("=" * 80)
    
    # Initialize ML Manager
    ml_manager = MLModelManager(random_state=42)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Create realistic target variable
    y = (
        X['feature_0'] * 2.5 +
        X['feature_1'] * 1.8 +
        X['feature_2'] * -1.2 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Train models
    print("\nTraining individual models...")
    results = ml_manager.train_individual_models(X_train, y_train, X_test, y_test)
    
    # Create ensemble
    print("\nCreating ensemble model...")
    ensemble = ml_manager.create_ensemble_model()
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    
    print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
    print(f"Ensemble R²: {ensemble_r2:.4f}")
    
    # Show model summary
    print("\nModel Performance Summary:")
    summary = ml_manager.get_model_summary()
    print(summary)
    
    # Feature importance
    best_model_name = ml_manager.get_best_model_name()
    print(f"\nBest Model: {best_model_name}")
    
    if best_model_name:
        importance = ml_manager.get_feature_importance(best_model_name, X)
        if importance is not None:
            print("\nTop 5 Most Important Features:")
            print(importance.head())
    
    print("\nML Model Manager demonstration completed successfully!")


if __name__ == "__main__":
    main()