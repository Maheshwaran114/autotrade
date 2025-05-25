#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trade Signal Filter Model for Bank Nifty Options Trading System.
This module implements a filter to validate trading signals based on market conditions.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class SignalFilterModel:
    """
    Model to filter trade signals based on market conditions.
    This helps improve the quality of trading signals by identifying conditions
    favorable for specific trade setups.
    """
    
    def __init__(
        self,
        strategy_name: str,
        model_path: Optional[str] = None,
        feature_threshold: float = 0.01
    ):
        """
        Initialize the signal filter model.
        
        Args:
            strategy_name: Name of the trading strategy this filter applies to
            model_path: Path to a pre-trained model file
            feature_threshold: Minimum importance threshold for features
        """
        self.strategy_name = strategy_name
        self.model_path = model_path
        self.feature_threshold = feature_threshold
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        
        # If model path provided, try to load it
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            logger.info(f"No model path provided or model not found. "
                       f"Use train() to train a new model.")
    
    def load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Load model and metadata
            model_info = joblib.load(self.model_path)
            
            # Restore model attributes
            self.model = model_info.get('model')
            self.feature_names = model_info.get('feature_names', [])
            self.strategy_name = model_info.get('strategy_name', self.strategy_name)
            
            logger.info(f"Signal filter model loaded from {self.model_path}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            return False
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.model_loaded:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_info = {
            'model': self.model,
            'feature_names': self.feature_names,
            'strategy_name': self.strategy_name,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_info, filepath)
        logger.info(f"Signal filter model saved to {filepath}")
    
    def prepare_training_data(
        self,
        signals_df: pd.DataFrame,
        market_data_df: pd.DataFrame,
        label_column: str = 'success',
        min_success_rate: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by combining signal data with market features.
        
        Args:
            signals_df: DataFrame with trading signals
            market_data_df: DataFrame with market features
            label_column: Column name for signal success/failure label
            min_success_rate: Minimum success rate required to proceed (sanity check)
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Check if signals DataFrame contains required column
            if label_column not in signals_df.columns:
                raise ValueError(f"Signals DataFrame must contain '{label_column}' column")
                
            # Check signal success rate
            success_rate = signals_df[label_column].mean()
            logger.info(f"Signal success rate: {success_rate:.2%}")
            
            if success_rate < min_success_rate:
                logger.warning(f"Signal success rate {success_rate:.2%} is below "
                              f"minimum threshold {min_success_rate:.2%}")
            
            # Ensure signals_df has datetime index 
            if not isinstance(signals_df.index, pd.DatetimeIndex):
                if 'date' in signals_df.columns:
                    signals_df = signals_df.set_index('date')
                elif 'timestamp' in signals_df.columns:
                    signals_df = signals_df.set_index('timestamp')
                else:
                    raise ValueError("Signals DataFrame must have date column or datetime index")

            # Same for market_data_df
            if not isinstance(market_data_df.index, pd.DatetimeIndex):
                if 'date' in market_data_df.columns:
                    market_data_df = market_data_df.set_index('date')
                elif 'timestamp' in market_data_df.columns:
                    market_data_df = market_data_df.set_index('timestamp')
                else:
                    raise ValueError("Market data DataFrame must have date column or datetime index")
            
            # Merge signals with market data
            merged_df = signals_df.join(
                market_data_df, 
                how='inner',
                lsuffix='_signal',
                rsuffix='_market'
            )
            
            if merged_df.empty:
                raise ValueError("No matching data after joining signals with market data. "
                               "Check that the timestamps align.")
                
            # Extract labels
            y = merged_df[label_column]
            
            # Extract features (exclude label and any signal-specific columns)
            exclude_columns = [label_column, 'strategy', 'signal_id', 'trade_id']
            feature_cols = [col for col in merged_df.columns 
                           if col not in exclude_columns 
                           and not col.startswith('_')]
            
            X = merged_df[feature_cols]
            
            # Store feature names for later use
            self.feature_names = X.columns.tolist()
            
            logger.info(f"Prepared training data with {X.shape[0]} samples and {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train the signal filter model.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Initialize RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
            
            # Train the model
            logger.info(f"Training signal filter model on {X_train.shape[0]} samples")
            self.model.fit(X_train, y_train)
            self.model_loaded = True
            
            # Evaluate on train and test sets
            train_preds = self.model.predict(X_train)
            test_preds = self.model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            
            logger.info(f"Train accuracy: {train_accuracy:.4f}")
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            logger.info(f"5-fold CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Get feature importance
            importances = self.model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Identify important features based on threshold
            important_features = importance_df[
                importance_df['importance'] > self.feature_threshold
            ]
            
            logger.info(f"Identified {len(important_features)} important features "
                       f"(threshold: {self.feature_threshold})")
            
            # Return results
            return {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'train_report': classification_report(y_train, train_preds, output_dict=True),
                'test_report': classification_report(y_test, test_preds, output_dict=True),
                'feature_importance': importance_df,
                'important_features': important_features
            }
            
        except Exception as e:
            logger.error(f"Error training signal filter model: {e}")
            raise
    
    def filter_signal(
        self,
        signal_data: Dict,
        market_features: Dict,
        threshold: float = 0.5
    ) -> Dict:
        """
        Filter a trading signal based on market conditions.
        
        Args:
            signal_data: Dictionary with signal information
            market_features: Dictionary with market features
            threshold: Probability threshold for accepting signals
            
        Returns:
            Dictionary with filter result and confidence
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, signals will not be filtered")
            return {
                'filter_result': True,  # Default to allowing signals
                'confidence': 0.5,
                'threshold': threshold
            }
        
        try:
            # Convert to DataFrame
            features_df = pd.DataFrame(market_features, index=[0])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}. Adding defaults.")
                for feature in missing_features:
                    features_df[feature] = 0.0  # Default value
            
            # Select and order features to match trained model
            features_df = features_df[self.feature_names]
            
            # Get probability of signal success
            proba = self.model.predict_proba(features_df)[0]
            
            # Class 1 probability (success)
            success_proba = proba[1]
            
            # Determine if signal passes filter
            passes_filter = success_proba >= threshold
            
            # Detailed signal analysis
            feature_contribution = {}
            if hasattr(self.model, 'feature_importances_'):
                # Get feature importance
                importances = self.model.feature_importances_
                
                # Get top 5 features by importance
                top_indices = np.argsort(importances)[-5:]
                
                # Get feature contribution
                for idx in top_indices:
                    feature_name = self.feature_names[idx]
                    feature_value = features_df.iloc[0, idx]
                    feature_contribution[feature_name] = {
                        'value': feature_value,
                        'importance': importances[idx]
                    }
            
            logger.info(f"Signal filter result: {passes_filter} with confidence {success_proba:.2%}")
            
            return {
                'filter_result': bool(passes_filter),
                'confidence': float(success_proba),
                'threshold': threshold,
                'signal_data': signal_data,
                'feature_contribution': feature_contribution,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error filtering signal: {e}")
            # Default to accepting signal when errors occur
            return {
                'filter_result': True,
                'confidence': 0.0,
                'threshold': threshold,
                'error': str(e)
            }
    
    def plot_feature_importance(
        self,
        top_n: int = 15,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
            save_path: Path to save the plot (if None, just display)
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot plot feature importance")
            return
            
        try:
            # Get feature importance
            importances = self.model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Take top N features
            plot_df = importance_df.head(top_n)
            
            # Plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=plot_df)
            plt.title(f'Top {top_n} Features for {self.strategy_name} Signal Filter')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")


# For testing purposes
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple test
    try:
        # Sample data for testing
        # In a real scenario, this would come from historical signals and market data
        np.random.seed(42)
        n_samples = 200
        
        # Generate sample signals
        signals = pd.DataFrame({
            'date': pd.date_range(start='2025-01-01', periods=n_samples),
            'signal_type': np.random.choice(['long', 'short'], n_samples),
            'entry_price': np.random.normal(48000, 500, n_samples),
            'success': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
        }).set_index('date')
        
        # Generate sample market features
        market_data = pd.DataFrame({
            'date': signals.index,
            'volatility': np.random.normal(0.8, 0.2, n_samples),
            'rsi': np.random.normal(50, 10, n_samples),
            'macd': np.random.normal(0, 1, n_samples),
            'volume': np.random.normal(1000000, 200000, n_samples),
            'gap_pct': np.random.normal(0.1, 0.5, n_samples),
            'day_of_week': np.random.randint(0, 5, n_samples)
        }).set_index('date')
        
        # Create the filter model
        filter_model = SignalFilterModel(strategy_name="sample_strategy")
        
        # Prepare training data
        X, y = filter_model.prepare_training_data(signals, market_data)
        
        # Train the model
        results = filter_model.train(X, y)
        
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print("\nTop 5 Features by Importance:")
        print(results['feature_importance'].head(5))
        
        # Test the filter on a new signal
        new_signal = {
            'signal_type': 'long',
            'entry_price': 48200,
            'timestamp': datetime.now().isoformat()
        }
        
        new_market_features = {
            'volatility': 0.9,
            'rsi': 65,
            'macd': 0.5,
            'volume': 1200000,
            'gap_pct': 0.3,
            'day_of_week': 2
        }
        
        filter_result = filter_model.filter_signal(new_signal, new_market_features, threshold=0.6)
        
        print("\nFilter Result:")
        print(f"  Signal accepted: {filter_result['filter_result']}")
        print(f"  Confidence: {filter_result['confidence']:.2%}")
        
        # Plot feature importance
        filter_model.plot_feature_importance()
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        filter_model.save_model("models/signal_filter.pkl")
        
        print("\nModel saved to models/signal_filter.pkl")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
