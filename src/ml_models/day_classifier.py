#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Day-Type Classifier for Bank Nifty Options Trading System.
This module implements machine learning models to classify trading days into different regimes.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

# Configure logging
logger = logging.getLogger(__name__)


class DayClassifier:
    """
    Machine learning classifier for identifying trading day types
    (Trend, RangeBound, Event, MildBias, Momentum).
    """
    
    DAY_TYPES = ['Trend', 'RangeBound', 'Event', 'MildBias', 'Momentum']
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict] = None,
        calibrate: bool = True,
        model_path: Optional[str] = None
    ):
        """
        Initialize the day type classifier.
        
        Args:
            model_type: Type of classifier ('random_forest', 'gradient_boosting', 'svm')
            model_params: Parameters for the classifier
            calibrate: Whether to calibrate the classifier for better probability estimates
            model_path: Path to a pre-trained model file
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.calibrate = calibrate
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            # Initialize base classifier
            self._initialize_model()
        
        logger.info(f"Initialized DayClassifier with {model_type} model type")
    
    def _initialize_model(self) -> None:
        """Initialize the underlying classifier model based on model_type"""
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            # Update with user-provided params
            params = {**default_params, **self.model_params}
            base_model = RandomForestClassifier(**params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'subsample': 0.8
            }
            # Update with user-provided params
            params = {**default_params, **self.model_params}
            base_model = GradientBoostingClassifier(**params)
            
        elif self.model_type == 'svm':
            default_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            # Update with user-provided params
            params = {**default_params, **self.model_params}
            base_model = SVC(**params)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Apply calibration if requested (helps get better probability estimates)
        if self.calibrate:
            self.model = CalibratedClassifierCV(
                base_model,
                cv=5,
                method='sigmoid'
            )
        else:
            self.model = base_model
    
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
            self.model = model_info['model']
            self.model_type = model_info.get('model_type', self.model_type)
            self.model_params = model_info.get('model_params', {})
            self.feature_names = model_info.get('feature_names', [])
            self.calibrate = model_info.get('calibrate', self.calibrate)
            
            logger.info(f"Model loaded from {self.model_path}")
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            # Initialize a new model as fallback
            self._initialize_model()
            return False
    
    def preprocess_data(self, market_data: Dict) -> pd.DataFrame:
        """
        Preprocess market data for classification.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            pd.DataFrame: Preprocessed features for model input
        """
        try:
            # Convert to DataFrame if not already
            if not isinstance(market_data, pd.DataFrame):
                df = pd.DataFrame(market_data, index=[0])
            else:
                df = market_data.copy()
            
            # Ensure feature names match those used during training
            if self.feature_names:
                missing_features = set(self.feature_names) - set(df.columns)
                extra_features = set(df.columns) - set(self.feature_names)
                
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}. Adding with default values.")
                    for feature in missing_features:
                        df[feature] = 0  # Default value
                
                # Select only needed features in the correct order
                df = df[self.feature_names]
            
            logger.info("Preprocessed market data for classification")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train the day type classifier model.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            X_val: Validation feature matrix (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        # Store feature names for later use
        self.feature_names = X_train.columns.tolist()
        
        # Train the model
        logger.info(f"Training {self.model_type} classifier on {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        self.model_loaded = True
        
        # Compute training metrics
        train_preds = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Initialize results dictionary
        results = {
            'train_accuracy': train_accuracy,
            'train_report': classification_report(y_train, train_preds, output_dict=True)
        }
        
        # Compute validation metrics if validation data is provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_preds)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            
            results.update({
                'val_accuracy': val_accuracy,
                'val_report': classification_report(y_val, val_preds, output_dict=True)
            })
        
        return results
    
    def classify_day(self, market_data: Dict) -> Dict:
        """
        Classify the trading day based on market data.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dict: Classification results with probabilities
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, initializing default model")
            self._initialize_model()
        
        try:
            # Preprocess the data
            features = self.preprocess_data(market_data)
            
            # Get prediction and probabilities
            day_type = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Create probabilities dictionary
            probs_dict = {
                class_name: float(prob)
                for class_name, prob in zip(self.DAY_TYPES, probabilities)
            }
            
            logger.info(f"Day classified as {day_type}")
            
            return {
                "classification": day_type,
                "probabilities": probs_dict,
                "confidence": float(probs_dict.get(day_type, 0.0)),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during day classification: {e}")
            # Fallback to a default classification
            return {
                "classification": "Unknown",
                "probabilities": {type_name: 0.0 for type_name in self.DAY_TYPES},
                "confidence": 0.0,
                "timestamp": pd.Timestamp.now().isoformat(),
                "error": str(e)
            }


# Additional methods for model evaluation and analysis
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict day types for input features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted day types
        """
        if not self.model_loaded:
            raise ValueError("Model not trained yet. Call train() first.")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict day type probabilities for input features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of day type probabilities [n_samples, n_classes]
        """
        if not self.model_loaded:
            raise ValueError("Model not trained yet. Call train() first.")
            
        return self.model.predict_proba(X)
    
    def crossvalidate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation to evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        # Define cross-validation strategy
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        logger.info(f"Performing {cv}-fold cross-validation")
        cv_scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring='accuracy')
        
        # Compute metrics
        results = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_accuracy_min': cv_scores.min(),
            'cv_accuracy_max': cv_scores.max()
        }
        
        logger.info(f"Cross-validation accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        
        return results
    
    def hypertune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning to find the best model parameters.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Grid of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and scores
        """
        logger.info("Starting hyperparameter tuning")
        
        # Create a base model to tune (without calibration)
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'svm':
            base_model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Define cross-validation strategy
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Perform grid search
        grid_search.fit(X, y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best {scoring} score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Update model with best parameters
        self.model_params = best_params
        self._initialize_model()
        
        # Return results
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }
    
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
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'calibrate': self.calibrate,
            'day_types': self.DAY_TYPES
        }
        
        joblib.dump(model_info, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.model_loaded:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # For CalibratedClassifierCV, get the base estimator
        if self.calibrate:
            # Access the base estimator of the first calibrator
            base_model = self.model.calibrated_classifiers_[0].base_estimator
        else:
            base_model = self.model
        
        # Check if model supports feature importance
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
        else:
            logger.warning(f"{self.model_type} does not support direct feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        X: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix for model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (if None, will generate predictions from X)
            X: Feature matrix (required if y_pred is None)
            normalize: Whether to normalize confusion matrix
            save_path: Path to save the plot (if None, just display)
        """
        if y_pred is None:
            if X is None:
                raise ValueError("Either y_pred or X must be provided")
            y_pred = self.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.DAY_TYPES,
            yticklabels=self.DAY_TYPES
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
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
        # Get feature importance
        importance_df = self.get_feature_importance()
        
        if importance_df.empty:
            logger.warning("Feature importance not available for this model")
            return
        
        # Take top N features
        plot_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=plot_df)
        plt.title('Feature Importance')
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


# For testing purposes
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.features.feature_engineering import FeatureEngineer
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Check if we can load a model
        model_path = "models/day_classifier.pkl"
        if os.path.exists(model_path):
            print(f"Testing with existing model at {model_path}")
            classifier = DayClassifier(model_path=model_path)
            
            # Test classification
            sample_data = {
                "open": 48000.0,
                "high": 48500.0,
                "low": 47800.0,
                "close": 48200.0,
                "volume": 1000000,
                # Add missing features based on your model requirements
            }
            
            result = classifier.classify_day(sample_data)
            print(f"Day classified as: {result['classification']} with {result['confidence']:.2f} confidence")
            
        else:
            print("No existing model found. Creating and training a new model...")
            
            # Try to load labeled data
            labeled_data_path = "data/processed/labeled_days.parquet"
            if not os.path.exists(labeled_data_path):
                print("No labeled data found. Creating sample data for testing...")
                
                # Create sample data
                dates = pd.date_range(start='2025-01-01', periods=100)
                np.random.seed(42)
                
                df = pd.DataFrame({
                    'date': dates,
                    'open': np.random.normal(48000, 500, 100),
                    'high': np.random.normal(48200, 600, 100),
                    'low': np.random.normal(47800, 600, 100),
                    'close': np.random.normal(48100, 500, 100),
                    'volume': np.random.normal(1000000, 200000, 100),
                    'label': np.random.choice(['Trend', 'RangeBound', 'Event', 'MildBias', 'Momentum'], 100)
                })
                
                # Create parquet directory if it doesn't exist
                os.makedirs("data/processed", exist_ok=True)
                
                # Save sample data
                df.to_parquet(labeled_data_path)
                print(f"Saved sample data to {labeled_data_path}")
                
            # Process features using feature engineering
            engineer = FeatureEngineer()
            datasets = engineer.prepare_ml_datasets(labeled_data_path)
            
            if not datasets:
                print("Failed to prepare datasets. Check if the data is valid.")
                sys.exit(1)
                
            # Create and train the classifier
            classifier = DayClassifier(model_type='random_forest')
            results = classifier.train(
                datasets['X_train'],
                datasets['y_train'],
                datasets['X_test'],
                datasets['y_test']
            )
            
            print(f"Training accuracy: {results['train_accuracy']:.4f}")
            if 'val_accuracy' in results:
                print(f"Validation accuracy: {results['val_accuracy']:.4f}")
            
            # Create output directories
            os.makedirs("models", exist_ok=True)
            os.makedirs("reports", exist_ok=True)
            
            # Plot feature importance
            classifier.plot_feature_importance(save_path="reports/day_classifier_feature_importance.png")
            
            # Plot confusion matrix
            classifier.plot_confusion_matrix(
                datasets['y_test'], 
                X=datasets['X_test'],
                save_path="reports/day_classifier_confusion_matrix.png"
            )
            
            # Save the model
            classifier.save_model("models/day_classifier.pkl")
            
            print("Model training and evaluation complete.")
            print("Results saved to reports/ directory")
            print("Model saved to models/day_classifier.pkl")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
