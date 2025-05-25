#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration module for connecting ML models with the backtesting framework.
This module provides utilities for using day-type classifiers and signal filters
within backtests of trading strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class MLBacktestIntegration:
    """
    Class to integrate ML models with backtesting framework.
    Provides functionality to use day-type classifiers and signal filters
    to enhance trading strategy backtests.
    """
    
    def __init__(
        self,
        day_classifier_path: Optional[str] = None,
        signal_filter_paths: Dict[str, str] = None,
        feature_data_path: Optional[str] = None,
    ):
        """
        Initialize ML backtest integration.
        
        Args:
            day_classifier_path: Path to the day classifier model
            signal_filter_paths: Dictionary mapping strategy names to signal filter model paths
            feature_data_path: Path to precomputed feature data
        """
        self.day_classifier_path = day_classifier_path
        self.signal_filter_paths = signal_filter_paths or {}
        self.feature_data_path = feature_data_path
        
        # Initialize models
        self.day_classifier = None
        self.signal_filters = {}
        self.feature_data = None
        
        # Flag to track if models are loaded
        self.models_loaded = False
        
        # Try to load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """
        Load ML models from disk.
        
        Returns:
            bool: True if all models loaded successfully
        """
        try:
            # Import here to avoid circular imports
            from src.ml_models.day_classifier import DayClassifier
            from src.ml_models.signal_filter import SignalFilterModel
            
            # Load day classifier if path provided
            if self.day_classifier_path and os.path.exists(self.day_classifier_path):
                logger.info(f"Loading day classifier from {self.day_classifier_path}")
                self.day_classifier = DayClassifier(model_path=self.day_classifier_path)
                
            # Load signal filters if paths provided
            for strategy_name, model_path in self.signal_filter_paths.items():
                if os.path.exists(model_path):
                    logger.info(f"Loading signal filter for {strategy_name} from {model_path}")
                    self.signal_filters[strategy_name] = SignalFilterModel(
                        strategy_name=strategy_name,
                        model_path=model_path
                    )
                else:
                    logger.warning(f"Signal filter path for {strategy_name} not found: {model_path}")
            
            # Load feature data if path provided
            if self.feature_data_path and os.path.exists(self.feature_data_path):
                logger.info(f"Loading feature data from {self.feature_data_path}")
                self.feature_data = pd.read_parquet(self.feature_data_path)
                
                # Ensure index is datetime
                if not isinstance(self.feature_data.index, pd.DatetimeIndex):
                    if 'date' in self.feature_data.columns:
                        self.feature_data = self.feature_data.set_index('date')
                    elif 'timestamp' in self.feature_data.columns:
                        self.feature_data = self.feature_data.set_index('timestamp')
            
            # Mark models as loaded if at least one model is loaded
            self.models_loaded = (
                self.day_classifier is not None or 
                len(self.signal_filters) > 0
            )
            
            return self.models_loaded
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            return False
    
    def get_day_type(self, data_date: Union[str, date, datetime], market_features: Dict) -> Dict:
        """
        Get the predicted day type for a given date.
        
        Args:
            data_date: Date to predict day type for
            market_features: Dictionary with market features
            
        Returns:
            Dictionary with day type prediction
        """
        if not self.models_loaded or self.day_classifier is None:
            logger.warning("Day classifier not loaded, returning default day type")
            return {
                'day_type': 'Unknown',
                'probabilities': {},
                'confidence': 0.0,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        
        try:
            # Convert date to string format if needed
            if isinstance(data_date, (date, datetime)):
                date_str = data_date.strftime('%Y-%m-%d')
            else:
                date_str = data_date
                
            # Try to get pre-computed features
            if self.feature_data is not None:
                # Find data for the specific date
                if date_str in self.feature_data.index:
                    features = self.feature_data.loc[date_str].to_dict()
                else:
                    # Use provided market features if date not in feature data
                    features = market_features
            else:
                # Use provided market features if no feature data loaded
                features = market_features
            
            # Get day type prediction
            prediction = self.day_classifier.classify_day(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting day type: {e}")
            return {
                'day_type': 'Unknown',
                'probabilities': {},
                'confidence': 0.0,
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def filter_signal(
        self,
        strategy_name: str,
        signal_data: Dict,
        market_features: Dict,
        threshold: float = 0.5
    ) -> Dict:
        """
        Filter a trading signal using the appropriate signal filter.
        
        Args:
            strategy_name: Name of the strategy that generated the signal
            signal_data: Dictionary with signal information
            market_features: Dictionary with market features
            threshold: Probability threshold for accepting signals
            
        Returns:
            Dictionary with filter result
        """
        if not self.models_loaded or strategy_name not in self.signal_filters:
            logger.warning(f"Signal filter for {strategy_name} not loaded, signals will pass through")
            return {
                'filter_result': True,  # Default to allowing signals
                'confidence': 0.5,
                'threshold': threshold
            }
        
        try:
            # Get the appropriate filter
            signal_filter = self.signal_filters[strategy_name]
            
            # Filter the signal
            result = signal_filter.filter_signal(signal_data, market_features, threshold)
            
            return result
            
        except Exception as e:
            logger.error(f"Error filtering signal: {e}")
            return {
                'filter_result': True,  # Default to allowing signals on error
                'confidence': 0.0,
                'threshold': threshold,
                'error': str(e)
            }
    
    def prepare_backtest_data(
        self,
        price_data: pd.DataFrame,
        feature_data: Optional[pd.DataFrame] = None,
        predict_day_types: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting by adding predicted day types.
        
        Args:
            price_data: DataFrame with price data
            feature_data: DataFrame with feature data (optional)
            predict_day_types: Whether to predict day types
            
        Returns:
            DataFrame with added day type predictions
        """
        # Make a copy to avoid modifying the original
        result_df = price_data.copy()
        
        # Add day type predictions if requested and day classifier is available
        if predict_day_types and self.day_classifier is not None:
            logger.info("Adding day type predictions to backtest data")
            
            # Use provided feature data or self.feature_data
            features_to_use = feature_data if feature_data is not None else self.feature_data
            
            # Prepare to store predictions
            day_types = []
            confidences = []
            
            # Process each day
            for idx, row in result_df.iterrows():
                date_key = idx
                
                if features_to_use is not None and date_key in features_to_use.index:
                    # Use pre-computed features
                    features = features_to_use.loc[date_key].to_dict()
                else:
                    # Use price data as features
                    features = row.to_dict()
                
                # Get prediction
                prediction = self.day_classifier.classify_day(features)
                
                # Store results
                day_types.append(prediction['classification'])
                confidences.append(prediction.get('confidence', 0.0))
            
            # Add to result DataFrame
            result_df['predicted_day_type'] = day_types
            result_df['day_type_confidence'] = confidences
        
        return result_df
    
    def enable_ml_for_strategy(self, strategy_instance, strategy_name: str = None) -> None:
        """
        Enable ML capabilities for a strategy instance.
        
        Args:
            strategy_instance: Trading strategy instance
            strategy_name: Name of the strategy (if None, will try to get from instance)
        """
        if not self.models_loaded:
            logger.warning("ML models not loaded, ML capabilities will be limited")
        
        # Get strategy name if not provided
        if strategy_name is None:
            strategy_name = getattr(strategy_instance, 'name', 'default_strategy')
        
        # Add ML-related methods to the strategy instance
        def get_day_type(self, data_date, market_features=None):
            if market_features is None:
                market_features = {}  # Default empty dict
            return self.ml_integration.get_day_type(data_date, market_features)
        
        def filter_signal(self, signal_data, market_features=None, threshold=0.5):
            if market_features is None:
                market_features = {}  # Default empty dict
            return self.ml_integration.filter_signal(
                strategy_name, signal_data, market_features, threshold
            )
        
        # Add reference to self and methods to the strategy instance
        strategy_instance.ml_integration = self
        strategy_instance.get_day_type = get_day_type.__get__(strategy_instance)
        strategy_instance.filter_signal = filter_signal.__get__(strategy_instance)
        
        logger.info(f"ML capabilities enabled for strategy: {strategy_name}")


# For testing purposes
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test paths (modify these to match your actual paths)
    day_classifier_path = "models/day_classifier.pkl"
    signal_filter_paths = {
        "delta_theta": "models/signal_filter_delta_theta.pkl",
        "gamma_scalping": "models/signal_filter_gamma_scalping.pkl"
    }
    
    # Check if models exist
    if not os.path.exists(day_classifier_path):
        logger.warning(f"Day classifier model not found at {day_classifier_path}")
        day_classifier_path = None
    
    valid_signal_filters = {}
    for strategy, path in signal_filter_paths.items():
        if os.path.exists(path):
            valid_signal_filters[strategy] = path
        else:
            logger.warning(f"Signal filter for {strategy} not found at {path}")
    
    # Create integration instance
    ml_integration = MLBacktestIntegration(
        day_classifier_path=day_classifier_path,
        signal_filter_paths=valid_signal_filters
    )
    
    # Test day type prediction
    test_date = pd.Timestamp('2025-05-15')
    test_features = {
        'open': 48000,
        'high': 48500,
        'low': 47800,
        'close': 48200,
        'volume': 1000000,
        'realized_vol': 0.8,
        'gap_pct': 0.2,
        'returns_5d': 1.5
    }
    
    day_prediction = ml_integration.get_day_type(test_date, test_features)
    print(f"Predicted day type: {day_prediction.get('classification', 'Unknown')}")
    print(f"Confidence: {day_prediction.get('confidence', 0.0):.2f}")
    
    # Test signal filter
    test_signal = {
        'strategy': 'delta_theta',
        'direction': 'long',
        'entry_price': 48200
    }
    
    if 'delta_theta' in valid_signal_filters:
        filter_result = ml_integration.filter_signal(
            'delta_theta', test_signal, test_features
        )
        print(f"Signal filter result: {filter_result['filter_result']}")
        print(f"Signal confidence: {filter_result.get('confidence', 0.0):.2f}")
    
    print("ML Backtest Integration test complete.")
