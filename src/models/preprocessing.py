"""
Feature preprocessing pipeline for Bank Nifty Options Trading System.

This module provides StandardScaler-based feature normalization for ML models,
ensuring all features are scaled to have zero mean and unit variance.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class StandardScalerPipeline:
    """
    Feature scaling pipeline using StandardScaler.
    
    Provides methods to fit, transform, and save/load the scaler
    for consistent preprocessing across training and inference.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> 'StandardScalerPipeline':
        """
        Fit the scaler on training features.
        
        Args:
            X: Feature matrix (n_samples × n_features)
            
        Returns:
            self: Fitted pipeline
        """
        logger.info(f"Fitting StandardScaler on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Store feature names for validation
        self.feature_names = list(X.columns)
        
        # Fit scaler
        self.scaler.fit(X.values)
        self.is_fitted = True
        
        # Log scaling parameters
        logger.info(f"Feature scaling parameters:")
        for i, feature in enumerate(self.feature_names):
            mean = self.scaler.mean_[i]
            std = np.sqrt(self.scaler.var_[i])
            logger.info(f"  {feature}: mean={mean:.4f}, std={std:.4f}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            pd.DataFrame: Scaled features with same index and columns
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming")
        
        # Validate feature names match
        if list(X.columns) != self.feature_names:
            raise ValueError(f"Feature names don't match. Expected: {self.feature_names}, Got: {list(X.columns)}")
        
        # Transform features
        X_scaled = self.scaler.transform(X.values)
        
        # Return as DataFrame with original index and columns
        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler and transform features in one step.
        
        Args:
            X: Feature matrix to fit and transform
            
        Returns:
            pd.DataFrame: Scaled features
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            pd.DataFrame: Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before inverse transforming")
        
        # Inverse transform
        X_original = self.scaler.inverse_transform(X_scaled.values)
        
        # Return as DataFrame with original index and columns
        return pd.DataFrame(X_original, index=X_scaled.index, columns=X_scaled.columns)
    
    def save(self, filepath: str) -> None:
        """
        Save fitted scaler to disk.
        
        Args:
            filepath: Path to save the scaler
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline state
        pipeline_state = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        logger.info(f"Saved StandardScaler pipeline to {filepath}")
    
    def load(self, filepath: str) -> 'StandardScalerPipeline':
        """
        Load fitted scaler from disk.
        
        Args:
            filepath: Path to load the scaler from
            
        Returns:
            self: Loaded pipeline
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        self.scaler = pipeline_state['scaler']
        self.feature_names = pipeline_state['feature_names']
        self.is_fitted = pipeline_state['is_fitted']
        
        logger.info(f"Loaded StandardScaler pipeline from {filepath}")
        logger.info(f"Pipeline fitted for {len(self.feature_names)} features: {self.feature_names}")
        
        return self


def preprocess_features(features_file: str = None, 
                       save_scaler: bool = True, 
                       scaler_file: str = None) -> Tuple[pd.DataFrame, StandardScalerPipeline]:
    """
    Main preprocessing function to load and scale features.
    
    Args:
        features_file: Path to features pickle file (defaults to processed/features.pkl)
        save_scaler: Whether to save the fitted scaler
        scaler_file: Path to save scaler (defaults to models/scaler.pkl)
        
    Returns:
        Tuple[pd.DataFrame, StandardScalerPipeline]: Scaled features and fitted pipeline
    """
    # Set default paths
    if features_file is None:
        features_file = PROCESSED_DIR / "features.pkl"
    if scaler_file is None:
        scaler_file = MODELS_DIR / "scaler.pkl"
    
    # Load features
    logger.info(f"Loading features from {features_file}")
    
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    with open(features_file, 'rb') as f:
        X = pickle.load(f)
    
    logger.info(f"Loaded feature matrix: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Initialize and fit scaler
    pipeline = StandardScalerPipeline()
    X_scaled = pipeline.fit_transform(X)
    
    # Save scaler if requested
    if save_scaler:
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pipeline.save(scaler_file)
    
    logger.info("Feature preprocessing completed successfully")
    
    return X_scaled, pipeline


def load_preprocessor(scaler_file: str = None) -> StandardScalerPipeline:
    """
    Load a previously saved preprocessing pipeline.
    
    Args:
        scaler_file: Path to scaler file (defaults to models/scaler.pkl)
        
    Returns:
        StandardScalerPipeline: Loaded preprocessing pipeline
    """
    if scaler_file is None:
        scaler_file = MODELS_DIR / "scaler.pkl"
    
    pipeline = StandardScalerPipeline()
    return pipeline.load(scaler_file)


# Example usage and testing
if __name__ == "__main__":
    """Example usage of the preprocessing pipeline."""
    try:
        # Load and preprocess features
        X_scaled, pipeline = preprocess_features()
        
        print("\n" + "="*60)
        print("FEATURE PREPROCESSING RESULTS")
        print("="*60)
        
        print(f"\nOriginal features shape: {X_scaled.shape}")
        print(f"Features: {list(X_scaled.columns)}")
        
        print(f"\nScaled features statistics:")
        print(X_scaled.describe())
        
        print(f"\nFeature means (should be ~0): {X_scaled.mean().round(6).tolist()}")
        print(f"Feature stds (should be ~1): {X_scaled.std().round(6).tolist()}")
        
        # Test inverse transform
        X_original = pipeline.inverse_transform(X_scaled)
        print(f"\nInverse transform test passed: Original data recovered")
        
        # Test save/load cycle
        test_scaler_file = MODELS_DIR / "test_scaler.pkl" 
        pipeline.save(test_scaler_file)
        
        # Load and test
        pipeline_loaded = load_preprocessor(test_scaler_file)
        X_scaled_loaded = pipeline_loaded.transform(X_original)
        
        # Check if results match
        diff = np.abs(X_scaled.values - X_scaled_loaded.values).max()
        print(f"Save/load test: Max difference = {diff:.10f} (should be ~0)")
        
        # Clean up test file
        test_scaler_file.unlink()
        
        print("\n✅ All preprocessing tests passed!")
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        raise
