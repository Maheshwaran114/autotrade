#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature selection module for Bank Nifty Options trading system.
This module provides utilities for selecting the most important features
to improve model performance and reduce dimensionality.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Implements various feature selection methods to identify
    the most important features for ML models.
    """

    def __init__(self):
        """Initialize the feature selector"""
        logger.info("FeatureSelector initialized")
    
    def univariate_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top k features based on univariate statistical tests (ANOVA F-value)
        
        Args:
            X: Feature matrix
            y: Target labels
            k: Number of top features to select
            
        Returns:
            Tuple of (selected feature names, scores)
        """
        try:
            # Apply SelectKBest with ANOVA F-value
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            
            # Get feature names and scores
            feature_names = X.columns.values[selected_indices].tolist()
            scores = selector.scores_[selected_indices]
            
            logger.info(f"Selected {len(feature_names)} features using univariate selection")
            
            return feature_names, scores
            
        except Exception as e:
            logger.error(f"Error during univariate feature selection: {e}")
            return [], np.array([])
    
    def rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 10,
        step: float = 0.1
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        
        Args:
            X: Feature matrix
            y: Target labels
            n_features: Number of features to select
            step: Step size for RFE (proportion of features to remove at each step)
            
        Returns:
            List of selected feature names
        """
        try:
            # Initialize estimator (Random Forest works well for feature ranking)
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
            
            # Calculate step as number of features
            step_size = max(1, int(X.shape[1] * step))
            
            # Initialize RFE
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=step_size,
                verbose=1
            )
            
            # Fit RFE
            rfe.fit(X, y)
            
            # Get selected feature names
            selected_features = X.columns[rfe.support_].tolist()
            
            logger.info(f"Selected {len(selected_features)} features using RFE")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error during RFE feature selection: {e}")
            return []
    
    def importance_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.01
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features based on importance scores from a tree-based model
        
        Args:
            X: Feature matrix
            y: Target labels
            threshold: Minimum importance threshold
            
        Returns:
            Tuple of (selected feature names, feature importance DataFrame)
        """
        try:
            # Initialize Random Forest
            rf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit model
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create DataFrame with features and importance scores
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # Select features above threshold
            selected_df = importance_df[importance_df['importance'] > threshold]
            selected_features = selected_df['feature'].tolist()
            
            logger.info(f"Selected {len(selected_features)} features using importance threshold {threshold}")
            
            return selected_features, importance_df
            
        except Exception as e:
            logger.error(f"Error during importance-based feature selection: {e}")
            return [], pd.DataFrame()
    
    def correlation_based_selection(
        self,
        X: pd.DataFrame,
        correlation_threshold: float = 0.8
    ) -> List[str]:
        """
        Remove highly correlated features
        
        Args:
            X: Feature matrix
            correlation_threshold: Threshold to identify highly correlated features
            
        Returns:
            List of selected feature names (with correlated features removed)
        """
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Create upper triangle mask
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation above threshold
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            
            # Get remaining features
            selected_features = [col for col in X.columns if col not in to_drop]
            
            logger.info(f"Selected {len(selected_features)} features by removing {len(to_drop)} highly correlated features")
            logger.info(f"Removed features: {to_drop}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error during correlation-based feature selection: {e}")
            return []
    
    def pca_transform(
        self,
        X: pd.DataFrame,
        n_components: int = None,
        variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, float]:
        """
        Transform features using PCA
        
        Args:
            X: Feature matrix
            n_components: Number of components to keep (if None, use variance_threshold)
            variance_threshold: Minimum explained variance to retain
            
        Returns:
            Tuple of (transformed data, explained variance ratio)
        """
        try:
            # Initialize PCA
            if n_components is None:
                pca = PCA(n_components=variance_threshold, svd_solver='full')
            else:
                pca = PCA(n_components=n_components)
            
            # Fit and transform
            transformed_data = pca.fit_transform(X)
            
            # Get explained variance
            explained_variance = pca.explained_variance_ratio_.sum()
            
            logger.info(f"PCA transformed data to {transformed_data.shape[1]} components with {explained_variance:.3f} explained variance")
            
            return transformed_data, explained_variance
            
        except Exception as e:
            logger.error(f"Error during PCA transformation: {e}")
            return np.array([]), 0.0
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature names and importance scores
            top_n: Number of top features to plot
            save_path: Path to save the plot (if None, just display)
        """
        try:
            # Take top N features
            plot_df = importance_df.head(top_n)
            
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar plot
            sns.barplot(x='importance', y='feature', data=plot_df)
            
            plt.title(f'Top {top_n} Feature Importance')
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
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        method: str = 'combined',
        params: Dict = None
    ) -> pd.DataFrame:
        """
        Perform feature selection using specified method
        
        Args:
            X: Feature matrix
            y: Target labels (required for some methods)
            method: Selection method ('correlation', 'importance', 'univariate', 'rfe', 'pca', 'combined')
            params: Parameters for the specified method
            
        Returns:
            DataFrame with selected features
        """
        if params is None:
            params = {}
            
        try:
            if method == 'correlation':
                threshold = params.get('threshold', 0.8)
                selected_features = self.correlation_based_selection(X, threshold)
                return X[selected_features]
                
            elif method == 'importance':
                if y is None:
                    logger.error("Target labels (y) required for importance-based selection")
                    return X
                    
                threshold = params.get('threshold', 0.01)
                selected_features, _ = self.importance_based_selection(X, y, threshold)
                return X[selected_features]
                
            elif method == 'univariate':
                if y is None:
                    logger.error("Target labels (y) required for univariate selection")
                    return X
                    
                k = params.get('k', min(10, X.shape[1]))
                selected_features, _ = self.univariate_selection(X, y, k)
                return X[selected_features]
                
            elif method == 'rfe':
                if y is None:
                    logger.error("Target labels (y) required for RFE selection")
                    return X
                    
                n_features = params.get('n_features', min(10, X.shape[1]))
                step = params.get('step', 0.1)
                selected_features = self.rfe_selection(X, y, n_features, step)
                return X[selected_features]
                
            elif method == 'pca':
                n_components = params.get('n_components', None)
                variance_threshold = params.get('variance_threshold', 0.95)
                
                transformed_data, _ = self.pca_transform(X, n_components, variance_threshold)
                
                # Convert to DataFrame
                columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
                return pd.DataFrame(transformed_data, columns=columns, index=X.index)
                
            elif method == 'combined':
                # First remove highly correlated features
                corr_threshold = params.get('corr_threshold', 0.8)
                X_filtered = X[self.correlation_based_selection(X, corr_threshold)]
                
                if y is None:
                    logger.warning("Target labels (y) required for complete combined selection, skipping importance filtering")
                    return X_filtered
                
                # Then select based on importance
                importance_threshold = params.get('importance_threshold', 0.01)
                selected_features, _ = self.importance_based_selection(X_filtered, y, importance_threshold)
                
                return X_filtered[selected_features]
                
            else:
                logger.error(f"Unknown feature selection method: {method}")
                return X
                
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            return X


# For testing purposes
if __name__ == "__main__":
    import joblib
    import os
    import seaborn as sns
    
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create feature selector
    selector = FeatureSelector()
    
    # Try to load processed data
    try:
        X_path = "data/processed/features.pkl"
        y_path = "data/processed/labels.pkl"
        
        if os.path.exists(X_path) and os.path.exists(y_path):
            print(f"Loading data from {X_path} and {y_path}")
            X = joblib.load(X_path)
            y = joblib.load(y_path)
            
            print(f"Loaded data: {X.shape} features, {y.shape} labels")
            
            # Test correlation-based selection
            print("\nTesting correlation-based selection...")
            selected_features = selector.correlation_based_selection(X, 0.8)
            print(f"Selected {len(selected_features)} features")
            
            # If target labels available
            print("\nTesting importance-based selection...")
            selected_features, importance_df = selector.importance_based_selection(X, y, 0.01)
            print(f"Selected {len(selected_features)} features")
            print("Top 5 features by importance:")
            print(importance_df.head(5))
            
            # Test combined selection
            print("\nTesting combined selection...")
            X_selected = selector.select_features(X, y, method='combined')
            print(f"Selected {X_selected.shape[1]} features using combined method")
            
        else:
            print("No processed data found. Cannot test feature selection.")
            
    except Exception as e:
        print(f"Error testing feature selector: {e}")
