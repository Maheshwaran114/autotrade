#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration script to demonstrate how feature engineering integrates with ML models.
This serves as an example for Task 2.4: Train & Validate Day-Type Classifier.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_day_classifier(X_train, y_train, X_test, y_test):
    """
    Train a simple day-type classifier to demonstrate integration
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Trained classifier model
    """
    # Initialize classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    logger.info(f"Training classifier on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Print classification report
    logger.info("Classification report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Create reports directory if not exists
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png")
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('Top 15 Feature Importance')
    plt.savefig("reports/feature_importance.png")
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv("reports/model_feature_importance.csv", index=False)
    
    return clf


def main():
    """Main function to demonstrate integration of feature engineering with ML"""
    try:
        labeled_data_path = "data/processed/labeled_days.parquet"
        
        # Check if labeled data exists
        if not os.path.exists(labeled_data_path):
            logger.error(f"Labeled data file not found at {labeled_data_path}")
            logger.info("Please run 'python src/features/test_features.py' first to generate sample data")
            return 1
            
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Process data and create ML datasets
        logger.info("Preparing ML datasets...")
        datasets = engineer.prepare_ml_datasets(
            labeled_data_path=labeled_data_path,
            output_dir="data/processed",
            train_test_split=True,
            test_size=0.2,
            scale_features=True
        )
        
        if not datasets or 'X_train' not in datasets:
            logger.error("Failed to prepare ML datasets")
            return 1
            
        # Create feature selector
        selector = FeatureSelector()
        
        # Apply feature selection (optional)
        logger.info("Applying feature selection...")
        X_train = selector.select_features(
            datasets['X_train'], 
            datasets['y_train'], 
            method='combined',
            params={'corr_threshold': 0.8, 'importance_threshold': 0.005}
        )
        
        X_test = datasets['X_test'][X_train.columns]
        
        logger.info(f"Selected {X_train.shape[1]} features after feature selection")
        
        # Train and evaluate model
        logger.info("Training day-type classifier...")
        model = train_day_classifier(
            X_train, 
            datasets['y_train'], 
            X_test, 
            datasets['y_test']
        )
        
        # Save model
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/day_classifier_prototype.pkl")
        logger.info("Model saved to models/day_classifier_prototype.pkl")
        
        # Save the selected features list for future reference
        with open("models/selected_features.txt", "w") as f:
            for feature in X_train.columns:
                f.write(f"{feature}\n")
        logger.info("Selected features saved to models/selected_features.txt")
        
        logger.info("Integration demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in integration demonstration: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
