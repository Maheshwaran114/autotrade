#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the Day-Type Classifier.
This script trains, validates, and evaluates a machine learning classifier
to predict Bank Nifty trading day types.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_models.day_classifier import DayClassifier
from src.features.feature_engineering import FeatureEngineer
from src.features.feature_selection import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/day_classifier_training.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train the Day-Type Classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/labeled_days.parquet",
        help="Path to the labeled data parquet file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "gradient_boosting", "svm"],
        default="random_forest",
        help="Type of classifier model to use"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/day_classifier.pkl",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Disable probability calibration"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Perform hyperparameter tuning"
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="Perform feature selection"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Directory to save reports and visualizations"
    )
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories if they don't exist"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Create a timestamped subdirectory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.report_dir}/day_classifier_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def train_day_classifier(args, run_dir):
    """Train and evaluate the day type classifier"""
    # Load and process data
    logger.info(f"Loading data from {args.data_path}")
    
    try:
        # Check if data exists
        if not os.path.exists(args.data_path):
            logger.error(f"Data file not found: {args.data_path}")
            return False
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Prepare datasets
        datasets = engineer.prepare_ml_datasets(
            labeled_data_path=args.data_path,
            output_dir="data/processed",
            train_test_split=True,
            test_size=args.test_size,
            scale_features=True
        )
        
        if not datasets or 'X_train' not in datasets:
            logger.error("Failed to prepare datasets")
            return False
        
        X_train = datasets['X_train']
        y_train = datasets['y_train']
        X_test = datasets['X_test']
        y_test = datasets['y_test']
        
        logger.info(f"Dataset prepared: {X_train.shape[0]} training samples, "
                   f"{X_test.shape[0]} testing samples with {X_train.shape[1]} features")
        
        # Save dataset summary
        with open(f"{run_dir}/dataset_summary.txt", "w") as f:
            f.write(f"Dataset summary:\n")
            f.write(f"- Training samples: {X_train.shape[0]}\n")
            f.write(f"- Testing samples: {X_test.shape[0]}\n")
            f.write(f"- Features: {X_train.shape[1]}\n\n")
            f.write(f"Label distribution (training):\n")
            for label, count in y_train.value_counts().items():
                f.write(f"- {label}: {count} ({100*count/len(y_train):.1f}%)\n")
            f.write(f"\nLabel distribution (testing):\n")
            for label, count in y_test.value_counts().items():
                f.write(f"- {label}: {count} ({100*count/len(y_test):.1f}%)\n")
        
        # Perform feature selection if requested
        if args.feature_selection:
            logger.info("Performing feature selection")
            selector = FeatureSelector()
            
            # Select features using combined method
            X_train = selector.select_features(
                X_train, 
                y_train, 
                method='combined',
                params={'corr_threshold': 0.8, 'importance_threshold': 0.005}
            )
            
            X_test = X_test[X_train.columns]
            
            logger.info(f"Selected {X_train.shape[1]} features after feature selection")
            
            # Save selected features list
            with open(f"{run_dir}/selected_features.txt", "w") as f:
                for feature in X_train.columns:
                    f.write(f"{feature}\n")
        
        # Initialize classifier
        classifier = DayClassifier(
            model_type=args.model_type,
            calibrate=not args.no_calibrate
        )
        
        # Perform hyperparameter tuning if requested
        if args.tune:
            logger.info("Performing hyperparameter tuning")
            
            if args.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                }
            elif args.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0]
                }
            elif args.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 0.01],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            
            tuning_results = classifier.hypertune(
                X_train, 
                y_train,
                param_grid=param_grid,
                cv=args.cv_folds
            )
            
            # Save tuning results
            with open(f"{run_dir}/tuning_results.txt", "w") as f:
                f.write(f"Hyperparameter tuning results:\n")
                f.write(f"Best score: {tuning_results['best_score']:.4f}\n")
                f.write(f"Best parameters:\n")
                for param, value in tuning_results['best_params'].items():
                    f.write(f"- {param}: {value}\n")
        
        # Train the model
        logger.info("Training the classifier")
        train_results = classifier.train(X_train, y_train, X_test, y_test)
        
        # Perform cross-validation
        logger.info(f"Performing {args.cv_folds}-fold cross-validation")
        cv_results = classifier.crossvalidate(X_train, y_train, cv=args.cv_folds)
        
        # Generate feature importance visualization
        logger.info("Generating feature importance visualization")
        classifier.plot_feature_importance(
            top_n=min(20, X_train.shape[1]),
            save_path=f"{run_dir}/feature_importance.png"
        )
        
        # Generate confusion matrix visualization
        logger.info("Generating confusion matrix visualization")
        classifier.plot_confusion_matrix(
            y_test, 
            X=X_test,
            normalize=True,
            save_path=f"{run_dir}/confusion_matrix.png"
        )
        
        # Save detailed classification report
        y_pred = classifier.predict(X_test)
        with open(f"{run_dir}/classification_report.txt", "w") as f:
            from sklearn.metrics import classification_report
            f.write(classification_report(y_test, y_pred))
            
            f.write("\nClassification Metrics:\n")
            f.write(f"Training accuracy: {train_results['train_accuracy']:.4f}\n")
            f.write(f"Testing accuracy: {train_results['val_accuracy']:.4f}\n")
            f.write(f"Cross-validation accuracy: {cv_results['cv_accuracy_mean']:.4f} "
                   f"± {cv_results['cv_accuracy_std']:.4f}\n")
        
        # Save the model
        logger.info(f"Saving model to {args.model_path}")
        classifier.save_model(args.model_path)
        
        logger.info("Model training and evaluation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return False


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    run_dir = setup_directories(args)
    logger.info(f"Training run output will be saved to {run_dir}")
    
    # Train the model
    success = train_day_classifier(args, run_dir)
    
    if success:
        logger.info("Training completed successfully")
        return 0
    else:
        logger.error("Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
