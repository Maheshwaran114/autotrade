#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation script for the Day-Type Classifier.
This script evaluates a trained classifier on validation data
and generates performance reports.
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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_models.day_classifier import DayClassifier
from src.features.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/day_classifier_validation.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Validate the Day-Type Classifier")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/day_classifier.pkl",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/validation_data.parquet",
        help="Path to the validation data parquet file (if not provided, will use test split from processed data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/validation",
        help="Directory to save validation reports"
    )
    
    return parser.parse_args()


def setup_directories(output_dir):
    """Create necessary directories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    return output_dir


def validate_classifier(args):
    """Validate the trained day type classifier"""
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return False
    
    try:
        # Load the trained model
        logger.info(f"Loading model from {args.model_path}")
        classifier = DayClassifier(model_path=args.model_path)
        
        # Load validation data
        validation_data = None
        
        # First try to load specified validation data
        if os.path.exists(args.data_path):
            logger.info(f"Loading validation data from {args.data_path}")
            df = pd.read_parquet(args.data_path)
            
            # Check if data contains required columns
            if 'label' in df.columns:
                validation_data = df
            else:
                logger.warning(f"Validation data does not contain 'label' column, will use test split from processed data")
        
        # If validation data not available, use test split from processed data
        if validation_data is None:
            test_features_path = "data/processed/X_test.pkl"
            test_labels_path = "data/processed/y_test.pkl"
            
            if os.path.exists(test_features_path) and os.path.exists(test_labels_path):
                logger.info(f"Using test split from processed data")
                import joblib
                X_test = joblib.load(test_features_path)
                y_test = joblib.load(test_labels_path)
            else:
                logger.error("No validation data found and no test split available")
                return False
        else:
            # Process validation data
            engineer = FeatureEngineer()
            features_df = engineer.compute_features(validation_data)
            
            if 'label' not in features_df.columns:
                logger.error("Label column not found in processed validation data")
                return False
                
            y_test = features_df['label']
            
            # Select feature columns (exclude label and any other non-feature columns)
            non_feature_cols = ['label', 'day_of_week']  # Add any other columns to exclude
            X_test = features_df.drop(columns=non_feature_cols, errors='ignore')
            
            # Scale features
            X_test = engineer.transform_features(X_test)
        
        # Make predictions
        logger.info("Making predictions on validation data")
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f"Validation metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classifier.DAY_TYPES,
            yticklabels=classifier.DAY_TYPES
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/confusion_matrix.png")
        plt.close()
        
        # Generate normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=classifier.DAY_TYPES,
            yticklabels=classifier.DAY_TYPES
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/confusion_matrix_normalized.png")
        plt.close()
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Save report to CSV
        report_df.to_csv(f"{args.output_dir}/classification_report.csv")
        
        # Save detailed report
        with open(f"{args.output_dir}/validation_report.txt", "w") as f:
            f.write("# Day-Type Classifier Validation Report\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Validation data: {args.data_path if os.path.exists(args.data_path) else 'Used test split'}\n")
            f.write(f"Number of samples: {len(y_test)}\n\n")
            
            f.write("## Overall Metrics\n\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")
            
            f.write("## Class-wise Metrics\n\n")
            f.write(classification_report(y_test, y_pred))
            
            f.write("\n## Validation Sample Distribution\n\n")
            for label, count in y_test.value_counts().items():
                f.write(f"- {label}: {count} ({100*count/len(y_test):.1f}%)\n")
            
            f.write("\n## Confusion Matrix\n\n")
            f.write(f"See {args.output_dir}/confusion_matrix.png\n")
        
        # Create a classification errors analysis
        errors_idx = y_test.index[y_test != y_pred]
        errors_df = pd.DataFrame({
            'True': y_test[errors_idx],
            'Predicted': y_pred[errors_idx]
        })
        
        # Add probabilities for misclassifications
        for i, day_type in enumerate(classifier.DAY_TYPES):
            errors_df[f'Prob_{day_type}'] = y_pred_proba[errors_idx.to_numpy(), i]
        
        # Save errors to CSV for analysis
        errors_df.to_csv(f"{args.output_dir}/misclassifications.csv")
        
        # Create error distribution plot
        plt.figure(figsize=(12, 8))
        errors_count = errors_df.groupby(['True', 'Predicted']).size().unstack(fill_value=0)
        sns.heatmap(errors_count, annot=True, fmt='d', cmap='Reds')
        plt.title('Misclassification Distribution')
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/misclassification_distribution.png")
        plt.close()
        
        logger.info(f"Validation completed. Reports saved to {args.output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return False


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Validate the classifier
    success = validate_classifier(args)
    
    if success:
        logger.info("Validation completed successfully")
        return 0
    else:
        logger.error("Validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
