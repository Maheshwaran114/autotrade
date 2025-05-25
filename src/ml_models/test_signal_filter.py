#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for testing the Trade Signal Filter model on historical data.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ml_models.signal_filter import SignalFilterModel
from src.features.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/signal_filter_test.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test the Trade Signal Filter model")
    parser.add_argument(
        "--signals-path",
        type=str,
        required=True,
        help="Path to the historical signals CSV/Parquet file"
    )
    parser.add_argument(
        "--market-data-path",
        type=str,
        required=True,
        help="Path to the market data CSV/Parquet file"
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="default_strategy",
        help="Name of the trading strategy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/signal_filter",
        help="Directory to save test reports"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a pre-trained model (if not provided, a new model will be trained)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained/tested model"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="success",
        help="Column name for signal success/failure label"
    )
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}/{args.strategy_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def load_data(args):
    """Load signals and market data"""
    # Load signals data
    if args.signals_path.endswith('.csv'):
        signals_df = pd.read_csv(args.signals_path, parse_dates=['date'])
    elif args.signals_path.endswith('.parquet'):
        signals_df = pd.read_parquet(args.signals_path)
    else:
        raise ValueError(f"Unsupported file format: {args.signals_path}")
        
    # Load market data
    if args.market_data_path.endswith('.csv'):
        market_data_df = pd.read_csv(args.market_data_path, parse_dates=['date'])
    elif args.market_data_path.endswith('.parquet'):
        market_data_df = pd.read_parquet(args.market_data_path)
    else:
        raise ValueError(f"Unsupported file format: {args.market_data_path}")
    
    # Validate that the required label column exists
    if args.label_column not in signals_df.columns:
        raise ValueError(f"Signals data must contain '{args.label_column}' column")
    
    logger.info(f"Loaded {len(signals_df)} signals and {len(market_data_df)} market data points")
    
    return signals_df, market_data_df


def compare_filtered_vs_unfiltered(
    y_true, 
    y_pred, 
    probas, 
    thresholds, 
    strategy_name, 
    save_dir
):
    """Compare performance of filtered vs unfiltered signals"""
    
    results = []
    
    # Baseline (unfiltered)
    baseline_accuracy = y_true.mean()
    results.append({
        'threshold': 0.0, 
        'signals_kept': 1.0,
        'accuracy': baseline_accuracy,
        'profit_factor': 1.0  # Normalized to baseline
    })
    
    # For each threshold
    for threshold in thresholds:
        # Filter signals
        mask = probas >= threshold
        
        # Skip if no signals pass the threshold
        if mask.sum() == 0:
            continue
            
        # Calculate metrics
        signals_kept = mask.mean()
        filtered_accuracy = y_true[mask].mean() if mask.sum() > 0 else 0
        
        # Proxy for profit factor (accuracy improvement ratio)
        profit_factor = filtered_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
        
        results.append({
            'threshold': threshold,
            'signals_kept': signals_kept,
            'accuracy': filtered_accuracy,
            'profit_factor': profit_factor
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(f"{save_dir}/threshold_analysis.csv", index=False)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy vs signals kept
    sns.lineplot(
        data=results_df, 
        x='signals_kept', 
        y='accuracy',
        ax=ax1
    )
    ax1.set_title('Signal Accuracy vs Proportion of Signals Kept')
    ax1.set_xlabel('Proportion of Signals Kept')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    
    # Plot profit factor vs signals kept
    sns.lineplot(
        data=results_df, 
        x='signals_kept', 
        y='profit_factor',
        ax=ax2
    )
    ax2.set_title('Profit Factor vs Proportion of Signals Kept')
    ax2.set_xlabel('Proportion of Signals Kept')
    ax2.set_ylabel('Profit Factor')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/threshold_analysis.png")
    plt.close()
    
    # Find optimal threshold
    optimal_idx = results_df['profit_factor'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_accuracy = results_df.loc[optimal_idx, 'accuracy']
    optimal_signals_kept = results_df.loc[optimal_idx, 'signals_kept']
    optimal_profit_factor = results_df.loc[optimal_idx, 'profit_factor']
    
    logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
    logger.info(f"  Signals kept: {optimal_signals_kept:.2%}")
    logger.info(f"  Filtered accuracy: {optimal_accuracy:.2%}")
    logger.info(f"  Improvement factor: {optimal_profit_factor:.2f}x")
    
    # Plot ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, probas)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{save_dir}/roc_curve.png")
    plt.close()
    
    return optimal_threshold, results_df


def test_signal_filter(args, run_dir):
    """Test the Signal Filter model"""
    try:
        # Load data
        signals_df, market_data_df = load_data(args)
        
        # Create SignalFilterModel
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"Loading existing model from {args.model_path}")
            filter_model = SignalFilterModel(
                strategy_name=args.strategy_name,
                model_path=args.model_path
            )
        else:
            logger.info(f"Creating new model for {args.strategy_name}")
            filter_model = SignalFilterModel(strategy_name=args.strategy_name)
        
        # Prepare data
        X, y = filter_model.prepare_training_data(
            signals_df=signals_df,
            market_data_df=market_data_df,
            label_column=args.label_column
        )
        
        # If no pre-trained model, train a new one
        if not filter_model.model_loaded:
            logger.info("Training new model")
            train_results = filter_model.train(X, y, test_size=args.test_size)
            
            # Log training results
            logger.info(f"Train accuracy: {train_results['train_accuracy']:.4f}")
            logger.info(f"Test accuracy: {train_results['test_accuracy']:.4f}")
            logger.info(f"CV accuracy: {train_results['cv_accuracy_mean']:.4f} ± "
                       f"{train_results['cv_accuracy_std']:.4f}")
        
        # Generate feature importance plot
        filter_model.plot_feature_importance(
            top_n=15,
            save_path=f"{run_dir}/feature_importance.png"
        )
        
        # Run a comprehensive test across different threshold values
        logger.info("Testing model with different threshold values")
        
        # Get predictions and probabilities
        y_pred = filter_model.model.predict(X)
        y_proba = filter_model.model.predict_proba(X)[:, 1]  # Probability of class 1 (success)
        
        # Test different threshold values
        thresholds = np.arange(0.1, 1.0, 0.05)
        optimal_threshold, threshold_results = compare_filtered_vs_unfiltered(
            y_true=y,
            y_pred=y_pred,
            probas=y_proba,
            thresholds=thresholds,
            strategy_name=args.strategy_name,
            save_dir=run_dir
        )
        
        # Save model if requested
        if args.save_model:
            model_path = f"models/signal_filter_{args.strategy_name}.pkl"
            filter_model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Create summary report
        with open(f"{run_dir}/summary_report.md", "w") as f:
            f.write(f"# Signal Filter Model: {args.strategy_name}\n\n")
            f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Data Summary\n\n")
            f.write(f"- Total signals: {len(signals_df)}\n")
            f.write(f"- Success rate (unfiltered): {y.mean():.2%}\n")
            f.write(f"- Features used: {len(filter_model.feature_names)}\n\n")
            
            f.write("## Model Performance\n\n")
            if not filter_model.model_loaded:
                f.write(f"- Training accuracy: {train_results['train_accuracy']:.2%}\n")
                f.write(f"- Test accuracy: {train_results['test_accuracy']:.2%}\n")
                f.write(f"- Cross-validation: {train_results['cv_accuracy_mean']:.2%} ± "
                       f"{train_results['cv_accuracy_std']:.2%}\n\n")
            
            f.write("## Optimal Filter Settings\n\n")
            f.write(f"- Optimal threshold: {optimal_threshold:.2f}\n")
            optimal_row = threshold_results[threshold_results['threshold'] == optimal_threshold].iloc[0]
            f.write(f"- Signals kept: {optimal_row['signals_kept']:.2%}\n")
            f.write(f"- Filtered accuracy: {optimal_row['accuracy']:.2%}\n")
            f.write(f"- Improvement factor: {optimal_row['profit_factor']:.2f}x\n\n")
            
            f.write("## Top Features\n\n")
            if hasattr(filter_model.model, 'feature_importances_'):
                importances = filter_model.model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                top_features = [(filter_model.feature_names[i], importances[i]) 
                               for i in sorted_idx[:10]]
                
                for feature, importance in top_features:
                    f.write(f"- {feature}: {importance:.4f}\n")
        
        logger.info(f"Test completed. Reports saved to {run_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error testing signal filter: {e}", exc_info=True)
        return False


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    run_dir = setup_directories(args)
    
    # Test the signal filter
    success = test_signal_filter(args, run_dir)
    
    if success:
        logger.info("Signal filter test completed successfully")
        return 0
    else:
        logger.error("Signal filter test failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
