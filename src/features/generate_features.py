#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate and analyze features for the Bank Nifty Options trading system.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_features(features_df, labels_series, output_dir):
    """
    Analyze features and their relationship with labels
    
    Args:
        features_df: DataFrame of features
        labels_series: Series of labels
        output_dir: Directory to save analysis outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    logger.info("Calculating basic statistics...")
    stats = features_df.describe()
    stats.to_csv(f"{output_dir}/feature_statistics.csv")
    
    # Correlation matrix
    logger.info("Calculating correlation matrix...")
    corr_matrix = features_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # Top correlated features
    logger.info("Finding top correlated features...")
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than 0.8
    high_corr = [(upper.columns[i], upper.columns[j], upper.iloc[i, j]) 
                for i in range(len(upper.columns)) 
                for j in range(len(upper.columns)) 
                if i < j and abs(upper.iloc[i, j]) > 0.8]
    
    if high_corr:
        logger.info("High correlation pairs (>0.8):")
        high_corr_df = pd.DataFrame(high_corr, columns=['Feature1', 'Feature2', 'Correlation'])
        high_corr_df = high_corr_df.sort_values('Correlation', ascending=False)
        high_corr_df.to_csv(f"{output_dir}/high_correlation_features.csv", index=False)
        for f1, f2, c in high_corr:
            logger.info(f"  {f1} - {f2}: {c:.3f}")
    else:
        logger.info("No highly correlated feature pairs found.")
    
    # Feature distribution by label
    logger.info("Analyzing feature distributions by label...")
    # Combine features and labels
    df = pd.concat([features_df, labels_series], axis=1)
    
    # Select key features for distribution plots (top 5 based on variance)
    variance = features_df.var().sort_values(ascending=False)
    top_features = variance.index[:5].tolist()
    
    for feature in top_features:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='label', y=feature, data=df)
        plt.title(f"{feature} Distribution by Trading Regime")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_by_label.png")
        plt.close()
    
    # Create a summary markdown file
    with open(f"{output_dir}/feature_analysis_summary.md", "w") as f:
        f.write("# Feature Analysis Summary\n\n")
        f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"Total features: {features_df.shape[1]}\n")
        f.write(f"Total samples: {features_df.shape[0]}\n\n")
        
        f.write("## Label Distribution\n\n")
        label_dist = labels_series.value_counts()
        for label, count in label_dist.items():
            f.write(f"- {label}: {count} samples ({count/len(labels_series)*100:.1f}%)\n")
        
        f.write("\n## Top Features by Variance\n\n")
        for i, (feature, var) in enumerate(variance.head(10).items()):
            f.write(f"{i+1}. **{feature}**: {var:.6f}\n")
        
        f.write("\n## Highly Correlated Feature Pairs\n\n")
        if high_corr:
            for f1, f2, c in high_corr:
                f.write(f"- {f1} - {f2}: {c:.3f}\n")
        else:
            f.write("No highly correlated feature pairs found.\n")
    
    logger.info(f"Feature analysis complete. Results saved to {output_dir}")


def main():
    """Main function to run feature engineering and analysis"""
    try:
        labeled_data_path = "data/processed/labeled_days.parquet"
        output_dir = "data/processed"
        analysis_dir = "reports/feature_analysis"
        
        logger.info(f"Starting feature engineering process using {labeled_data_path}")
        
        # Create feature engineer
        engineer = FeatureEngineer(scaler_type='standard')
        
        # Prepare datasets
        datasets = engineer.prepare_ml_datasets(
            labeled_data_path=labeled_data_path,
            output_dir=output_dir,
            train_test_split=True,
            test_size=0.2,
            scale_features=True
        )
        
        if datasets:
            logger.info(f"Successfully created features with shape: {datasets['X'].shape}")
            
            # Analyze features
            analyze_features(
                features_df=datasets['X'],
                labels_series=datasets['y'],
                output_dir=analysis_dir
            )
            
            logger.info("Feature engineering and analysis complete.")
        else:
            logger.error("Failed to create features. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
