# src/features/feature_engineering.py

"""
Compute and save feature matrix X and label vector y for day-type classification.
Inputs:  data/processed/labeled_days.parquet
Outputs: data/processed/features.pkl, data/processed/labels.pkl
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. PARAMETERS
IV_CHANGE_WINDOW = 1    # days to compute IV change
GAP_FEATURE = True
OR_FEATURE = True
VOLATILITY_FEATURE = True
IV_FEATURE = True
SECTOR_STRENGTH_FEATURE = True

def load_labeled_days(path="data/processed/labeled_days.parquet"):
    """Load labeled trading days data."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")

def compute_iv_change(df):
    """Compute IV percent change from previous day."""
    # Percent change of atm_iv from previous day (using available column)
    df["iv_change"] = df["atm_iv"].pct_change(periods=IV_CHANGE_WINDOW).fillna(0)
    return df

def assemble_features(df):
    """Assemble feature matrix from labeled data."""
    feats = pd.DataFrame(index=df.index)
    
    # 2. Gap % = (open - prev_close)/prev_close
    if GAP_FEATURE:
        feats["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    
    # 3. Opening range width (using available or_range_pct, normalized)
    if OR_FEATURE:
        feats["or_width"] = df["or_range_pct"]  # Already normalized as percentage of close
    
    # 4. Realized volatility (using available volatility column)
    if VOLATILITY_FEATURE:
        feats["intraday_volatility"] = df["volatility"]
    
    # 5. IV percentile and change
    if IV_FEATURE:
        feats["iv_pct"] = df["iv_percentile"]
        feats["iv_change"] = df["iv_change"]
    
    # 6. Sector Strength (if available - fallback to volume ratio)
    if SECTOR_STRENGTH_FEATURE:
        if "sector_strength" in df.columns:
            feats["sector_strength"] = df["sector_strength"]
        else:
            # Use volume ratio as proxy for sector strength
            feats["sector_strength"] = df["volume_ratio"]
    
    # 7. Additional features from available data
    feats["daily_return"] = df["daily_return"]
    feats["price_range"] = df["price_range"]
    feats["body_size"] = df["body_size"]
    feats["or_breakout_direction"] = df["or_breakout_direction"]
    feats["atm_iv"] = df["atm_iv"]
    feats["volume_ratio"] = df["volume_ratio"]
    
    # 8. Fill any remaining NaNs
    feats = feats.fillna(0)
    
    return feats

def assemble_labels(df):
    """Extract labels from labeled data."""
    # Use regime column as labels
    return df["regime"].copy()

def save_data(X, y, feature_path="data/processed/features.pkl", label_path="data/processed/labels.pkl"):
    """Save feature matrix and labels to pickle files."""
    import pickle
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Use pickle instead of joblib for better compatibility
    with open(feature_path, 'wb') as f:
        pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(label_path, 'wb') as f:
        pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    """Main feature engineering pipeline."""
    print("Starting feature engineering...")
    
    # Load
    print("Loading labeled days data...")
    df = load_labeled_days()
    print(f"Loaded {len(df)} days of data")
    
    # Compute derived features
    print("Computing IV change...")
    df = compute_iv_change(df)
    
    # Assemble
    print("Assembling features...")
    X = assemble_features(df)
    print("Assembling labels...")
    y = assemble_labels(df)
    
    # Save
    print("Saving features and labels...")
    save_data(X, y)
    
    # Report
    print(f"\nFeature Engineering Complete!")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print("Feature columns:", list(X.columns))
    print("\nSample features:")
    print(X.head())
    print("\nSample labels:")
    print(y.head())
    print(f"\nLabel distribution:")
    print(y.value_counts())
    
    # Append to docs
    print("\nUpdating documentation...")
    with open("docs/phase2_report.md", "a") as f:
        f.write(f"\n\n## 2.3 Feature Engineering\n")
        f.write(f"- Features saved: data/processed/features.pkl ({X.shape[0]}×{X.shape[1]})\n")
        f.write(f"- Labels saved:   data/processed/labels.pkl ({y.shape[0]})\n")
        f.write("- Feature columns: " + ", ".join(X.columns) + "\n")
        f.write(f"- Label distribution: {dict(y.value_counts())}\n")
        f.write("- Sample features:\n")
        f.write("```\n")
        f.write(str(X.head()))
        f.write("\n```\n")
        f.write("- Sample labels:\n")
        f.write("```\n")
        f.write(str(y.head()))
        f.write("\n```\n")
    print("Appended Phase 2.3 summary to docs/phase2_report.md")

if __name__ == "__main__":
    main()
