#!/usr/bin/env python3
"""
Task 2.2: Label Trading Regimes
Execute day labeling on collected historical data
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_ingest.label_days import DayLabeler

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    banknifty_path = processed_dir / 'banknifty_index.parquet'
    options_path = processed_dir / 'banknifty_options_chain.parquet'
    output_path = processed_dir / 'labeled_days.parquet'
    
    print("=" * 60)
    print("TASK 2.2: LABEL TRADING REGIMES")
    print("=" * 60)
    
    # Check if input files exist
    if not banknifty_path.exists():
        print(f"❌ Error: Bank Nifty data not found at {banknifty_path}")
        print("Please run Task 2.1 first to collect historical data.")
        return False
    
    print(f"📊 Loading Bank Nifty data from: {banknifty_path}")
    
    # Initialize day labeler
    labeler = DayLabeler()
    
    # Load data
    banknifty_df, options_df = labeler.load_data(
        str(banknifty_path), 
        str(options_path) if options_path.exists() else None
    )
    
    if banknifty_df.empty:
        print("❌ Error: Failed to load Bank Nifty data")
        return False
    
    print(f"✅ Loaded {len(banknifty_df)} Bank Nifty records")
    if options_df is not None and not options_df.empty:
        print(f"✅ Loaded {len(options_df)} option records")
    
    # Calculate features and classify days
    print("\n🔧 Calculating features and classifying trading days...")
    labeled_df = labeler.classify_days(banknifty_df, options_df)
    
    if labeled_df.empty:
        print("❌ Error: Failed to classify trading days")
        return False
    
    # Display results
    print(f"\n✅ Successfully classified {len(labeled_df)} trading days")
    
    # Show distribution of day types
    type_counts = labeled_df['day_type'].value_counts()
    total_days = len(labeled_df)
    
    print("\n📈 Trading Day Type Distribution:")
    print("-" * 40)
    for day_type, count in type_counts.items():
        percentage = count / total_days * 100
        print(f"{day_type:12s}: {count:3d} days ({percentage:5.1f}%)")
    
    # Show sample of labeled data
    print(f"\n📋 Sample of labeled data:")
    print("-" * 80)
    sample_cols = ['date', 'open', 'high', 'low', 'close', 'open_to_close_return', 
                   'high_low_range', 'day_type']
    if all(col in labeled_df.columns for col in sample_cols):
        print(labeled_df[sample_cols].head(10).to_string(index=False, float_format='%.2f'))
    else:
        print(labeled_df.head(10).to_string(index=False, float_format='%.2f'))
    
    # Save results
    print(f"\n💾 Saving labeled data to: {output_path}")
    labeled_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Task 2.2 completed successfully!")
    print(f"📁 Output file: {output_path}")
    print(f"📊 Total trading days classified: {len(labeled_df)}")
    
    # Create summary for documentation
    summary = f"""
# Task 2.2: Label Trading Regimes - Summary

## Classification Results
- **Total trading days**: {len(labeled_df)}
- **Classification method**: Rule-based algorithm using price action patterns

## Day Type Distribution
"""
    for day_type, count in type_counts.items():
        percentage = count / total_days * 100
        summary += f"- **{day_type}**: {count} days ({percentage:.1f}%)\n"
    
    summary += f"""
## Features Used for Classification
- Open-to-close return percentage
- High-low range percentage
- Gap percentage (open vs previous close)
- First 30-minute range analysis
- Volume patterns (when available)

## Output Files
- **Labeled data**: `{output_path.relative_to(base_dir)}`
- **Schema**: date, OHLCV data, calculated features, day_type classification

## Status: ✅ COMPLETED
Task 2.2 successfully completed on {labeled_df['date'].min().strftime('%Y-%m-%d')} to {labeled_df['date'].max().strftime('%Y-%m-%d')}
"""
    
    # Save summary
    summary_path = base_dir / 'docs' / 'task2_2_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"📄 Summary saved to: {summary_path}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
