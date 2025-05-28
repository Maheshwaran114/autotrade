"""
Phase 2.2: Label Trading Regimes

This module classifies each trading day into one of five market regimes:
- Trend
- RangeBound
- Event
- MildBias
- Momentum

Based on index and option features with defined thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load index and options data from processed parquet files.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (index_data, options_data)
    """
    logger.info("Loading processed data files...")
    
    # Load index data
    index_file = PROCESSED_DIR / "banknifty_index.parquet"
    if not index_file.exists():
        raise FileNotFoundError(f"Index data file not found: {index_file}")
    
    index_data = pd.read_parquet(index_file)
    logger.info(f"Loaded index data: {len(index_data)} records")
    
    # Load options data
    options_file = PROCESSED_DIR / "banknifty_options_chain.parquet"
    if not options_file.exists():
        raise FileNotFoundError(f"Options data file not found: {options_file}")
    
    options_data = pd.read_parquet(options_file)
    logger.info(f"Loaded options data: {len(options_data)} records")
    
    return index_data, options_data


def compute_daily_metrics(index_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily OHLCV metrics, VWAP, returns, and volatility from minute-level data.
    
    Args:
        index_data: Minute-level index data with columns [date, open, high, low, close, volume]
    
    Returns:
        pd.DataFrame: Daily metrics with columns [date, open, high, low, close, volume, vwap, 
                     daily_return, volatility, prev_close]
    """
    logger.info("Computing daily metrics from minute-level data...")
    
    # Convert date to datetime if needed and extract date part
    if isinstance(index_data['date'].iloc[0], str):
        index_data['date'] = pd.to_datetime(index_data['date'])
    
    # Extract date part for grouping
    index_data['trading_date'] = index_data['date'].dt.date
    
    # Group by trading date and compute daily metrics
    daily_metrics = []
    
    for date, group in index_data.groupby('trading_date'):
        # Sort by time to ensure proper order
        group = group.sort_values('date')
        
        # OHLCV
        open_price = group['open'].iloc[0]  # First minute's open
        high_price = group['high'].max()
        low_price = group['low'].min()
        close_price = group['close'].iloc[-1]  # Last minute's close
        total_volume = group['volume'].sum()
        
        # VWAP calculation
        vwap = (group['close'] * group['volume']).sum() / total_volume if total_volume > 0 else close_price
        
        daily_metrics.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': total_volume,
            'vwap': vwap
        })
    
    daily_df = pd.DataFrame(daily_metrics)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)
    
    # Calculate returns and volatility
    daily_df['prev_close'] = daily_df['close'].shift(1)
    daily_df['daily_return'] = (daily_df['close'] - daily_df['prev_close']) / daily_df['prev_close']
    
    # Calculate 5-day rolling volatility (requires minimum 2 periods for meaningful calculation)
    daily_df['volatility'] = daily_df['daily_return'].rolling(window=5, min_periods=2).std()
    
    logger.info(f"Computed daily metrics for {len(daily_df)} trading days")
    return daily_df


def extract_opening_range_features(index_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract opening range features from the first 30 minutes of trading.
    
    Args:
        index_data: Minute-level index data
    
    Returns:
        pd.DataFrame: Opening range features with columns [date, or_high, or_low, or_range, 
                     or_volume, or_breakout_direction]
    """
    logger.info("Extracting opening range features (first 30 minutes)...")
    
    # Convert date to datetime if needed
    if isinstance(index_data['date'].iloc[0], str):
        index_data['date'] = pd.to_datetime(index_data['date'])
    
    # Extract date and time components
    index_data['trading_date'] = index_data['date'].dt.date
    index_data['time'] = index_data['date'].dt.time
    
    opening_features = []
    
    for date, group in index_data.groupby('trading_date'):
        # Sort by time
        group = group.sort_values('date')
        
        # Get first 30 minutes (assuming trading starts at 9:15 AM)
        start_time = group['date'].iloc[0].replace(hour=9, minute=15, second=0, microsecond=0)
        end_time = start_time + pd.Timedelta(minutes=30)
        
        # Filter for opening range period
        or_data = group[(group['date'] >= start_time) & (group['date'] <= end_time)]
        
        if len(or_data) == 0:
            continue
        
        # Calculate opening range metrics
        or_high = or_data['high'].max()
        or_low = or_data['low'].min()
        or_range = or_high - or_low
        or_volume = or_data['volume'].sum()
        
        # Determine breakout direction (comparing close vs opening range)
        day_close = group['close'].iloc[-1]
        if day_close > or_high:
            or_breakout_direction = 1  # Upward breakout
        elif day_close < or_low:
            or_breakout_direction = -1  # Downward breakout
        else:
            or_breakout_direction = 0  # No breakout
        
        opening_features.append({
            'date': date,
            'or_high': or_high,
            'or_low': or_low,
            'or_range': or_range,
            'or_volume': or_volume,
            'or_breakout_direction': or_breakout_direction
        })
    
    or_df = pd.DataFrame(opening_features)
    logger.info(f"Extracted opening range features for {len(or_df)} trading days")
    return or_df


def calculate_iv_percentiles(options_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate IV percentiles and ATM IV from options data.
    Uses expanding window for true historical percentiles.
    
    Args:
        options_data: Options chain data
    
    Returns:
        pd.DataFrame: IV metrics with columns [date, iv_percentile, atm_iv, iv_rank]
    """
    logger.info("Calculating IV percentiles from options data...")
    
    # Convert date to datetime if needed
    if isinstance(options_data['date'].iloc[0], str):
        options_data['date'] = pd.to_datetime(options_data['date']).dt.date
    
    iv_metrics = []
    historical_atm_ivs = []  # Store historical ATM IVs for expanding percentile calculation
    
    for date, group in options_data.groupby('date'):
        if len(group) == 0:
            continue
        
        # Get spot price for the day
        spot_price = group['spot_price'].iloc[0]
        
        # Find ATM options (closest to spot price)
        group['strike_diff'] = abs(group['strike'] - spot_price)
        atm_strikes = group.nsmallest(2, 'strike_diff')['strike'].unique()
        atm_options = group[group['strike'].isin(atm_strikes)]
        
        # Calculate ATM IV (average of CE and PE)
        if len(atm_options) > 0:
            atm_iv = atm_options['iv'].mean()
        else:
            atm_iv = group['iv'].median()  # Fallback to median IV
        
        # Add current ATM IV to historical data
        historical_atm_ivs.append(atm_iv)
        
        # Calculate IV percentile using expanding window (current vs all historical)
        if len(historical_atm_ivs) == 1:
            iv_percentile = 0.5  # First day defaults to 50th percentile
        else:
            # Calculate percentile of current ATM IV relative to all historical ATM IVs
            iv_percentile = len([iv for iv in historical_atm_ivs[:-1] if iv <= atm_iv]) / (len(historical_atm_ivs) - 1)
        
        # Calculate IV rank (current ATM IV relative to all option IVs for this day)
        all_ivs = group['iv'].values
        iv_rank = np.percentile(all_ivs, 50)  # Median IV as rank reference
        
        iv_metrics.append({
            'date': date,
            'iv_percentile': iv_percentile,
            'atm_iv': atm_iv,
            'iv_rank': iv_rank
        })
    
    iv_df = pd.DataFrame(iv_metrics)
    logger.info(f"Calculated IV metrics for {len(iv_df)} trading days")
    return iv_df


def label_regimes(daily_metrics: pd.DataFrame, opening_features: pd.DataFrame, 
                 iv_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Apply regime labeling logic based on specified thresholds.
    
    Regime Classification Rules:
    - Trend: High momentum + directional bias + OR breakout in same direction
    - RangeBound: Low volatility + no significant OR breakout + low IV percentile
    - Event: High volatility + high IV + large price gaps
    - MildBias: Moderate momentum + small OR range + medium volatility
    - Momentum: Very high momentum + strong OR breakout + high volume
    
    Args:
        daily_metrics: Daily OHLCV and derived metrics
        opening_features: Opening range features
        iv_metrics: IV percentiles and ATM IV
    
    Returns:
        pd.DataFrame: Labeled data with regime classifications
    """
    logger.info("Applying regime labeling logic...")
    
    # Merge all features
    labeled_data = daily_metrics.copy()
    
    # Ensure consistent date types for merging
    labeled_data['date'] = pd.to_datetime(labeled_data['date'])
    opening_features['date'] = pd.to_datetime(opening_features['date'])
    iv_metrics['date'] = pd.to_datetime(iv_metrics['date'])
    
    labeled_data = labeled_data.merge(opening_features, on='date', how='left')
    labeled_data = labeled_data.merge(iv_metrics, on='date', how='left')
    
    # Fill missing values using pandas compatible method
    labeled_data = labeled_data.ffill().bfill()
    
    # Calculate additional features for classification
    labeled_data['price_range'] = (labeled_data['high'] - labeled_data['low']) / labeled_data['close']
    labeled_data['body_size'] = abs(labeled_data['close'] - labeled_data['open']) / labeled_data['close']
    labeled_data['volume_ma'] = labeled_data['volume'].rolling(window=10, min_periods=1).mean()
    labeled_data['volume_ratio'] = labeled_data['volume'] / labeled_data['volume_ma']
    labeled_data['or_range_pct'] = labeled_data['or_range'] / labeled_data['close']
    
    # Define thresholds
    HIGH_VOLATILITY_THRESHOLD = 0.02  # 2%
    LOW_VOLATILITY_THRESHOLD = 0.008  # 0.8%
    HIGH_MOMENTUM_THRESHOLD = 0.015   # 1.5% daily return
    MODERATE_MOMENTUM_THRESHOLD = 0.008  # 0.8% daily return
    HIGH_IV_THRESHOLD = 0.7  # 70th percentile
    LOW_IV_THRESHOLD = 0.3   # 30th percentile
    HIGH_VOLUME_THRESHOLD = 1.5  # 1.5x average volume
    LARGE_OR_RANGE_THRESHOLD = 0.012  # 1.2% of price
    SMALL_OR_RANGE_THRESHOLD = 0.006  # 0.6% of price
    
    # Initialize regime column
    labeled_data['regime'] = 'MildBias'  # Default
    
    # Apply classification rules
    for idx, row in labeled_data.iterrows():
        abs_return = abs(row['daily_return']) if not pd.isna(row['daily_return']) else 0
        volatility = row['volatility'] if not pd.isna(row['volatility']) else 0
        iv_percentile = row['iv_percentile'] if not pd.isna(row['iv_percentile']) else 0.5
        volume_ratio = row['volume_ratio'] if not pd.isna(row['volume_ratio']) else 1
        or_range_pct = row['or_range_pct'] if not pd.isna(row['or_range_pct']) else 0
        or_breakout = row['or_breakout_direction'] if not pd.isna(row['or_breakout_direction']) else 0
        
        # Momentum: Very high momentum + strong OR breakout + high volume
        if (abs_return > HIGH_MOMENTUM_THRESHOLD and 
            abs(or_breakout) == 1 and 
            volume_ratio > HIGH_VOLUME_THRESHOLD):
            labeled_data.at[idx, 'regime'] = 'Momentum'
        
        # Event: High volatility + high IV + large price gaps
        elif (volatility > HIGH_VOLATILITY_THRESHOLD and 
              iv_percentile > HIGH_IV_THRESHOLD and 
              or_range_pct > LARGE_OR_RANGE_THRESHOLD):
            labeled_data.at[idx, 'regime'] = 'Event'
        
        # Trend: High momentum + directional bias + OR breakout in same direction
        elif (abs_return > MODERATE_MOMENTUM_THRESHOLD and 
              or_breakout != 0 and 
              np.sign(row['daily_return'] if not pd.isna(row['daily_return']) else 0) == or_breakout):
            labeled_data.at[idx, 'regime'] = 'Trend'
        
        # RangeBound: Low volatility + no significant OR breakout + low IV percentile
        elif (volatility < LOW_VOLATILITY_THRESHOLD and 
              or_breakout == 0 and 
              iv_percentile < LOW_IV_THRESHOLD and 
              or_range_pct < SMALL_OR_RANGE_THRESHOLD):
            labeled_data.at[idx, 'regime'] = 'RangeBound'
        
        # MildBias: Default for moderate conditions
        # (already initialized as default)
    
    # Log regime distribution
    regime_counts = labeled_data['regime'].value_counts()
    logger.info(f"Regime distribution:\n{regime_counts}")
    
    return labeled_data


def main():
    """
    Main function to orchestrate the labeling process.
    """
    logger.info("Starting Phase 2.2: Label Trading Regimes")
    
    try:
        # Load data
        index_data, options_data = load_data()
        
        # Compute daily metrics
        daily_metrics = compute_daily_metrics(index_data)
        
        # Extract opening range features
        opening_features = extract_opening_range_features(index_data)
        
        # Calculate IV percentiles
        iv_metrics = calculate_iv_percentiles(options_data)
        
        # Label regimes
        labeled_data = label_regimes(daily_metrics, opening_features, iv_metrics)
        
        # Save output
        output_file = PROCESSED_DIR / "labeled_days.parquet"
        labeled_data.to_parquet(output_file, index=False)
        logger.info(f"Saved labeled data to: {output_file}")
        
        # Summary statistics
        logger.info(f"Total trading days labeled: {len(labeled_data)}")
        logger.info(f"Date range: {labeled_data['date'].min()} to {labeled_data['date'].max()}")
        logger.info("Regime distribution:")
        for regime, count in labeled_data['regime'].value_counts().items():
            logger.info(f"  {regime}: {count} days ({count/len(labeled_data)*100:.1f}%)")
        
        # Update phase 2 report
        update_phase2_report(labeled_data)
        
        logger.info("Phase 2.2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Phase 2.2: {str(e)}")
        raise


def update_phase2_report(labeled_data: pd.DataFrame):
    """
    Update the phase2_report.md with labeling results.
    
    Args:
        labeled_data: The labeled dataset
    """
    logger.info("Updating phase2_report.md...")
    
    report_file = PROJECT_ROOT / "docs" / "phase2_report.md"
    
    # Generate report content
    regime_stats = labeled_data['regime'].value_counts()
    total_days = len(labeled_data)
    
    report_content = f"""
## Phase 2.2: Label Trading Regimes - COMPLETED ✅

**Objective**: Classify each trading day into market regimes based on index and option features.

### Implementation Details:
- **Module**: `src/data_ingest/label_days.py`
- **Input Files**: 
  - `data/processed/banknifty_index.parquet` (minute-level data)
  - `data/processed/banknifty_options_chain.parquet` (options data)
- **Output File**: `data/processed/labeled_days.parquet`

### Features Computed:
1. **Daily Metrics**: OHLCV, VWAP, returns, volatility
2. **Opening Range Features**: First 30-min high/low/range/volume/breakout direction
3. **IV Metrics**: IV percentiles, ATM IV, IV rank

### Regime Classification Results:
- **Total Trading Days**: {total_days}
- **Date Range**: {labeled_data['date'].min().strftime('%Y-%m-%d')} to {labeled_data['date'].max().strftime('%Y-%m-%d')}

#### Regime Distribution:
"""
    
    for regime, count in regime_stats.items():
        percentage = count / total_days * 100
        report_content += f"- **{regime}**: {count} days ({percentage:.1f}%)\n"
    
    report_content += f"""
### Classification Thresholds:
- High Volatility: >2.0%
- High Momentum: >1.5% daily return
- High IV Percentile: >70%
- High Volume: >1.5x average
- Large OR Range: >1.2% of price

### Output Schema:
The `labeled_days.parquet` file contains {len(labeled_data.columns)} columns including:
- Basic OHLCV data and derived metrics
- Opening range features
- IV percentiles and rankings  
- **regime**: Primary classification (Trend/RangeBound/Event/MildBias/Momentum)

**Status**: ✅ COMPLETED - Ready for Phase 3 model training
"""
    
    # Read existing report and append
    if report_file.exists():
        with open(report_file, 'r') as f:
            existing_content = f.read()
        
        # Add new section
        updated_content = existing_content + "\n" + report_content
    else:
        updated_content = report_content
    
    # Write updated report
    with open(report_file, 'w') as f:
        f.write(updated_content)
    
    logger.info("Phase 2 report updated successfully")


if __name__ == "__main__":
    main()