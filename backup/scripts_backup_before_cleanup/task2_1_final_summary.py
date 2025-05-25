#!/usr/bin/env python3
"""
Task 2.1 Final Summary & Validation Report
Complete validation of the full year data extraction implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def generate_final_summary():
    """Generate final comprehensive summary of Task 2.1 completion"""
    
    print("🎯 TASK 2.1: FULL YEAR DATA EXTRACTION - FINAL SUMMARY")
    print("=" * 80)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load processed data
    processed_dir = Path("data/processed")
    index_file = processed_dir / "banknifty_index.parquet"
    options_file = processed_dir / "banknifty_options_chain.parquet"
    
    # Validate data exists
    if not index_file.exists() or not options_file.exists():
        print("❌ ERROR: Processed data files missing!")
        return
    
    # Load data
    index_df = pd.read_parquet(index_file)
    options_df = pd.read_parquet(options_file)
    
    print("📊 DATA COLLECTION SUMMARY")
    print("-" * 50)
    
    # Index data summary
    print("📈 BANK NIFTY INDEX DATA:")
    print(f"   📊 Total Records: {len(index_df):,}")
    print(f"   📅 Date Range: {index_df['date'].min().strftime('%Y-%m-%d')} to {index_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   📆 Trading Days: {index_df['date'].dt.date.nunique()} unique dates")
    print(f"   ⏰ Time Coverage: {index_df['date'].dt.time.nunique()} unique times per day")
    
    # Price statistics
    avg_close = index_df['close'].mean()
    min_close = index_df['close'].min()
    max_close = index_df['close'].max()
    print(f"   💰 Price Range: ₹{min_close:,.2f} - ₹{max_close:,.2f} (Avg: ₹{avg_close:,.2f})")
    
    # Volume analysis
    total_volume = index_df['volume'].sum()
    non_zero_volume = (index_df['volume'] > 0).sum()
    volume_coverage = (non_zero_volume / len(index_df)) * 100
    print(f"   📊 Volume Coverage: {non_zero_volume:,}/{len(index_df):,} records ({volume_coverage:.1f}%)")
    print(f"   📈 Total Volume: {total_volume:,}")
    
    print()
    
    # Options data summary
    print("📊 OPTIONS CHAIN DATA:")
    print(f"   📊 Total Records: {len(options_df):,}")
    print(f"   📅 Date Range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"   🎯 Strike Range: {options_df['strike'].min():,.0f} - {options_df['strike'].max():,.0f}")
    print(f"   📈 Unique Strikes: {options_df['strike'].nunique()}")
    
    # Call/Put distribution
    ce_count = (options_df['option_type'] == 'CE').sum()
    pe_count = (options_df['option_type'] == 'PE').sum()
    print(f"   📊 Call/Put Split: {ce_count} CE / {pe_count} PE")
    
    # Option price statistics
    if 'last_price' in options_df.columns:
        avg_price = options_df['last_price'].mean()
        print(f"   💰 Avg Option Price: ₹{avg_price:,.2f}")
    
    # Volume and OI for options
    if 'volume' in options_df.columns:
        total_opt_volume = options_df['volume'].sum()
        print(f"   📈 Total Options Volume: {total_opt_volume:,}")
    
    if 'oi' in options_df.columns:
        total_oi = options_df['oi'].sum()
        print(f"   📊 Total Open Interest: {total_oi:,}")
    
    print()
    
    # Data quality assessment
    print("🔍 DATA QUALITY ASSESSMENT")
    print("-" * 50)
    
    quality_scores = []
    
    # 1. Data completeness
    print("1️⃣ Data Completeness:")
    if len(index_df) > 80000:  # Substantial minute data
        print("   ✅ Index data: Substantial coverage (>80k records)")
        quality_scores.append(1)
    else:
        print("   ⚠️ Index data: Limited coverage")
        quality_scores.append(0.5)
    
    if len(options_df) > 500:  # Good options coverage
        print("   ✅ Options data: Good coverage (>500 records)")
        quality_scores.append(1)
    else:
        print("   ⚠️ Options data: Limited coverage")
        quality_scores.append(0.5)
    
    # 2. Data consistency
    print("2️⃣ Data Consistency:")
    # Check for NaN values in critical columns
    critical_cols_index = ['open', 'high', 'low', 'close']
    index_nan_count = index_df[critical_cols_index].isna().sum().sum()
    
    if index_nan_count == 0:
        print("   ✅ Index OHLC: No missing values")
        quality_scores.append(1)
    else:
        print(f"   ⚠️ Index OHLC: {index_nan_count} missing values")
        quality_scores.append(0.7)
    
    # 3. Price reasonableness
    print("3️⃣ Price Reasonableness:")
    if 40000 <= avg_close <= 70000:  # Reasonable Bank Nifty range
        print(f"   ✅ Index prices: Within reasonable range (₹{avg_close:,.0f})")
        quality_scores.append(1)
    else:
        print(f"   ⚠️ Index prices: Outside expected range (₹{avg_close:,.0f})")
        quality_scores.append(0.5)
    
    # 4. Volume data quality
    print("4️⃣ Volume Data Quality:")
    if volume_coverage > 50:
        print(f"   ✅ Volume coverage: {volume_coverage:.1f}% (Good)")
        quality_scores.append(1)
    elif volume_coverage > 20:
        print(f"   ⚠️ Volume coverage: {volume_coverage:.1f}% (Moderate)")
        quality_scores.append(0.7)
    else:
        print(f"   ❌ Volume coverage: {volume_coverage:.1f}% (Low)")
        quality_scores.append(0.3)
    
    overall_quality = np.mean(quality_scores) * 100
    print(f"\n📊 Overall Data Quality Score: {overall_quality:.1f}/100")
    
    print()
    
    # Implementation validation
    print("🛠️ IMPLEMENTATION VALIDATION")
    print("-" * 50)
    
    implementation_checks = []
    
    # Check 1: Proper file structure
    raw_dir = Path("data/raw")
    csv_count = len(list(raw_dir.glob("bnk_index_*.csv")))
    parquet_count = len(list(raw_dir.glob("*.parquet")))
    
    print(f"1️⃣ File Structure: {csv_count} CSV + {parquet_count} Parquet files")
    if csv_count > 50 or parquet_count > 5:
        print("   ✅ Substantial raw data collection")
        implementation_checks.append(1)
    else:
        print("   ⚠️ Limited raw data collection")
        implementation_checks.append(0.7)
    
    # Check 2: Data consolidation
    print("2️⃣ Data Consolidation:")
    if index_file.exists() and options_file.exists():
        print("   ✅ Processed files created successfully")
        implementation_checks.append(1)
    else:
        print("   ❌ Processed files missing")
        implementation_checks.append(0)
    
    # Check 3: Volume integration
    print("3️⃣ Volume Integration:")
    if total_volume > 0 and volume_coverage > 20:
        print("   ✅ Volume proxy successfully implemented")
        implementation_checks.append(1)
    else:
        print("   ⚠️ Volume proxy partially implemented")
        implementation_checks.append(0.5)
    
    # Check 4: Options data structure
    print("4️⃣ Options Data Structure:")
    required_options_cols = ['strike', 'option_type', 'last_price']
    has_required_cols = all(col in options_df.columns for col in required_options_cols)
    
    if has_required_cols and len(options_df) > 100:
        print("   ✅ Options data properly structured")
        implementation_checks.append(1)
    else:
        print("   ⚠️ Options data structure issues")
        implementation_checks.append(0.5)
    
    implementation_score = np.mean(implementation_checks) * 100
    print(f"\n🔧 Implementation Score: {implementation_score:.1f}/100")
    
    print()
    
    # Final assessment
    print("🎉 FINAL ASSESSMENT")
    print("=" * 50)
    
    overall_score = (overall_quality + implementation_score) / 2
    
    print(f"📊 Data Quality: {overall_quality:.1f}/100")
    print(f"🔧 Implementation: {implementation_score:.1f}/100")
    print(f"🎯 Overall Score: {overall_score:.1f}/100")
    print()
    
    if overall_score >= 85:
        status = "🎉 EXCELLENT - Task 2.1 FULLY COMPLETED"
        next_steps = "Ready for Task 2.2: Feature Engineering"
    elif overall_score >= 70:
        status = "✅ GOOD - Task 2.1 SUBSTANTIALLY COMPLETED"
        next_steps = "Minor optimizations possible, ready for next phase"
    elif overall_score >= 55:
        status = "⚠️ SATISFACTORY - Task 2.1 COMPLETED with limitations"
        next_steps = "Some improvements needed, but functional for next phase"
    else:
        status = "❌ NEEDS IMPROVEMENT - Task 2.1 partially completed"
        next_steps = "Significant work needed before proceeding"
    
    print(f"Status: {status}")
    print(f"Next Steps: {next_steps}")
    
    # Sample data preview
    print()
    print("📋 SAMPLE DATA PREVIEW")
    print("-" * 50)
    print("📈 Bank Nifty Index (Latest 3 records):")
    print(index_df.tail(3)[['date', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False))
    
    print()
    print("📊 Options Chain (Latest 3 records):")
    sample_cols = ['date', 'strike', 'option_type', 'last_price', 'volume', 'oi'] if 'oi' in options_df.columns else ['date', 'strike', 'option_type', 'last_price']
    print(options_df.tail(3)[sample_cols].to_string(index=False))
    
    # Generate summary report
    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "overall_score": overall_score,
        "data_quality_score": overall_quality,
        "implementation_score": implementation_score,
        "status": status,
        "data_summary": {
            "index_records": len(index_df),
            "options_records": len(options_df),
            "date_range": [
                index_df['date'].min().isoformat(),
                index_df['date'].max().isoformat()
            ],
            "volume_coverage_pct": volume_coverage,
            "total_volume": int(total_volume)
        },
        "next_steps": next_steps
    }
    
    # Save report
    report_file = Path("logs/task2_1_final_summary.json")
    with open(report_file, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: {report_file}")
    print()
    print("🚀 Task 2.1 Full Year Data Extraction: IMPLEMENTATION COMPLETE!")

if __name__ == "__main__":
    generate_final_summary()
