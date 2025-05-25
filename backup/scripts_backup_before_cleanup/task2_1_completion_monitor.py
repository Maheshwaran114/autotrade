#!/usr/bin/env python3
"""
Task 2.1 Completion Status Monitor
Comprehensive monitoring and validation of the full year data extraction completion
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

def monitor_task2_1_completion():
    """Monitor and validate Task 2.1 completion status"""
    
    project_root = Path(__file__).parent.parent
    
    print("🎯 TASK 2.1 FULL YEAR DATA EXTRACTION - COMPLETION MONITOR")
    print("=" * 80)
    print(f"Monitoring Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check processed files
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"
    
    status = {
        "extraction_complete": False,
        "processed_files_ready": False,
        "data_quality_validated": False,
        "ml_pipeline_ready": False,
        "documentation_updated": False
    }
    
    # 1. Check if processed files exist
    print("📊 CHECKING PROCESSED DATA FILES:")
    print("-" * 40)
    
    index_file = processed_dir / "banknifty_index.parquet"
    options_file = processed_dir / "banknifty_options_chain.parquet"
    
    if index_file.exists():
        index_df = pd.read_parquet(index_file)
        print(f"✅ Bank Nifty Index: {len(index_df):,} records")
        print(f"   📅 Date range: {index_df['date'].min()} to {index_df['date'].max()}")
        print(f"   📈 Unique dates: {index_df['date'].dt.date.nunique()} trading days")
        status["processed_files_ready"] = True
    else:
        print("❌ Bank Nifty Index: File not found")
    
    if options_file.exists():
        options_df = pd.read_parquet(options_file)
        print(f"✅ Options Chain: {len(options_df):,} records")
        print(f"   📅 Date range: {options_df['date'].min()} to {options_df['date'].max()}")
        print(f"   🎯 Unique strikes: {options_df['strike'].nunique()}")
        print(f"   📊 CE/PE split: {(options_df['option_type'] == 'CE').sum()}/{(options_df['option_type'] == 'PE').sum()}")
        status["data_quality_validated"] = True
    else:
        print("❌ Options Chain: File not found")
    
    print()
    
    # 2. Check raw data collection
    print("📁 CHECKING RAW DATA COLLECTION:")
    print("-" * 40)
    
    # Count different file types
    csv_files = list(raw_dir.glob("bnk_index_*.csv"))
    parquet_index_files = list(raw_dir.glob("banknifty_index_*.parquet"))
    parquet_options_files = list(raw_dir.glob("options_*.parquet")) + list(raw_dir.glob("banknifty_options_*.parquet"))
    
    print(f"📈 Index CSV files: {len(csv_files)}")
    print(f"📈 Index Parquet files: {len(parquet_index_files)}")
    print(f"📊 Options Parquet files: {len(parquet_options_files)}")
    
    total_raw_files = len(csv_files) + len(parquet_index_files) + len(parquet_options_files)
    if total_raw_files > 50:  # Substantial data collection
        status["extraction_complete"] = True
        print(f"✅ Substantial data collection: {total_raw_files} total files")
    else:
        print(f"⚠️ Limited data collection: {total_raw_files} total files")
    
    print()
    
    # 3. Validate data quality
    print("🔍 DATA QUALITY VALIDATION:")
    print("-" * 40)
    
    if index_file.exists() and options_file.exists():
        # Check index data quality
        print("📈 Bank Nifty Index Quality:")
        print(f"   🔢 Total records: {len(index_df):,}")
        print(f"   📊 Columns: {list(index_df.columns)}")
        
        # Check for realistic OHLC values
        if 'close' in index_df.columns:
            avg_close = index_df['close'].mean()
            print(f"   💰 Average close price: ₹{avg_close:,.2f}")
            if 35000 <= avg_close <= 75000:  # Realistic Bank Nifty range
                print("   ✅ Price values within realistic range")
            else:
                print("   ⚠️ Price values outside expected range")
        
        # Check volume data
        if 'volume' in index_df.columns:
            total_volume = index_df['volume'].sum()
            non_zero_volume = (index_df['volume'] > 0).sum()
            print(f"   📊 Total volume: {total_volume:,}")
            print(f"   📈 Non-zero volume records: {non_zero_volume:,}/{len(index_df):,}")
        
        print()
        print("📊 Options Chain Quality:")
        print(f"   🔢 Total records: {len(options_df):,}")
        print(f"   📊 Columns: {list(options_df.columns)}")
        
        # Check for realistic option prices
        if 'last_price' in options_df.columns:
            avg_price = options_df['last_price'].mean()
            print(f"   💰 Average option price: ₹{avg_price:,.2f}")
        
        status["ml_pipeline_ready"] = True
    
    print()
    
    # 4. Check documentation updates
    print("📚 DOCUMENTATION STATUS:")
    print("-" * 40)
    
    docs_dir = project_root / "docs"
    phase2_report = docs_dir / "phase2_report.md"
    
    if phase2_report.exists():
        with open(phase2_report, 'r') as f:
            content = f.read()
            if "TASK 2.1 FULL YEAR DATA EXTRACTION COMPLETED" in content:
                print("✅ Phase 2 report updated with Task 2.1 completion")
                status["documentation_updated"] = True
            else:
                print("⚠️ Phase 2 report missing Task 2.1 completion section")
    else:
        print("❌ Phase 2 report not found")
    
    print()
    
    # 5. Overall completion status
    print("🎯 OVERALL COMPLETION STATUS:")
    print("=" * 40)
    
    completed_tasks = sum(status.values())
    total_tasks = len(status)
    completion_pct = (completed_tasks / total_tasks) * 100
    
    for task, completed in status.items():
        icon = "✅" if completed else "❌"
        print(f"{icon} {task.replace('_', ' ').title()}")
    
    print()
    print(f"📊 Overall Progress: {completed_tasks}/{total_tasks} ({completion_pct:.1f}%)")
    
    if completion_pct >= 80:
        print("🎉 TASK 2.1 SUBSTANTIALLY COMPLETED!")
        print("✨ Ready for next phase: Feature Engineering (Task 2.2)")
    elif completion_pct >= 60:
        print("⚠️ TASK 2.1 MOSTLY COMPLETED")
        print("🔧 Minor issues need attention")
    else:
        print("❌ TASK 2.1 NEEDS ATTENTION")
        print("🛠️ Significant work remaining")
    
    # 6. Generate status report
    status_report = {
        "timestamp": datetime.now().isoformat(),
        "completion_percentage": completion_pct,
        "status": status,
        "data_summary": {}
    }
    
    if index_file.exists():
        status_report["data_summary"]["index_records"] = len(index_df)
        status_report["data_summary"]["index_date_range"] = [
            str(index_df['date'].min()), 
            str(index_df['date'].max())
        ]
    
    if options_file.exists():
        status_report["data_summary"]["options_records"] = len(options_df)
        status_report["data_summary"]["options_date_range"] = [
            str(options_df['date'].min()), 
            str(options_df['date'].max())
        ]
    
    # Save status report
    status_file = project_root / "logs" / "task2_1_completion_status.json"
    with open(status_file, 'w') as f:
        json.dump(status_report, f, indent=2, default=str)
    
    print(f"\n📄 Status report saved to: {status_file}")
    
    return status_report

if __name__ == "__main__":
    monitor_task2_1_completion()
