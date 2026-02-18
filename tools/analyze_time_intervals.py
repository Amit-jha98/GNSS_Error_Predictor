"""
Analyze time intervals in GNSS dataset to check if they follow the required 15-minute intervals
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import os

def analyze_time_intervals(file_path):
    """Analyze time intervals in a CSV file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df.columns = [' '.join(col.strip().split()) for col in df.columns]
    
    # Try different timestamp formats
    formats = ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y%m%d%H%M%S']
    parsed = False
    for fmt in formats:
        try:
            df['timestamp'] = pd.to_datetime(df['utc_time'], format=fmt)
            parsed = True
            print(f"✓ Successfully parsed timestamps using format: {fmt}")
            break
        except:
            continue
    
    if not parsed:
        print("✗ Could not parse timestamps")
        return
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate time differences
    time_diffs = df['timestamp'].diff()
    
    # Basic statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total records: {len(df)}")
    print(f"   - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   - Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Analyze intervals
    print(f"\n⏱️  Time Interval Analysis:")
    time_diffs_minutes = time_diffs.dt.total_seconds() / 60
    
    print(f"   - Min interval: {time_diffs_minutes.min():.2f} minutes")
    print(f"   - Max interval: {time_diffs_minutes.max():.2f} minutes")
    print(f"   - Mean interval: {time_diffs_minutes.mean():.2f} minutes")
    print(f"   - Median interval: {time_diffs_minutes.median():.2f} minutes")
    print(f"   - Std deviation: {time_diffs_minutes.std():.2f} minutes")
    
    # Expected interval is 15 minutes
    expected_interval = 15
    print(f"\n🎯 Expected Interval: {expected_interval} minutes")
    
    # Check adherence to 15-minute intervals
    tolerance = 1  # 1 minute tolerance
    consistent_intervals = time_diffs_minutes.apply(
        lambda x: abs(x - expected_interval) <= tolerance if pd.notna(x) else False
    )
    
    consistent_count = consistent_intervals.sum()
    total_intervals = len(time_diffs) - 1  # Exclude the first NaN
    consistency_percentage = (consistent_count / total_intervals) * 100 if total_intervals > 0 else 0
    
    print(f"   - Intervals matching 15min (±{tolerance}min): {consistent_count}/{total_intervals} ({consistency_percentage:.1f}%)")
    
    # Find unique interval patterns
    print(f"\n📋 Unique Interval Patterns (in minutes):")
    unique_intervals = time_diffs_minutes.dropna().value_counts().sort_index()
    for interval, count in unique_intervals.head(20).items():
        if interval == expected_interval:
            marker = " ← EXPECTED"
        elif abs(interval - expected_interval) <= tolerance:
            marker = " ← CLOSE TO EXPECTED"
        else:
            marker = ""
        print(f"   - {interval:8.2f} min: {count:4d} occurrences{marker}")
    
    # Identify problematic intervals
    print(f"\n⚠️  Problematic Intervals (not 15min ±{tolerance}min):")
    problematic = df[~consistent_intervals & time_diffs.notna()].copy()
    problematic['interval_minutes'] = time_diffs_minutes[~consistent_intervals & time_diffs.notna()]
    
    if len(problematic) > 0:
        print(f"   Found {len(problematic)} problematic intervals:")
        for idx, row in problematic.head(10).iterrows():
            prev_idx = idx - 1
            if prev_idx >= 0 and prev_idx in df.index:
                prev_time = df.loc[prev_idx, 'timestamp']
                curr_time = row['timestamp']
                interval = row['interval_minutes']
                print(f"   - {prev_time} → {curr_time} ({interval:.2f} min)")
        
        if len(problematic) > 10:
            print(f"   ... and {len(problematic) - 10} more")
    else:
        print(f"   ✓ No problematic intervals found!")
    
    # Check for large gaps (>2 hours)
    large_gaps = time_diffs_minutes > 120
    if large_gaps.any():
        print(f"\n🕳️  Large Data Gaps (>2 hours):")
        gap_indices = df[large_gaps].index
        for idx in gap_indices:
            prev_idx = idx - 1
            if prev_idx >= 0 and prev_idx in df.index:
                prev_time = df.loc[prev_idx, 'timestamp']
                curr_time = df.loc[idx, 'timestamp']
                gap_hours = time_diffs_minutes.loc[idx] / 60
                print(f"   - Gap of {gap_hours:.1f} hours: {prev_time} → {curr_time}")
    
    # Check if dataset covers 7 days as required
    expected_days = 7
    actual_days = (df['timestamp'].max() - df['timestamp'].min()).days
    
    print(f"\n📅 Dataset Duration Check:")
    print(f"   - Expected: {expected_days} days")
    print(f"   - Actual: {actual_days} days")
    if actual_days == expected_days:
        print(f"   ✓ Duration matches requirement")
    else:
        print(f"   ✗ Duration does NOT match requirement (off by {abs(actual_days - expected_days)} days)")
    
    # Calculate expected number of records for 7 days at 15-minute intervals
    minutes_per_day = 24 * 60
    intervals_per_day = minutes_per_day / expected_interval
    expected_records = expected_days * intervals_per_day
    
    print(f"\n📊 Expected vs Actual Records:")
    print(f"   - Expected records (7 days, 15min intervals): {expected_records:.0f}")
    print(f"   - Actual records: {len(df)}")
    print(f"   - Difference: {len(df) - expected_records:.0f} ({((len(df) / expected_records) - 1) * 100:.1f}%)")
    
    return df, time_diffs_minutes

def main():
    # Dataset folder
    dataset_folder = r"d:\SIH_FINAL_MODEL\GNSS_MODEL_03\dataset"
    
    # Find all CSV files
    csv_files = [
        os.path.join(dataset_folder, f) 
        for f in os.listdir(dataset_folder) 
        if f.endswith('.csv')
    ]
    
    print("\n" + "="*80)
    print("GNSS DATASET TIME INTERVAL ANALYSIS")
    print("="*80)
    print(f"\nProblem Requirement: 15-minute intervals for 7 days")
    print(f"Expected: ~672 records per satellite (7 days × 96 intervals/day)")
    
    all_results = {}
    
    for file_path in csv_files:
        try:
            df, intervals = analyze_time_intervals(file_path)
            all_results[os.path.basename(file_path)] = {
                'df': df,
                'intervals': intervals
            }
        except Exception as e:
            print(f"\n✗ Error analyzing {os.path.basename(file_path)}: {e}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for filename, result in all_results.items():
        intervals = result['intervals']
        df = result['df']
        
        # Check consistency
        consistent = intervals.apply(lambda x: abs(x - 15) <= 1 if pd.notna(x) else False).sum()
        total = len(intervals) - 1
        consistency_pct = (consistent / total * 100) if total > 0 else 0
        
        duration = (df['timestamp'].max() - df['timestamp'].min()).days
        
        print(f"\n{filename}:")
        print(f"   - Records: {len(df)}")
        print(f"   - Duration: {duration} days")
        print(f"   - 15min consistency: {consistency_pct:.1f}%")
        
        if consistency_pct < 90:
            print(f"   ⚠️  WARNING: Dataset does NOT follow 15-minute intervals consistently!")
        else:
            print(f"   ✓ Dataset mostly follows 15-minute intervals")

if __name__ == "__main__":
    main()
