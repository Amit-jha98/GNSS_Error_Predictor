"""
Investigate why only 4 test samples with k-fold validation
"""
import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
data_folder = "dataset"
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
all_files = csv_files + excel_files

print("="*80)
print("DATA INVESTIGATION: Why only 4 test samples?")
print("="*80)

all_dfs = []
for file_path in all_files:
    filename = os.path.basename(file_path)
    print(f"\nProcessing: {filename}")
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        df.columns = [' '.join(col.strip().split()) for col in df.columns]
        
        # Determine orbit class
        if 'GEO' in filename.upper():
            orbit_class = 'GEO'
        elif 'MEO' in filename.upper():
            orbit_class = 'MEO'
        elif 'GSO' in filename.upper():
            orbit_class = 'GSO'
        else:
            orbit_class = 'Unknown'
        
        sat_id = f"{orbit_class}_{filename.replace('.csv', '').replace('.xlsx', '').split('_')[-1]}"
        df['satellite_id'] = sat_id
        
        print(f"  Satellite: {sat_id}")
        print(f"  Total samples: {len(df)}")
        
        # Calculate splits
        n = len(df)
        test_cutoff_idx = int(n * 0.85)
        train_val_size = test_cutoff_idx
        test_size = n - test_cutoff_idx
        
        print(f"  Split (85/15): Train+Val={train_val_size}, Test={test_size}")
        
        all_dfs.append(df)
        
    except Exception as e:
        print(f"  Error loading {filename}: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total files: {len(all_dfs)}")
print(f"Total satellites: {len(all_dfs)}")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined)} samples")
    
    # Calculate test set size
    test_samples = []
    for sat_id in combined['satellite_id'].unique():
        sat_data = combined[combined['satellite_id'] == sat_id]
        n = len(sat_data)
        test_cutoff_idx = int(n * 0.85)
        test_size = n - test_cutoff_idx
        test_samples.append(test_size)
        print(f"  {sat_id}: {n} total → {test_size} test samples")
    
    print(f"\nTotal test samples across all satellites: {sum(test_samples)}")
    print(f"Satellites contributing to test set: {len([t for t in test_samples if t >= 5])}")

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print("\nWhy you get 4 test samples:")
print()
print("1. K-FOLD DESIGN:")
print("   • Code uses 85% for training+validation")
print("   • Only 15% held out as final test set")
print("   • This is CORRECT for k-fold - training data is maximized")
print()
print("2. SEQUENCE-LEVEL TESTING:")
print("   • Model predicts on sequences, not individual timestamps")
print("   • With sequence_length=7, you need 7 consecutive points to make 1 prediction")
print("   • 4 test samples = 4 complete sequences that can be evaluated")
print()
print("3. DATA FILTERING:")
print("   • Code filters out sequences with outliers or missing data")
print("   • Some test sequences may have been removed during quality checks")
print()
print("COMPARISON WITH EARLIER RUNS:")
print("  'Earlier it worked well even with less data' suggests:")
print("  • You may have used regular train/test split (not k-fold) before")
print("  • Regular split uses 80/20, giving more test samples")
print("  • K-fold uses 85/15 to maximize training data for better model")
print("  • Trade-off: Better model training vs. More test samples")
print()
print("IS 4 SAMPLES TOO FEW?")
print("  ✗ For statistical power: YES - cannot reliably test normality")
print("  ✓ For model validation: Depends on your use case")
print("  ✓ For demonstrating concept: Sufficient if predictions are accurate")
print()
print("SOLUTIONS:")
print("  1. Use more data: More satellites or longer time series")
print("  2. Use regular split: Change to 80/20 split (not k-fold)")
print("  3. Accept limitation: Focus on MAE/RMSE, not normality test")
print("  4. Reduce sequence length: Use shorter input sequences")
print()
print("="*80)
