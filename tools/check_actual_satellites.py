"""
Check actual satellite count in the data files
"""
import pandas as pd
import os
import glob

data_folder = "dataset"
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

print("="*80)
print("ACTUAL SATELLITE COUNT IN DATA FILES")
print("="*80)

total_satellites = set()

for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"\n{filename}:")
    
    df = pd.read_csv(file_path)
    df.columns = [' '.join(col.strip().split()) for col in df.columns]
    
    # Check if there's a satellite column
    if 'satellite' in df.columns:
        satellites = df['satellite'].unique()
        print(f"  Has 'satellite' column: YES")
        print(f"  Unique satellites: {len(satellites)}")
        print(f"  Satellite IDs: {sorted(satellites)}")
        total_satellites.update(satellites)
    else:
        print(f"  Has 'satellite' column: NO")
        print(f"  Treated as: 1 satellite (entire file)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total unique satellites across all files: {len(total_satellites)}")
print(f"Satellite IDs: {sorted(total_satellites)}")

print("\n" + "="*80)
print("🐛 BUG FOUND: DATA LOADING IGNORES 'satellite' COLUMN!")
print("="*80)
print()
print("PROBLEM:")
print("  • Your Train2 files have 32 satellites per file (G01-G32)")
print("  • But data_utils.py treats ENTIRE FILE as 1 satellite")
print("  • It creates satellite_id from FILENAME, not from 'satellite' column")
print("  • Result: 32 satellites collapsed into 1 → massive data loss!")
print()
print("CURRENT BEHAVIOR:")
print("  DATA_GEO_Train2.csv → satellite_id='GEO_Train2' (ignores G01-G32)")
print("  DATA_MEO_Train2.csv → satellite_id='MEO_Train2' (ignores G01-G32)")
print()
print("CORRECT BEHAVIOR:")
print("  DATA_GEO_Train2.csv → 32 satellites: GEO_G01, GEO_G02, ..., GEO_G32")
print("  DATA_MEO_Train2.csv → Similar split")
print()
print("IMPACT:")
print("  ✗ You should have 60+ satellites, but system only sees 4")
print("  ✗ Test sequences: Should be 60+, but you only get 4")
print("  ✗ Normality test: Would have n≈60 (reliable) but has n=4 (useless)")
print()
print("FIX NEEDED:")
print("  Modify data_utils.py load_and_prepare_data() to:")
print("  1. Check if 'satellite' column exists")
print("  2. If yes, use it to create individual satellite_id values")
print("  3. Split dataframe by satellite column, not by file")
print()
print("="*80)
