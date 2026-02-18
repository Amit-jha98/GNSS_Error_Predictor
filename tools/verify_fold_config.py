"""
Check if 3-fold will work with 70/30 split
"""
import sys
sys.path.insert(0, '.')
from config import ModelConfig
from data_utils import load_and_prepare_data

config = ModelConfig()

print("="*80)
print("VERIFYING 3-FOLD WITH 70/30 SPLIT")
print("="*80)

df = load_and_prepare_data("dataset")

print(f"\nConfiguration:")
print(f"  Sequence length: {config.sequence_length}")
print(f"  Number of folds: {config.n_folds}")
print(f"  Test split: 30% (70% for train+val)")

satellites_ok = 0
satellites_fail = 0

print(f"\nPer-satellite analysis:")
print(f"{'Satellite':<15} {'Total':<8} {'Train+Val':<10} {'Fold size':<10} {'Status'}")
print("-"*65)

for sat_id in sorted(df['satellite_id'].unique())[:15]:  # Show first 15
    sat_data = df[df['satellite_id'] == sat_id]
    n = len(sat_data)
    
    # 70/30 split
    train_val_size = int(n * 0.70)
    test_size = n - train_val_size
    
    # Fold size
    fold_size = train_val_size // config.n_folds
    
    # Check if fold size >= sequence_length
    if fold_size >= config.sequence_length:
        status = "✓ OK"
        satellites_ok += 1
    else:
        status = f"✗ FAIL (fold={fold_size} < seq={config.sequence_length})"
        satellites_fail += 1
    
    print(f"{sat_id:<15} {n:<8} {train_val_size:<10} {fold_size:<10} {status}")

total_sats = df['satellite_id'].nunique()
if total_sats > 15:
    print(f"... checking remaining {total_sats - 15} satellites ...")
    for sat_id in sorted(df['satellite_id'].unique())[15:]:
        sat_data = df[df['satellite_id'] == sat_id]
        n = len(sat_data)
        train_val_size = int(n * 0.70)
        fold_size = train_val_size // config.n_folds
        
        if fold_size >= config.sequence_length:
            satellites_ok += 1
        else:
            satellites_fail += 1

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total satellites: {total_sats}")
print(f"Satellites OK for 3-fold: {satellites_ok}")
print(f"Satellites failing: {satellites_fail}")

if satellites_ok >= 30:
    print(f"\n✓ SUCCESS: {satellites_ok} satellites can use 3-fold cross-validation")
    print("✓ Training will work correctly")
elif satellites_ok >= 20:
    print(f"\n⚠ PARTIAL: {satellites_ok} satellites work, but {satellites_fail} will be skipped")
    print("  Training will work but with reduced satellites")
else:
    print(f"\n✗ FAILURE: Only {satellites_ok} satellites work")
    print(f"  Consider: reduce n_folds to 2, or use 80/20 split instead")

print("="*80)
