"""
Estimate test sample count with new 70/30 split
"""
import pandas as pd
import sys
sys.path.insert(0, '.')
from config import ModelConfig

config = ModelConfig()

print("="*80)
print("ESTIMATING TEST SAMPLES WITH 70/30 SPLIT")
print("="*80)

# Load data to estimate
from data_utils import load_and_prepare_data
df = load_and_prepare_data("dataset")

print(f"\nTotal satellites: {df['satellite_id'].nunique()}")
print(f"Sequence length: {config.sequence_length}")

test_sequence_estimate = 0
details = []

for sat_id in sorted(df['satellite_id'].unique()):
    sat_data = df[df['satellite_id'] == sat_id].sort_values('timestamp')
    n = len(sat_data)
    
    # 70/30 split
    test_cutoff_idx = int(n * 0.70)
    test_size = n - test_cutoff_idx
    
    # Each sequence needs sequence_length points
    # With stride=1, we can create (test_size - sequence_length + 1) sequences
    if test_size >= config.sequence_length:
        sequences_from_this_sat = test_size - config.sequence_length + 1
    else:
        sequences_from_this_sat = 0
    
    test_sequence_estimate += sequences_from_this_sat
    
    if sequences_from_this_sat > 0:
        details.append((sat_id, n, test_size, sequences_from_this_sat))

print(f"\n{'Satellite':<15} {'Total':<8} {'Test':<8} {'Sequences'}")
print("-"*50)
for sat_id, total, test, seqs in details[:10]:  # Show first 10
    print(f"{sat_id:<15} {total:<8} {test:<8} {seqs}")

if len(details) > 10:
    print(f"... and {len(details)-10} more satellites")
    for sat_id, total, test, seqs in details[-3:]:  # Show last 3
        print(f"{sat_id:<15} {total:<8} {test:<8} {seqs}")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Estimated test sequences: ~{test_sequence_estimate}")
print()
if test_sequence_estimate >= 50:
    print(f"✓ SUCCESS: {test_sequence_estimate} sequences > 50 minimum")
    print("✓ Normality testing will be RELIABLE")
    print("✓ Statistical power is adequate")
else:
    print(f"⚠ Only {test_sequence_estimate} sequences")
    print("  Need to adjust split further or use other methods")

print("\n" + "="*80)
print("TRADE-OFFS")
print("="*80)
print("70/30 split:")
print("  ✓ More test samples for reliable statistics")
print("  ✗ Less training data (but still plenty with k-fold)")
print("  ✓ Better validation of normality requirement")
print("  ✓ Model still has ~70% of data for training")
print("="*80)
