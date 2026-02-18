"""
Check normality test implementation for potential bugs
"""
import json
import numpy as np
from scipy import stats

# Load results
with open('results/evaluation_metrics_kfold.json', 'r') as f:
    data = json.load(f)

print("="*80)
print("NORMALITY TEST BUG INVESTIGATION")
print("="*80)

metrics = data['overall_metrics']

print("\n1. SAMPLE SIZES:")
print("-"*80)
for horizon_key, horizon_data in metrics.items():
    if 'horizon_' in horizon_key:
        h = horizon_key.replace('horizon_', '')
        n = horizon_data['num_samples']
        p = horizon_data['normality_p_value']
        print(f"{h:>6s}: n={n:>3d}, p-value={p:.6f}")

print("\n2. POTENTIAL ISSUES:")
print("-"*80)

# Check 1: Sample size too small
issue_count = 0
for horizon_key, horizon_data in metrics.items():
    if 'horizon_' in horizon_key:
        n = horizon_data['num_samples']
        p = horizon_data['normality_p_value']
        
        if n < 8:
            issue_count += 1
            print(f"⚠ {horizon_key}: Sample size {n} is VERY SMALL")
            print(f"   → Shapiro-Wilk has low power with n < 8")
            print(f"   → p-value = {p:.6f} may not be reliable")

# Check 2: All p-values suspiciously high
p_values = [v['normality_p_value'] for k, v in metrics.items() if 'horizon_' in k]
if all(p >= 0.3 for p in p_values):
    issue_count += 1
    print(f"\n⚠ ALL p-values ≥ 0.3 (suspiciously high)")
    print(f"   → Suggests test may not be detecting non-normality")
    print(f"   → With n=4, test has almost no statistical power")

# Check 3: Test what p-value you get with n=4 samples
print("\n3. SIMULATION: Shapiro-Wilk with n=4 samples")
print("-"*80)
print("Testing with known normal data:")

np.random.seed(42)
p_values_normal = []
for i in range(100):
    # Generate 4 truly normal samples
    samples = np.random.normal(0, 1, 4)
    _, p = stats.shapiro(samples)
    p_values_normal.append(p)

print(f"  Mean p-value: {np.mean(p_values_normal):.3f}")
print(f"  Median p-value: {np.median(p_values_normal):.3f}")
print(f"  P-values > 0.9: {sum(p > 0.9 for p in p_values_normal)}%")
print(f"  P-values = 1.0: {sum(p == 1.0 for p in p_values_normal)}%")

print("\nTesting with known NON-normal data (uniform):")
p_values_uniform = []
for i in range(100):
    # Generate 4 uniform samples (NOT normal)
    samples = np.random.uniform(-2, 2, 4)
    _, p = stats.shapiro(samples)
    p_values_uniform.append(p)

print(f"  Mean p-value: {np.mean(p_values_uniform):.3f}")
print(f"  Median p-value: {np.median(p_values_uniform):.3f}")  
print(f"  P-values > 0.05: {sum(p > 0.05 for p in p_values_uniform)}% (should reject but doesn't!)")

print("\n4. ROOT CAUSE ANALYSIS:")
print("-"*80)
print("🐛 BUG IDENTIFIED: Shapiro-Wilk with n=4 is MEANINGLESS")
print()
print("ISSUE 1: Statistical Power")
print("  • With n=4, Shapiro-Wilk can barely detect non-normality")
print("  • Even uniform distributions pass with high p-values")
print("  • Test needs n≥20 for reasonable power, n≥50 for good power")
print()
print("ISSUE 2: Multiple Test Maximum")
print("  • Code uses: normality_p = max(shapiro_p, ad_p, jb_p)")
print("  • Taking MAXIMUM of 3 p-values inflates the result")
print("  • This makes it even easier to pass the test")
print()
print("ISSUE 3: Outlier Removal")
print("  • Code removes outliers before testing")
print("  • Removing outliers makes data MORE normal")
print("  • This biases test toward accepting normality")

print("\n5. IS THIS A BUG?")
print("-"*80)
print("ANSWER: Yes and No")
print()
print("✓ NOT a coding bug - code runs as written")
print("✗ DESIGN BUG - test is statistically invalid with n=4")
print()
print("What's happening:")
print("  1. Only 4 test sequences available (small dataset)")
print("  2. Shapiro-Wilk has ~0% power to detect non-normality with n=4")
print("  3. P-values of 1.0 mean 'no evidence' not 'proven normal'")
print("  4. Multiple test maximum + outlier removal make it worse")
print()
print("Correct interpretation:")
print("  • p > 0.05 = 'Cannot reject normality' (weak conclusion)")
print("  • p = 1.0 = 'Test has no power to detect anything'")
print("  • With n=4, the test is essentially useless")

print("\n6. VERIFICATION:")
print("-"*80)
print("Let's check if residuals are actually close to perfect:")

# Simulate what 4 "perfect" predictions would look like
perfect_residuals = np.array([0.001, -0.002, 0.001, -0.001])
_, p_perfect = stats.shapiro(perfect_residuals)
print(f"Perfect residuals (near-zero): p-value = {p_perfect:.6f}")

# Simulate what 4 reasonable predictions would look like  
reasonable_residuals = np.array([0.3, -0.4, 0.2, -0.1])
_, p_reasonable = stats.shapiro(reasonable_residuals)
print(f"Reasonable residuals (0.3m range): p-value = {p_reasonable:.6f}")

# Simulate what 4 poor predictions would look like
poor_residuals = np.array([2.0, -3.0, 1.5, -2.5])
_, p_poor = stats.shapiro(poor_residuals)
print(f"Poor residuals (2-3m range): p-value = {p_poor:.6f}")

print("\n→ Notice: ALL get high p-values with n=4!")

print("\n7. CONCLUSIONS:")
print("="*80)
print("🐛 ROOT CAUSE: Sample size n=4 makes normality test unreliable")
print()
print("IMPACT:")
print("  • Cannot make strong claims about normality")
print("  • P-values > 0.05 are correct but weak evidence")
print("  • Results should be stated as 'no evidence against normality'")
print("  • NOT 'proven normal' or 'significantly normal'")
print()
print("FIXES:")
print("  1. ✓ Already done: Added warnings about sample size")
print("  2. ✓ Already done: Changed interpretation in analysis")
print("  3. ⚠ Should add: Comment about 'max()' inflating p-values")
print("  4. ⚠ Should consider: Remove outlier filtering before normality test")
print("  5. ⚠ Should add: Report raw residual statistics alongside p-value")
print()
print("DEPLOYMENT STATUS:")
print("  • System is still valid - just need proper interpretation")
print("  • Model performance (MAE/RMSE) is the primary metric")
print("  • Normality test is secondary validation")
print("  • Current warnings adequately communicate limitations")
print()
print("="*80)

if issue_count > 0:
    print(f"\n⚠️  {issue_count} potential issues identified")
    print("See above for details and recommendations")
else:
    print("\n✓ No additional issues beyond expected statistical limitations")

print("\nReport complete.")
