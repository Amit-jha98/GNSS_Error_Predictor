"""
Visualize time interval issues in GNSS dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import timedelta

def visualize_time_coverage(file_path, satellite_name):
    """Create visualization of time coverage and intervals"""
    # Read the CSV file
    df = pd.read_csv(file_path)
    df.columns = [' '.join(col.strip().split()) for col in df.columns]
    
    # Parse timestamps
    formats = ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S']
    for fmt in formats:
        try:
            df['timestamp'] = pd.to_datetime(df['utc_time'], format=fmt)
            break
        except:
            continue
    
    df = df.sort_values('timestamp')
    
    # Calculate intervals in minutes
    df['interval_minutes'] = df['timestamp'].diff().dt.total_seconds() / 60
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Time Interval Analysis: {satellite_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Timeline with data points
    ax1 = axes[0]
    ax1.scatter(df['timestamp'], [1]*len(df), alpha=0.6, s=20, color='blue')
    ax1.set_ylim(0.5, 1.5)
    ax1.set_ylabel('Data\nPoints', fontsize=10)
    ax1.set_title('Data Coverage Timeline', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yticks([])
    
    # Add expected vs actual duration
    duration = (df['timestamp'].max() - df['timestamp'].min()).days
    ax1.text(0.02, 0.95, f'Duration: {duration} days (Expected: 7 days)', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             verticalalignment='top')
    
    # Plot 2: Interval distribution
    ax2 = axes[1]
    intervals = df['interval_minutes'].dropna()
    
    # Create bins
    bins = [0, 5, 10, 15, 20, 30, 60, 120, 180, 300, 1500]
    hist_data = ax2.hist(intervals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical line for expected 15-minute interval
    ax2.axvline(x=15, color='red', linestyle='--', linewidth=2, label='Expected: 15 min')
    
    ax2.set_xlabel('Interval (minutes)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of Time Intervals Between Consecutive Observations', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {intervals.mean():.1f} min\nMedian: {intervals.median():.1f} min\nStd: {intervals.std():.1f} min'
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: Time series of intervals
    ax3 = axes[2]
    ax3.plot(df['timestamp'][1:], df['interval_minutes'][1:], 
             marker='o', markersize=3, linestyle='-', linewidth=0.5, color='darkgreen', alpha=0.7)
    ax3.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Expected: 15 min')
    ax3.axhline(y=14, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±1 min tolerance')
    ax3.axhline(y=16, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Interval (minutes)', fontsize=11)
    ax3.set_title('Time Intervals Over Time (Sequential Observations)', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, min(300, intervals.max() * 1.1))
    
    # Add consistency percentage
    consistent = ((intervals >= 14) & (intervals <= 16)).sum()
    consistency_pct = (consistent / len(intervals) * 100)
    ax3.text(0.02, 0.97, f'15-min Consistency: {consistency_pct:.1f}%', 
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', 
                      facecolor='lightgreen' if consistency_pct > 80 else 'lightcoral', 
                      alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(file_path), f'{satellite_name}_interval_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    
    plt.show()
    
    return fig

def create_comparison_chart(dataset_folder):
    """Create comparison chart of all datasets"""
    files = {
        'DATA_GEO_Train.csv': 'GEO Satellite',
        'DATA_MEO_Train.csv': 'MEO Satellite 1',
        'DATA_MEO_Train2.csv': 'MEO Satellite 2'
    }
    
    stats = []
    
    for filename, sat_name in files.items():
        file_path = os.path.join(dataset_folder, filename)
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        df.columns = [' '.join(col.strip().split()) for col in df.columns]
        
        formats = ['%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                df['timestamp'] = pd.to_datetime(df['utc_time'], format=fmt)
                break
            except:
                continue
        
        df = df.sort_values('timestamp')
        intervals = df['timestamp'].diff().dt.total_seconds() / 60
        
        duration = (df['timestamp'].max() - df['timestamp'].min()).days
        consistent = ((intervals >= 14) & (intervals <= 16)).sum()
        consistency_pct = (consistent / (len(intervals) - 1) * 100) if len(intervals) > 1 else 0
        
        stats.append({
            'name': sat_name,
            'records': len(df),
            'duration': duration,
            'consistency': consistency_pct,
            'mean_interval': intervals.mean(),
            'expected_records': 672
        })
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GNSS Dataset Comparison: Problem Requirements vs Actual Data', 
                 fontsize=16, fontweight='bold')
    
    names = [s['name'] for s in stats]
    
    # Plot 1: Record count comparison
    ax1 = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    ax1.bar(x - width/2, [s['expected_records'] for s in stats], width, 
            label='Expected (7 days, 15min)', color='green', alpha=0.7)
    ax1.bar(x + width/2, [s['records'] for s in stats], width, 
            label='Actual', color='red', alpha=0.7)
    ax1.set_ylabel('Number of Records', fontsize=11)
    ax1.set_title('Expected vs Actual Record Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Duration comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, [7]*len(stats), width, label='Expected', color='green', alpha=0.7)
    ax2.bar(x + width/2, [s['duration'] for s in stats], width, label='Actual', color='red', alpha=0.7)
    ax2.set_ylabel('Days', fontsize=11)
    ax2.set_title('Dataset Duration', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 8)
    
    # Plot 3: Consistency percentage
    ax3 = axes[1, 0]
    bars = ax3.bar(names, [s['consistency'] for s in stats], color='steelblue', alpha=0.7)
    ax3.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Acceptable (80%)')
    ax3.set_ylabel('Consistency (%)', fontsize=11)
    ax3.set_title('15-Minute Interval Consistency (±1 min)', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
    
    # Color bars based on consistency
    for i, bar in enumerate(bars):
        if stats[i]['consistency'] < 50:
            bar.set_color('red')
        elif stats[i]['consistency'] < 80:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    # Plot 4: Mean interval comparison
    ax4 = axes[1, 1]
    ax4.bar(names, [s['mean_interval'] for s in stats], color='purple', alpha=0.7)
    ax4.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Expected: 15 min')
    ax4.set_ylabel('Mean Interval (minutes)', fontsize=11)
    ax4.set_title('Average Time Interval Between Observations', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(names, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = os.path.join(dataset_folder, 'dataset_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison chart: {output_path}")
    
    plt.show()
    
    return fig

def main():
    dataset_folder = r"d:\SIH_FINAL_MODEL\GNSS_MODEL_03\dataset"
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR GNSS DATASET ANALYSIS")
    print("="*80 + "\n")
    
    # Individual file visualizations
    files = {
        'DATA_GEO_Train.csv': 'GEO_Satellite',
        'DATA_MEO_Train.csv': 'MEO_Satellite_1',
        'DATA_MEO_Train2.csv': 'MEO_Satellite_2'
    }
    
    for filename, sat_name in files.items():
        file_path = os.path.join(dataset_folder, filename)
        if os.path.exists(file_path):
            print(f"Processing {filename}...")
            try:
                visualize_time_coverage(file_path, sat_name)
            except Exception as e:
                print(f"✗ Error: {e}")
    
    # Comparison chart
    print("\nGenerating comparison chart...")
    try:
        create_comparison_chart(dataset_folder)
    except Exception as e:
        print(f"✗ Error creating comparison: {e}")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
