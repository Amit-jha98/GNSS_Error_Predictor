"""
Data utilities for GNSS Error Prediction System
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import warnings
from scipy import stats
from scipy.special import inv_boxcox, boxcox
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import os
import glob
from collections import defaultdict

from config import ModelConfig

logger = logging.getLogger(__name__)

# ==================== Data Loading Functions ====================
def load_and_prepare_data(data_folder: str) -> pd.DataFrame:
    """
    Loads and prepares GNSS data from CSV or Excel files in the specified folder.

    Args:
        data_folder (str): Path to the folder containing CSV or Excel files.

    Returns:
        pd.DataFrame: A unified DataFrame with normalized columns and derived features.

    Examples:
        >>> df = load_and_prepare_data("path/to/dt")
        >>> print(df.shape)
        (1000, 10)
    """
    logger.info(f"Loading data from folder: {data_folder}")
    
    # Find all CSV and Excel files
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    excel_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
    all_files = csv_files + excel_files
    logger.info(f"Found {len(all_files)} files")
    
    all_dfs = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            df.columns = [' '.join(col.strip().split()) for col in df.columns]  # Normalize column names
            
            # Determine orbit class from filename
            if 'GEO' in filename.upper():
                orbit_class = 'GEO'
            elif 'MEO' in filename.upper():
                orbit_class = 'MEO'
            elif 'GSO' in filename.upper():
                orbit_class = 'GSO'
            else:
                orbit_class = 'Unknown'
            
            # Map columns to standard names
            column_mapping = {}
            possible_mappings = {
                'timestamp': ['utc_time', 'timestamp'],
                'ephemeris_error_x': ['x_error (m)', 'x_error(m)'],
                'ephemeris_error_y': ['y_error (m)', 'y_error(m)'],
                'ephemeris_error_z': ['z_error (m)', 'z_error(m)'],
                'clock_error': ['satclockerror (m)', 'satclockerror(m)']
            }
            
            for standard, possibles in possible_mappings.items():
                for poss in possibles:
                    if poss in df.columns:
                        column_mapping[poss] = standard
                        break
            
            df = df.rename(columns=column_mapping)
            
            # Calculate orbit_error_3d if not present
            if 'orbit_error_3d' not in df.columns:
                df['orbit_error_3d'] = np.sqrt(
                    df['ephemeris_error_x']**2 + 
                    df['ephemeris_error_y']**2 + 
                    df['ephemeris_error_z']**2
                )
            
            # FIX: Check if 'satellite' column exists (multi-satellite file)
            if 'satellite' in df.columns:
                # File contains multiple satellites - use satellite column
                logger.info(f"  Found 'satellite' column with {df['satellite'].nunique()} unique satellites")
                df['satellite_id'] = orbit_class + '_' + df['satellite'].astype(str)
            else:
                # File represents single satellite - use filename
                sat_id = f"{orbit_class}_{filename.replace('.csv', '').replace('.xlsx', '').split('_')[-1]}"
                df['satellite_id'] = sat_id
            
            # Add orbit_class
            df['orbit_class'] = orbit_class
            
            # Convert timestamp
            formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y%m%d%H%M%S', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M']
            parsed = False
            for fmt in formats:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                    parsed = True
                    break
                except:
                    continue
            if not parsed:
                raise ValueError(f"Could not parse timestamps in {filename} with any known format")
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'satellite_id', 'orbit_class', 
                             'clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                             'ephemeris_error_z', 'orbit_error_3d']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {filename}: {missing_cols}")
                for col in missing_cols:
                    df[col] = 0.0
            
            logger.info(f"Loaded {len(df)} rows from {filename} ({orbit_class})")
            all_dfs.append(df[required_cols])
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue
    
    if not all_dfs:
        raise ValueError("No data could be loaded from files")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows, {combined_df['satellite_id'].nunique()} satellites")
    
    # Sort by satellite and timestamp
    combined_df = combined_df.sort_values(['satellite_id', 'timestamp'])
    
    # Validate data span (expected 7 days)
    time_span = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
    if time_span != 7:
        logger.warning(f"Data span is {time_span} days, expected 7 days")
    
    # Validate clock error units
    combined_df = validate_clock_error(combined_df)
    
    # Validate dataset against problem requirements
    validation_result = validate_dataset_requirements(combined_df)
    if not validation_result['valid']:
        logger.warning("Dataset does not fully meet problem requirements. Preprocessing will attempt to fix issues.")
    
    return combined_df

def validate_clock_error(df: pd.DataFrame) -> pd.DataFrame:
    clock_col = 'clock_error'
    if clock_col in df.columns:
        max_val = df[clock_col].abs().max()
        if max_val < 1e-3:  # Likely in seconds
            logger.warning("Clock errors appear to be in seconds, converting to meters")
            df[clock_col] = df[clock_col] * 3e8  # Speed of light
    return df

def validate_dataset_requirements(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate dataset against problem requirements:
    - 7 days of data
    - 15-minute intervals
    - Expected number of records
    
    Returns:
        Dict with 'valid', 'issues', and 'warnings' keys
    """
    issues = []
    warnings = []
    
    # Check overall duration
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    if duration_days < 7:
        issues.append(f"⚠️ Duration: {duration_days} days (expected: 7 days)")
        logger.warning(f"Dataset duration is {duration_days} days, expected 7 days")
    
    # Check per satellite
    expected_records_per_sat = 7 * 24 * 4  # 7 days * 24 hrs * 4 (15-min intervals/hr) = 672
    
    for sat_id in df['satellite_id'].unique():
        sat_data = df[df['satellite_id'] == sat_id].sort_values('timestamp')
        actual_records = len(sat_data)
        
        # Check record count
        if actual_records < expected_records_per_sat * 0.5:
            shortfall_pct = ((expected_records_per_sat - actual_records) / expected_records_per_sat) * 100
            warnings.append(f"{sat_id}: {actual_records} records ({shortfall_pct:.0f}% below expected {expected_records_per_sat})")
        
        # Check 15-minute interval consistency
        intervals = sat_data['timestamp'].diff().dt.total_seconds() / 60
        intervals = intervals.dropna()
        
        if len(intervals) > 0:
            consistent_15min = ((intervals >= 14) & (intervals <= 16)).sum()
            consistency_pct = (consistent_15min / len(intervals) * 100)
            mean_interval = intervals.mean()
            median_interval = intervals.median()
            
            if consistency_pct < 80:
                warnings.append(f"{sat_id}: Only {consistency_pct:.1f}% intervals are ~15min (mean: {mean_interval:.1f}min, median: {median_interval:.1f}min)")
                logger.info(f"Satellite {sat_id}: Mean interval = {mean_interval:.1f} minutes (expected: 15 minutes)")
        
        # Check for duplicates
        duplicates = sat_data.duplicated(subset=['timestamp']).sum()
        if duplicates > 0:
            warnings.append(f"{sat_id}: {duplicates} duplicate timestamps detected")
    
    # Log all issues and warnings
    if issues or warnings:
        logger.warning("="*80)
        logger.warning("DATASET VALIDATION REPORT")
        logger.warning("="*80)
        
        if issues:
            logger.warning("CRITICAL ISSUES:")
            for issue in issues:
                logger.warning(f"  {issue}")
        
        if warnings:
            logger.warning("WARNINGS:")
            for warning in warnings:
                logger.warning(f"  {warning}")
        
        logger.warning("="*80)
        logger.warning("Note: Data will be resampled to 15-minute intervals as required by problem statement")
        logger.warning("="*80)
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }

# ==================== Enhanced Data Preprocessing ====================
class RobustDataPreprocessor:
    """Handles all edge cases in GNSS data preprocessing with enhanced features"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scalers = {}
        self.outlier_masks = {}
        self.transform_params = {}
        self.feature_statistics = {}
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df
        df = df.copy()
        
        logger.debug(f"Initial DataFrame shape: {df.shape}")
        df = self._handle_timestamps(df)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        df = self._normalize_by_satellite(df)
        df = self._handle_data_gaps(df)
        df = self._add_advanced_temporal_features(df)
        df = self._add_physics_features(df)
        df = self._apply_normality_transforms(df, is_fit=True)
        df = self._add_interaction_features(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        logger.debug(f"Preprocessed DataFrame shape: {df.shape}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df
        df = df.copy()
        
        df = self._handle_timestamps(df)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        df = self._normalize_by_satellite_transform(df)
        df = self._handle_data_gaps(df)
        df = self._add_advanced_temporal_features(df)
        df = self._add_physics_features(df)
        df = self._apply_normality_transforms(df, is_fit=False)
        df = self._add_interaction_features(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    # ...existing code for all preprocessing methods...
    def _detect_dominant_interval(self, df: pd.DataFrame, sat_id: str) -> pd.Timedelta:
        """
        Detect the most common interval for a satellite to handle irregular sampling.
        
        Args:
            df: DataFrame with timestamp column
            sat_id: Satellite identifier
            
        Returns:
            Most common time interval as Timedelta
        """
        sat_data = df[df['satellite_id'] == sat_id].sort_values('timestamp')
        intervals = sat_data['timestamp'].diff().dropna()
        
        if len(intervals) == 0:
            return pd.Timedelta(minutes=15)
        
        # Round intervals to nearest 5 minutes to group similar intervals
        intervals_minutes = (intervals.dt.total_seconds() / 60).round(0)
        
        # Find most common interval
        interval_counts = intervals_minutes.value_counts()
        if len(interval_counts) > 0:
            dominant_minutes = interval_counts.index[0]
            logger.info(f"Satellite {sat_id}: detected dominant interval = {dominant_minutes:.0f} minutes")
            return pd.Timedelta(minutes=dominant_minutes)
        
        return pd.Timedelta(minutes=15)  # fallback
    
    def _normalize_by_satellite_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        error_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                     'ephemeris_error_z', 'orbit_error_3d']
        
        for sat_id in df['satellite_id'].unique():
            sat_mask = df['satellite_id'] == sat_id
            for col in error_cols:
                if col in df.columns:
                    values = df.loc[sat_mask, col].values.reshape(-1, 1)
                    key = f"{sat_id}_{col}"
                    if key in self.scalers and self.scalers[key] is not None:
                        scaler = self.scalers[key]
                        scaled = scaler.transform(values)
                        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
                        df.loc[sat_mask, f"{col}"] = scaled.flatten()
                    else:
                        df.loc[sat_mask, f"{col}"] = 0.0
        return df
    
    def _handle_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['satellite_id', 'timestamp'])
        df = df.drop_duplicates(subset=['satellite_id', 'timestamp'], keep='last')
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        error_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                     'ephemeris_error_z', 'orbit_error_3d']
        for col in error_cols:
            if col in df.columns:
                # Vectorized forward and backward fill per satellite
                df[col] = df.groupby('satellite_id')[col].ffill(limit=4).bfill(limit=4)
                # Interpolate remaining NaNs
                df[col] = df.groupby('satellite_id')[col].transform(
                    lambda x: x.interpolate(method='spline', order=3, limit_direction='both').fillna(0)
                )
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        error_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                     'ephemeris_error_z', 'orbit_error_3d']
        outlier_counts = {}
        for sat_id in df['satellite_id'].unique():
            sat_mask = df['satellite_id'] == sat_id
            for col in error_cols:
                if col in df.columns:
                    values = df.loc[sat_mask, col].values
                    if np.all(np.isnan(values)):
                        continue
                    
                    median = np.nanmedian(values)
                    mad = np.nanmedian(np.abs(values - median))
                    
                    if mad > 0:
                        modified_z_scores = 0.6745 * (values - median) / mad
                        outliers_mad = np.abs(modified_z_scores) > self.config.outlier_threshold
                        
                        q1, q3 = np.nanpercentile(values, [25, 75])
                        iqr = q3 - q1
                        outliers_iqr = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
                        
                        outliers_combined = outliers_mad & outliers_iqr
                        
                        outlier_counts[f"{sat_id}_{col}"] = np.sum(outliers_combined)
                        
                        if outliers_combined.any():
                            lower = np.nanpercentile(values[~outliers_combined], 2)
                            upper = np.nanpercentile(values[~outliers_combined], 98)
                            df.loc[sat_mask, col] = np.clip(df.loc[sat_mask, col], lower, upper)
                            self.outlier_masks[f"{sat_id}_{col}"] = outliers_combined
        logger.info(f"Outliers detected and clipped: {outlier_counts}")
        return df
    
    def _normalize_by_satellite(self, df: pd.DataFrame) -> pd.DataFrame:
        error_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                     'ephemeris_error_z', 'orbit_error_3d']
        
        for sat_id in df['satellite_id'].unique():
            sat_mask = df['satellite_id'] == sat_id
            for col in error_cols:
                if col in df.columns:
                    values = df.loc[sat_mask, col].values.reshape(-1, 1)
                    
                    if np.all(np.isnan(values)) or np.nanstd(values) < 1e-10:
                        df.loc[sat_mask, f"{col}"] = 0.0
                        self.scalers[f"{sat_id}_{col}"] = None
                        continue
                    
                    # Clean values first
                    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Calculate robust statistics
                    self.feature_statistics[f"{sat_id}_{col}"] = {
                        'mean': np.nanmean(clean_values),
                        'std': np.nanstd(clean_values),
                        'median': np.nanmedian(clean_values),
                        'mad': np.nanmedian(np.abs(clean_values - np.nanmedian(clean_values))),
                        'q25': np.nanpercentile(clean_values, 25),
                        'q75': np.nanpercentile(clean_values, 75)
                    }
                    
                    # Use more conservative quantile range for better normality
                    scaler = RobustScaler(quantile_range=(10, 90))
                    scaled = scaler.fit_transform(clean_values)
                    
                    # Apply Box-Cox-like transformation to encourage normality
                    scaled_flat = scaled.flatten()
                    if np.min(scaled_flat) <= 0:
                        # Shift to positive values
                        shifted = scaled_flat - np.min(scaled_flat) + 1e-6
                    else:
                        shifted = scaled_flat
                    
                    # Apply log transformation if beneficial for normality
                    try:
                        log_transformed = np.log1p(np.abs(shifted)) * np.sign(shifted)
                        # Test normality of both versions
                        if len(log_transformed) > 3:
                            from scipy.stats import normaltest
                            _, p_orig = normaltest(scaled_flat)
                            _, p_log = normaltest(log_transformed)
                            
                            if p_log > p_orig and p_log > 0.01:  # Use log if better normality
                                scaled_final = log_transformed
                            else:
                                scaled_final = scaled_flat
                        else:
                            scaled_final = scaled_flat
                    except:
                        scaled_final = scaled_flat
                    
                    # Final clipping to prevent extreme values
                    scaled_final = np.clip(scaled_final, -4, 4)
                    scaled_final = np.nan_to_num(scaled_final, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    df.loc[sat_mask, f"{col}"] = scaled_final
                    self.scalers[f"{sat_id}_{col}"] = scaler
        
        return df
    
    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['orbital_period'] = df['orbit_class'].map({'MEO': 12.0, 'GEO': 24.0, 'GSO': 24.0, 'Unknown': 24.0})
        df['mean_motion'] = 2 * np.pi / (df['orbital_period'] * 3600)
        df['grav_effect'] = 1.0 / (df['orbital_period'] ** (2/3))
        df['srp_proxy'] = np.abs(df['time_sin']) * df['orbit_class'].map({'MEO': 1.0, 'GEO': 1.5, 'GSO': 1.5, 'Unknown': 1.0})
        df['srp_seasonal'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25) * df['srp_proxy']
        df['j2_effect'] = df['orbit_class'].map({'MEO': 1.5, 'GEO': 0.3, 'GSO': 0.3, 'Unknown': 0.3})
        df['drag_proxy'] = df['orbit_class'].map({'MEO': 0.1, 'GEO': 0.01, 'GSO': 0.01, 'Unknown': 0.01})
        
        for sat_id in df['satellite_id'].unique():
            sat_mask = df['satellite_id'] == sat_id
            if 'clock_error' in df.columns:
                clock_vals = df.loc[sat_mask, 'clock_error'].values
                if len(clock_vals) > 2 and not np.all(np.isnan(clock_vals)):
                    clock_diff = np.diff(clock_vals)
                    allan_dev = np.sqrt(np.nanmean(clock_diff**2) / 2) if np.any(clock_diff) else 0.0
                    df.loc[sat_mask, 'clock_stability'] = allan_dev
                    
                    if len(clock_vals) > 10:
                        valid_indices = np.arange(len(clock_vals))[~np.isnan(clock_vals)]
                        valid_vals = clock_vals[~np.isnan(clock_vals)]
                        if len(valid_vals) > 10:
                            clock_drift = np.polyfit(valid_indices, valid_vals, 1)[0]
                            df.loc[sat_mask, 'clock_drift_rate'] = clock_drift
                        else:
                            df.loc[sat_mask, 'clock_drift_rate'] = 0.0
                    else:
                        df.loc[sat_mask, 'clock_stability'] = 0.0
                        df.loc[sat_mask, 'clock_drift_rate'] = 0.0
        
        df['relativistic_effect'] = df['grav_effect'] * 0.01
        return df
    
    def _add_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        lag_windows = [1, 2, 4, 8, 12]
        error_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                     'ephemeris_error_z']
        
        for sat_id in df['satellite_id'].unique():
            sat_mask = df['satellite_id'] == sat_id
            sat_data = df.loc[sat_mask].copy()
            
            for col in error_cols:
                if col in df.columns:
                    for lag in lag_windows:
                        df.loc[sat_mask, f'{col}_lag_{lag}'] = sat_data[col].shift(lag).fillna(0)
                    
                    df.loc[sat_mask, f'{col}_roll_mean_12'] = sat_data[col].rolling(12, min_periods=1).mean().fillna(0)
                    df.loc[sat_mask, f'{col}_roll_std_12'] = sat_data[col].rolling(12, min_periods=1).std().fillna(0)
                    df.loc[sat_mask, f'{col}_ewma'] = sat_data[col].ewm(span=12, adjust=False).mean().fillna(0)
        
        df['time_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        return df
    
    def _advanced_interpolate(self, sat_data: pd.DataFrame, column: str, max_gap_periods: int) -> pd.Series:
        """
        Advanced interpolation using multiple methods for near-real data generation.
        Combines spline interpolation with physics-aware constraints.
        """
        series = sat_data[column].copy()
        
        if series.isna().sum() == 0:
            return series
        
        # Get orbit class for physics-aware interpolation
        orbit_class = sat_data['orbit_class'].iloc[0] if 'orbit_class' in sat_data.columns else 'Unknown'
        
        # Check number of real data points
        n_real = (~series.isna()).sum()
        
        # Step 1: Choose interpolation method based on available data
        if n_real >= 4:
            # Enough points for cubic spline
            try:
                series_spline = series.interpolate(method='spline', order=3, limit=max_gap_periods)
            except Exception:
                # Spline failed, try quadratic
                try:
                    series_spline = series.interpolate(method='quadratic', limit=max_gap_periods)
                except Exception:
                    # Fall back to linear
                    series_spline = series.interpolate(method='linear', limit=max_gap_periods)
        elif n_real >= 2:
            # Use linear interpolation for few points
            series_spline = series.interpolate(method='linear', limit=max_gap_periods)
        else:
            # Not enough points, use forward/backward fill
            series_spline = series.ffill().bfill()
            return series_spline
        
        # Step 2: Apply physics-aware smoothing
        # Different error types have different temporal characteristics
        if 'clock' in column.lower():
            # Clock errors tend to drift smoothly - use lower frequency smoothing
            window_size = min(7, max_gap_periods)  # Smooth over ~1.75 hours
            series_smoothed = series_spline.rolling(window=window_size, center=True, min_periods=1).mean()
        elif 'ephemeris' in column.lower() or 'orbit' in column.lower():
            # Orbital errors have periodic components
            window_size = min(5, max_gap_periods)  # Shorter window for orbital dynamics
            series_smoothed = series_spline.rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            series_smoothed = series_spline
        
        # Step 3: Preserve real measurements (don't smooth actual data points)
        original_mask = ~series.isna()
        series_smoothed[original_mask] = series[original_mask]
        
        # Step 4: Add small realistic noise to synthetic points
        synthetic_mask = series.isna()
        if synthetic_mask.any() and original_mask.any():
            # Estimate noise level from real data
            real_values = series[original_mask].values
            if len(real_values) > 2:
                noise_std = np.std(np.diff(real_values)) * 0.1  # 10% of typical variation
                noise = np.random.normal(0, noise_std, synthetic_mask.sum())
                series_smoothed.loc[synthetic_mask] += noise
        
        # Step 5: Final gap filling for any remaining NaNs
        series_smoothed = series_smoothed.ffill(limit=2).bfill(limit=2).fillna(0)
        
        return series_smoothed
    
    def _handle_data_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Smart resampling: Only resample to 15-minute intervals if enabled and needed.
        Uses advanced interpolation methods to generate near-real synthetic data.
        """
        if df.empty:
            return df
        
        # Check if resampling is enabled
        if not self.config.enable_resampling:
            logger.info("Resampling disabled - using original intervals")
            return df
        
        filled_dfs = []
        target_interval = pd.Timedelta(minutes=15)  # Problem requirement
        max_gap_hours = self.config.max_interpolation_gap_hours
        
        for sat_id in df['satellite_id'].unique():
            sat_data = df[df['satellite_id'] == sat_id].copy()
            if sat_data.empty:
                continue
            
            sat_data = sat_data.sort_values('timestamp')
            
            # Detect dominant interval
            dominant_interval = self._detect_dominant_interval(df, sat_id)
            
            # Check if resampling is needed
            interval_diff_minutes = abs(dominant_interval.total_seconds() - target_interval.total_seconds()) / 60
            
            if interval_diff_minutes > 2:  # More than 2 minutes difference from 15-min target
                logger.info(f"Satellite {sat_id}: Resampling from {dominant_interval.total_seconds()/60:.1f}min to 15min intervals")
                logger.info(f"  Using {self.config.interpolation_method} interpolation method")
                
                start_time = sat_data['timestamp'].min()
                end_time = sat_data['timestamp'].max()
                regular_times = pd.date_range(start=start_time, end=end_time, freq='15min')
                
                # Store metadata before reindexing
                orbit_class = sat_data['orbit_class'].iloc[0] if len(sat_data) > 0 else 'Unknown'
                original_timestamps = set(sat_data['timestamp'].values)
                
                sat_data = sat_data.set_index('timestamp')
                sat_data = sat_data.reindex(regular_times)
                
                # Calculate interpolation limits
                max_gap_periods = int((dominant_interval.total_seconds() / 60) / 15)  # e.g., 2hr = 8 periods
                max_gap_hours_periods = int((max_gap_hours * 60) / 15)  # Convert max hours to periods
                max_gap_periods = max(1, min(max_gap_periods, max_gap_hours_periods))
                
                logger.info(f"  Interpolation limit: {max_gap_periods} periods ({max_gap_periods * 15} minutes)")
                
                # Track which points are synthetic
                if self.config.mark_synthetic_points:
                    sat_data['is_real_measurement'] = sat_data.index.isin(original_timestamps).astype(int)
                
                # Apply advanced interpolation to numeric columns
                numeric_cols = [col for col in sat_data.select_dtypes(include=[np.number]).columns 
                               if col not in ['is_real_measurement']]
                
                original_count = len(df[df['satellite_id']==sat_id])
                synthetic_count = 0
                
                for col in numeric_cols:
                    if self.config.interpolation_method == 'advanced':
                        # Use physics-aware advanced interpolation
                        sat_data[col] = self._advanced_interpolate(sat_data, col, max_gap_periods)
                    elif self.config.interpolation_method == 'spline':
                        # Cubic spline interpolation
                        try:
                            sat_data[col] = sat_data[col].interpolate(method='spline', order=3, limit=max_gap_periods)
                        except:
                            sat_data[col] = sat_data[col].interpolate(method='linear', limit=max_gap_periods)
                        sat_data[col] = sat_data[col].ffill(limit=2).bfill(limit=2).fillna(0)
                    else:
                        # Linear interpolation (default)
                        sat_data[col] = sat_data[col].interpolate(method='linear', limit=max_gap_periods)
                        sat_data[col] = sat_data[col].ffill(limit=2).bfill(limit=2).fillna(0)
                
                # Restore categorical columns
                sat_data['satellite_id'] = sat_id
                sat_data['orbit_class'] = orbit_class
                
                sat_data = sat_data.reset_index().rename(columns={'index': 'timestamp'})
                
                # Recalculate time features
                sat_data['hour'] = sat_data['timestamp'].dt.hour
                sat_data['day_of_year'] = sat_data['timestamp'].dt.dayofyear
                
                if 'is_real_measurement' in sat_data.columns:
                    synthetic_count = (sat_data['is_real_measurement'] == 0).sum()
                    real_pct = (original_count / len(sat_data)) * 100
                    synthetic_pct = (synthetic_count / len(sat_data)) * 100
                    logger.info(f"  Result: {len(sat_data)} records ({original_count} real [{real_pct:.1f}%], {synthetic_count} synthetic [{synthetic_pct:.1f}%])")
                else:
                    logger.info(f"  Result: Resampled from {original_count} to {len(sat_data)} records")
            else:
                # Data is already at ~15-minute intervals, just fill small gaps
                logger.info(f"Satellite {sat_id}: Already at ~15min intervals, filling small gaps only")
                
                # Only fill very small gaps (< 30 minutes)
                time_diffs = sat_data['timestamp'].diff()
                small_gaps = (time_diffs > pd.Timedelta(minutes=16)) & (time_diffs < pd.Timedelta(minutes=30))
                
                if small_gaps.any():
                    numeric_cols = sat_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        sat_data[col] = sat_data[col].interpolate(method='linear', limit=2)
            
            filled_dfs.append(sat_data)
        
        if not filled_dfs:
            return pd.DataFrame(columns=df.columns)
        
        result_df = pd.concat(filled_dfs, ignore_index=True)
        logger.info(f"Total records after gap handling: {len(result_df)}")
        return result_df
    
    def _apply_normality_transforms(self, df: pd.DataFrame, is_fit: bool = True) -> pd.DataFrame:
        error_cols = ['clock_error', 'ephemeris_error_x', 
                     'ephemeris_error_y', 'ephemeris_error_z']
        
        # IMPROVEMENT: Apply log-transform to orbit_error_3d for normality
        # orbit_3d is always positive (sqrt of sum of squares) -> log-normal behavior
        if 'orbit_error_3d' in df.columns:
            values = df['orbit_error_3d'].values
            valid_mask = (~np.isnan(values)) & (values > 0)
            log_values = np.zeros_like(values)
            log_values[valid_mask] = np.log(values[valid_mask] + 1e-6)
            log_values[~valid_mask] = 0
            df['orbit_error_3d_log'] = log_values
            
            # Store original for inverse transform
            if is_fit:
                self.transform_params['orbit_error_3d_log'] = {'method': 'log', 'offset': 1e-6}
        
        if is_fit:
            for col in error_cols:
                if col in df.columns:
                    values = df[col].values
                    valid_values = values[~np.isnan(values)]
                    
                    if len(valid_values) < 10:
                        df[f"{col}_transformed"] = values
                        self.transform_params[col] = None
                        continue
                    
                    test_sample = valid_values[:min(5000, len(valid_values))]
                    _, p_value = stats.shapiro(test_sample)
                    
                    if p_value < self.config.normality_test_alpha:
                        best_transform = None
                        best_p_value = p_value
                        
                        try:
                            min_val = valid_values.min()
                            shift = abs(min_val) + 1 if min_val <= 0 else 0
                            transformed, lambda_param = boxcox(valid_values + shift)
                            _, new_p = stats.shapiro(transformed[:5000])
                            if new_p > best_p_value:
                                best_transform = ('boxcox', lambda_param, shift)
                                best_p_value = new_p
                        except:
                            pass
                        
                        try:
                            transformed, lambda_param = stats.yeojohnson(valid_values)
                            _, new_p = stats.shapiro(transformed[:5000])
                            if new_p > best_p_value:
                                best_transform = ('yeojohnson', lambda_param, None)
                                best_p_value = new_p
                        except:
                            pass
                        
                        if best_transform:
                            method, lambda_param, shift = best_transform
                            transformed = np.zeros_like(values)
                            transformed[~np.isnan(values)] = boxcox(values[~np.isnan(values)] + shift, lambda_param) if method == 'boxcox' else stats.yeojohnson(values[~np.isnan(values)], lambda_param)
                            transformed[np.isnan(values)] = 0
                            df[f"{col}_transformed"] = transformed
                            self.transform_params[col] = {'method': method, 'lambda': lambda_param, 'shift': shift}
                        else:
                            df[f"{col}_transformed"] = values
                            self.transform_params[col] = None
                    else:
                        df[f"{col}_transformed"] = values
                        self.transform_params[col] = None
        else:
            for col in error_cols:
                if col in df.columns and col in self.transform_params and self.transform_params[col]:
                    params = self.transform_params[col]
                    method = params['method']
                    values = df[col].values
                    transformed = np.zeros_like(values)
                    valid_mask = ~np.isnan(values)
                    if method == 'boxcox':
                        shift = params['shift']
                        lambda_param = params['lambda']
                        transformed[valid_mask] = boxcox(values[valid_mask] + shift, lambda_param)
                    else:
                        lambda_param = params['lambda']
                        transformed[valid_mask] = stats.yeojohnson(values[valid_mask], lambda_param)
                    transformed[~valid_mask] = 0
                    df[f"{col}_transformed"] = transformed
                elif col in df.columns:
                    df[f"{col}_transformed"] = df[col].fillna(0)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'clock_error' in df.columns and 'ephemeris_error_x' in df.columns:
            df['clock_ephemeris_interaction'] = df['clock_error'] * df['ephemeris_error_x']
        
        orbit_encoding = {'MEO': 0, 'GEO': 1, 'GSO': 2, 'Unknown': 0}
        df['orbit_encoded'] = df['orbit_class'].map(orbit_encoding)
        
        if 'grav_effect' in df.columns:
            df['orbit_grav_interaction'] = df['orbit_encoded'] * df['grav_effect']
        
        if 'srp_proxy' in df.columns:
            df['orbit_srp_interaction'] = df['orbit_encoded'] * df['srp_proxy']
        
        return df

# ==================== Enhanced Dataset with Augmentation ====================
class GNSSDataset(Dataset):
    """PyTorch dataset for GNSS error prediction with data augmentation"""
    
    def __init__(self, df: pd.DataFrame, config: ModelConfig, mode: str = 'train'):
        self.df = df
        self.config = config
        self.mode = mode
        # Exclude synthetic marker from input features
        self.input_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                          if col not in ['satellite_id', 'is_real_measurement']]
        self.target_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                            'ephemeris_error_z', 'orbit_error_3d']
        self.sequences = []
        self.targets = []
        self.sequence_indices = []  # Store indices for test set mapping
        self.real_data_ratios = []  # Track how much real vs synthetic data in each sequence
        self._prepare_sequences()
        
        # Report data quality
        if 'is_real_measurement' in self.df.columns:
            total_real = (self.df['is_real_measurement'] == 1).sum()
            total_points = len(self.df)
            real_pct = (total_real / total_points * 100) if total_points > 0 else 0
            logger.info(f"Dataset mode: {mode}, Sequences: {len(self.sequences)}, "
                       f"Real data: {real_pct:.1f}% ({total_real}/{total_points})")
        else:
            logger.info(f"Dataset mode: {mode}, Number of sequences: {len(self.sequences)}")
    
    def _prepare_sequences(self):
        """
        Create sequences with TIME-BASED horizon targets (not index-based).
        Problem requires predictions at: 15min, 30min, 1hr, 2hr, 24hr into the future.
        """
        # Define horizons in MINUTES as per problem statement
        horizon_definitions = {
            '15min': 15,
            '30min': 30,
            '1hr': 60,
            '2hr': 120,
            '4hr': 240,
            '8hr': 480,
            '24hr': 1440
        }
        
        for sat_id in self.df['satellite_id'].unique():
            sat_mask = self.df['satellite_id'] == sat_id
            sat_data = self.df[sat_mask].sort_values('timestamp').copy()
            
            if len(sat_data) < self.config.sequence_length:
                logger.warning(f"Satellite {sat_id} has insufficient data points ({len(sat_data)} < {self.config.sequence_length})")
                continue
            
            sat_input = sat_data[self.input_cols].values
            sat_target = sat_data[self.target_cols].values
            sat_times = pd.to_datetime(sat_data['timestamp'].values)
            sat_indices = sat_data.index.values
            
            # Use stride of 1 for maximum coverage (can be adjusted for training speed)
            stride = 1 if self.mode == 'train' else 1
            
            for i in range(0, len(sat_input) - self.config.sequence_length + 1, stride):
                seq = sat_input[i:i + self.config.sequence_length]
                seq_end_time = sat_times[i + self.config.sequence_length - 1]
                
                # Track real vs synthetic data ratio if marked
                if 'is_real_measurement' in sat_data.columns:
                    seq_real_flags = sat_data['is_real_measurement'].iloc[i:i + self.config.sequence_length].values
                    real_ratio = seq_real_flags.mean() if len(seq_real_flags) > 0 else 0
                else:
                    real_ratio = 1.0  # Assume all real if not marked
                
                # Find targets based on TIME, not index
                targets = {}
                has_required_horizons = True
                
                for horizon_name, horizon_minutes in horizon_definitions.items():
                    # Calculate target time
                    target_time = seq_end_time + pd.Timedelta(minutes=horizon_minutes)
                    
                    # Find the data point closest to target_time
                    time_diffs_seconds = np.abs((sat_times - target_time).total_seconds())
                    nearest_idx = np.argmin(time_diffs_seconds)
                    
                    # Only use target if it's within acceptable tolerance (10 minutes)
                    if time_diffs_seconds[nearest_idx] < 600:  # 10 minutes tolerance
                        targets[horizon_name] = sat_target[nearest_idx]
                    else:
                        # Mark as unavailable
                        targets[horizon_name] = None
                        # For training, we need at least the first few horizons
                        if horizon_name in ['15min', '30min', '1hr'] and self.mode == 'train':
                            has_required_horizons = False
                
                # Only keep sequences that have the minimum required horizons for training
                if self.mode == 'train' and not has_required_horizons:
                    continue
                
                # For test mode, keep all sequences but mark missing targets
                # Replace None with zero vectors for missing targets
                for horizon_name in targets:
                    if targets[horizon_name] is None:
                        targets[horizon_name] = np.zeros(len(self.target_cols))
                
                seq = np.nan_to_num(seq, nan=0.0)
                self.sequences.append(seq)
                self.targets.append(targets)
                self.sequence_indices.append(sat_indices[i + self.config.sequence_length - 1])
                self.real_data_ratios.append(real_ratio)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()
        targets = self.targets[idx].copy()
        
        if self.mode == 'train':
            if np.random.random() < 0.5:
                noise = np.random.normal(0, self.config.augmentation_factor, seq.shape)
                seq = seq + noise
            
            if np.random.random() < 0.3:
                warp_factor = np.random.uniform(0.9, 1.1)
                indices = np.linspace(0, len(seq)-1, int(len(seq) * warp_factor))
                seq = np.array([np.interp(np.arange(len(seq[0])), np.arange(len(seq[0])), seq[int(min(i, len(seq)-1))]) 
                                for i in indices])
                if len(seq) > len(self.sequences[idx]):
                    seq = seq[:len(self.sequences[idx])]
                elif len(seq) < len(self.sequences[idx]):
                    pad_length = len(self.sequences[idx]) - len(seq)
                    seq = np.vstack([seq, np.tile(seq[-1], (pad_length, 1))])
        
        seq = torch.FloatTensor(seq)
        targets = {h: torch.FloatTensor(t) for h, t in targets.items()}
        
        return seq, targets, self.sequence_indices[idx]

# ==================== Data Handling Functions ====================
def ensure_sufficient_data(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    sufficient_sats = [
        sat for sat, group in df.groupby('satellite_id')
        if len(group) >= min_samples
    ]
    filtered_df = df[df['satellite_id'].isin(sufficient_sats)]
    total_sats = len(df['satellite_id'].unique())
    kept_sats = len(sufficient_sats)
    logger.info(f"Filtered to {kept_sats}/{total_sats} satellites with >= {min_samples} samples")
    
    if kept_sats == 0:
        logger.warning("No satellites meet minimum sample requirement, lowering threshold")
        min_samples = max(10, min_samples // 2)
        sufficient_sats = [
            sat for sat, group in df.groupby('satellite_id')
            if len(group) >= min_samples
        ]
        filtered_df = df[df['satellite_id'].isin(sufficient_sats)]
        logger.info(f"With reduced threshold ({min_samples}): {len(sufficient_sats)} satellites")
    
    return filtered_df

def create_robust_data_split(df: pd.DataFrame, config: ModelConfig):
    """
    Create TIME-BASED data split as per problem requirements:
    - Train on first N-2 days
    - Validation on N-1 day
    - Test on last day (simulating prediction for day 8)
    
    This ensures we're predicting future time points, not just random samples.
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for sat, group in df.groupby('satellite_id'):
        group = group.sort_values('timestamp')
        n = len(group)
        
        if n < config.sequence_length + 10:
            logger.warning(f"Satellite {sat} has insufficient data ({n} samples), skipping")
            continue
        
        # Get time boundaries
        start_time = group['timestamp'].min()
        end_time = group['timestamp'].max()
        total_duration = (end_time - start_time).total_seconds() / 3600  # hours
        total_days = (end_time - start_time).days
        
        logger.info(f"Satellite {sat}: {n} samples over {total_days} days ({total_duration/24:.1f} days)")
        
        if total_days < 3:
            logger.warning(f"Satellite {sat}: only {total_days} days of data, need at least 3 days for proper split")
            # Use percentage-based split for very short data
            test_size = max(int(n * 0.2), 1)
            val_size = max(int(n * 0.15), 1)
            train_size = n - test_size - val_size
            
            if train_size >= config.sequence_length:
                train_dfs.append(group.iloc[:train_size])
                val_dfs.append(group.iloc[train_size:train_size + val_size])
                test_dfs.append(group.iloc[train_size + val_size:])
            continue
        
        # TIME-BASED SPLIT (preferred method)
        # Problem: Train on 7 days, test on day 8
        # Our data: ~6-7 days, so split as: Train (70%), Val (15%), Test (15%)
        # But use TIME boundaries, not sample counts
        
        # Calculate time cutoffs
        if total_days >= 6:
            # Enough data for proper temporal split
            train_cutoff = start_time + pd.Timedelta(days=max(4, int(total_days * 0.7)))
            val_cutoff = start_time + pd.Timedelta(days=max(5, int(total_days * 0.85)))
        else:
            # Short data, split more conservatively
            train_cutoff = start_time + pd.Timedelta(days=max(2, int(total_days * 0.65)))
            val_cutoff = start_time + pd.Timedelta(days=max(3, int(total_days * 0.80)))
        
        # Split by time
        train_data = group[group['timestamp'] < train_cutoff]
        val_data = group[(group['timestamp'] >= train_cutoff) & (group['timestamp'] < val_cutoff)]
        test_data = group[group['timestamp'] >= val_cutoff]
        
        # Ensure minimum sizes
        if len(train_data) >= config.sequence_length + 10:
            train_dfs.append(train_data)
            
            if len(val_data) >= 5:
                val_dfs.append(val_data)
            else:
                logger.warning(f"Satellite {sat}: validation set too small ({len(val_data)} samples), skipping val")
            
            if len(test_data) >= 5:
                test_dfs.append(test_data)
            else:
                logger.warning(f"Satellite {sat}: test set too small ({len(test_data)} samples), skipping test")
        else:
            logger.warning(f"Satellite {sat}: insufficient training data ({len(train_data)} samples)")
    
    if not train_dfs:
        raise ValueError("No satellites have sufficient data for time-based splitting. Check data quality.")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    # Log split statistics
    logger.info("="*80)
    logger.info("TIME-BASED DATA SPLIT SUMMARY")
    logger.info("="*80)
    if len(train_df) > 0:
        logger.info(f"Train: {len(train_df)} samples from {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    if len(val_df) > 0:
        logger.info(f"Val:   {len(val_df)} samples from {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    if len(test_df) > 0:
        logger.info(f"Test:  {len(test_df)} samples from {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    logger.info("="*80)
    
    return train_df, val_df, test_df


def create_kfold_splits(df: pd.DataFrame, config: ModelConfig):
    """
    Create k-fold cross-validation splits for time-series data.
    Uses TIME-SERIES AWARE splitting to prevent data leakage.
    
    For the problem statement:
    - Each fold trains on past data, validates on future data
    - This mimics the real scenario: predict day 8 based on days 1-7
    - Ensures temporal ordering is preserved
    
    Returns:
        List of (train_df, val_df) tuples, one for each fold
        test_df is kept separate for final evaluation
    """
    logger.info(f"Creating {config.n_folds}-fold cross-validation splits...")
    
    # First, extract the test set (final day) - this is NEVER used in CV
    all_train_val_dfs = []
    test_dfs = []
    
    for sat, group in df.groupby('satellite_id'):
        group = group.sort_values('timestamp')
        n = len(group)
        
        if n < config.sequence_length + 10:
            logger.warning(f"Satellite {sat}: insufficient data ({n} samples), skipping")
            continue
        
        # Reserve last 30% for final test set (increased from 15% to get n>50 test sequences)
        # With 33 satellites and ~30% test data, we get multiple sequences per satellite
        test_cutoff_idx = int(n * 0.70)
        train_val_data = group.iloc[:test_cutoff_idx]
        test_data = group.iloc[test_cutoff_idx:]
        
        if len(train_val_data) >= config.sequence_length + config.n_folds:
            all_train_val_dfs.append(train_val_data)
            # Lower threshold to 3 to capture more test sequences
            if len(test_data) >= 3:
                test_dfs.append(test_data)
    
    if not all_train_val_dfs:
        raise ValueError("Insufficient data for k-fold cross-validation")
    
    # Combine train+val data
    combined_train_val = pd.concat(all_train_val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    # Now create k-fold splits on the train+val portion
    folds = []
    
    if config.kfold_mode == 'time_series':
        # TIME-SERIES AWARE K-FOLD
        # Split data temporally: each fold uses progressively later data for validation
        for sat, group in combined_train_val.groupby('satellite_id'):
            group = group.sort_values('timestamp')
            n = len(group)
            
            fold_size = n // config.n_folds
            if fold_size < config.sequence_length:
                logger.warning(f"Satellite {sat}: insufficient data for {config.n_folds} folds")
                continue
            
            sat_folds = []
            for fold_idx in range(config.n_folds):
                # Validation fold is a sliding window through time
                val_start = fold_idx * fold_size
                val_end = (fold_idx + 1) * fold_size if fold_idx < config.n_folds - 1 else n
                
                # Training data is all data BEFORE validation window
                # This ensures we never train on future data
                if val_start >= config.sequence_length:
                    train_data = group.iloc[:val_start]
                    val_data = group.iloc[val_start:val_end]
                    sat_folds.append((train_data, val_data))
            
            # Aggregate across satellites for each fold
            for fold_idx in range(len(sat_folds)):
                if fold_idx >= len(folds):
                    folds.append(([], []))
                folds[fold_idx][0].append(sat_folds[fold_idx][0])
                folds[fold_idx][1].append(sat_folds[fold_idx][1])
    
    else:
        # STANDARD K-FOLD (not recommended for time-series, but available)
        # Shuffle and split (use with caution - may leak future information)
        logger.warning("Using standard k-fold (not time-series aware). Consider using kfold_mode='time_series'")
        
        shuffled = combined_train_val.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(shuffled)
        fold_size = n // config.n_folds
        
        for fold_idx in range(config.n_folds):
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < config.n_folds - 1 else n
            
            val_indices = list(range(val_start, val_end))
            train_indices = list(range(0, val_start)) + list(range(val_end, n))
            
            folds.append((
                [shuffled.iloc[train_indices]],
                [shuffled.iloc[val_indices]]
            ))
    
    # Convert to DataFrames
    final_folds = []
    for fold_idx, (train_list, val_list) in enumerate(folds):
        if train_list and val_list:
            train_fold = pd.concat(train_list, ignore_index=True)
            val_fold = pd.concat(val_list, ignore_index=True)
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_fold)} samples, Val={len(val_fold)} samples")
            final_folds.append((train_fold, val_fold))
    
    logger.info(f"Created {len(final_folds)} folds for cross-validation")
    logger.info(f"Test set (held out): {len(test_df)} samples")
    
    return final_folds, test_df

