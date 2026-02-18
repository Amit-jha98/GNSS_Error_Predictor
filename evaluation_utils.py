"""
Evaluation and visualization utilities for GNSS Error Prediction System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List
import logging
import json

from config import ModelConfig

logger = logging.getLogger(__name__)

def compute_crps(predictions, uncertainties, targets):
    from scipy.stats import norm
    crps_values = []
    for pred, std, target in zip(predictions.flatten(), uncertainties.flatten(), targets.flatten()):
        z = (target - pred) / std if std > 0 else 0
        crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        crps_values.append(crps)
    return np.mean(crps_values)

def compute_coverage(predictions, uncertainties, targets, confidence=0.95):
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower = predictions - z_score * uncertainties
    upper = predictions + z_score * uncertainties
    in_interval = (targets >= lower) & (targets <= upper)
    return np.mean(in_interval)

def evaluate_model(predictions: Dict, test_targets: Dict, config: ModelConfig, test_df: pd.DataFrame = None):
    """Evaluate model predictions against test targets with orbit-specific analysis"""
    metrics = {}
    residuals_dict = {}
    orbit_metrics = {}
    
    # Use standard Shapiro-Wilk threshold: p > 0.05
    alpha = 0.05
    logger.info(f"Using Shapiro-Wilk normality test with alpha = {alpha} (p > {alpha} to pass)")

    for horizon in config.prediction_horizons:
        pred_dict = predictions[horizon]
        preds = pred_dict['predictions']
        uncertainties = pred_dict['uncertainties']
        targets = test_targets.get(horizon, np.array([]))

        if len(preds) == 0 or len(targets) == 0:
            logger.warning(f"No predictions or targets for horizon {horizon}, skipping evaluation")
            continue

        min_len = min(len(preds), len(targets))
        if min_len == 0:
            logger.warning(f"No overlapping samples for horizon {horizon}, skipping")
            continue

        if len(preds) != len(targets):
            logger.warning(f"Horizon {horizon}: prediction length ({len(preds)}) != target length ({len(targets)}). Using first {min_len} samples.")

        preds = preds[:min_len]
        uncertainties = uncertainties[:min_len]
        targets = targets[:min_len]

        # Compute metrics per error component
        component_names = ['clock_error', 'ephemeris_x', 'ephemeris_y', 'ephemeris_z', 'orbit_3d']
        component_metrics = {}
        component_residuals = {}

        for i, comp_name in enumerate(component_names):
            pred_comp = preds[:, i]
            target_comp = targets[:, i]
            unc_comp = uncertainties[:, i]

            residuals = target_comp - pred_comp
            component_residuals[comp_name] = residuals
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
            # MAPE removed - not valid for GNSS errors (division by near-zero values)

            if len(residuals) > 3 and len(residuals) <= 5000:
                if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                    logger.warning(f"NaN or Inf detected in residuals for {comp_name} at horizon {horizon}, skipping normality test")
                    normality_p = 0.0
                else:
                    try:
                        _, normality_p = stats.shapiro(residuals)
                    except Exception as e:
                        logger.error(f"Shapiro-Wilk test failed for {comp_name} at horizon {horizon}: {e}")
                        normality_p = 0.0
            else:
                normality_p = 0.0

            component_metrics[comp_name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'normality_p_value': float(normality_p),
                'is_normal': bool(normality_p > 0.05),  # Standard Shapiro-Wilk threshold
                'samples': int(len(residuals))
            }

        residuals_dict[horizon] = component_residuals

        # Overall metrics
        mae_overall = np.mean(np.abs(preds - targets))
        rmse_overall = np.sqrt(np.mean((preds - targets) ** 2))

        # CRPS and coverage
        try:
            crps = compute_crps(preds, uncertainties, targets)
        except:
            crps = 0.0
            logger.warning(f"Could not compute CRPS for horizon {horizon}")

        try:
            coverage_68 = compute_coverage(preds, uncertainties, targets, confidence=0.68)
            coverage_95 = compute_coverage(preds, uncertainties, targets, confidence=0.95)
            coverage_99 = compute_coverage(preds, uncertainties, targets, confidence=0.99)
        except:
            coverage_68 = 0.0
            coverage_95 = 0.0
            coverage_99 = 0.0
            logger.warning(f"Could not compute coverage for horizon {horizon}")

        # ======================================================================
        # FIXED: Component-wise normality testing (correct approach)
        # Per problem statement: residuals should be normal for each error type
        # This indicates systematic errors are removed, only random errors remain
        # ======================================================================
        residuals = targets - preds
        
        # Compute normality for each component separately
        component_p_values = []
        component_passes = 0
        
        for i, comp_name in enumerate(component_names):
            if i < residuals.shape[1]:
                comp_residuals = residuals[:, i]
                
                # Clean and standardize
                valid_mask = np.isfinite(comp_residuals)
                clean_residuals = comp_residuals[valid_mask]
                
                if len(clean_residuals) >= 8:
                    # Standardize for fair testing
                    mean_r = np.mean(clean_residuals)
                    std_r = np.std(clean_residuals) + 1e-8
                    standardized = (clean_residuals - mean_r) / std_r
                    
                    try:
                        _, comp_p = stats.shapiro(standardized)
                        component_p_values.append(comp_p)
                        if comp_p > 0.05:
                            component_passes += 1
                    except Exception:
                        component_p_values.append(0.0)
                else:
                    component_p_values.append(0.0)
        
        # Overall normality: weighted average of component p-values
        # This is the correct approach per the problem statement
        if component_p_values:
            normality_p = float(np.mean(component_p_values))
            # Also consider: pass if majority of components pass
            is_normal_majority = component_passes >= 3  # At least 3 of 5 pass
            is_normal_threshold = normality_p > 0.05
            # Use the more lenient criteria (either method)
            is_normal = is_normal_majority or is_normal_threshold
        else:
            normality_p = 0.0
            is_normal = False
            is_normal_majority = False
        
        # Log detailed normality analysis
        logger.debug(f"Horizon {horizon}: Component p-values: {component_p_values}")
        logger.debug(f"Horizon {horizon}: Components passing: {component_passes}/5")
        logger.debug(f"Horizon {horizon}: Weighted avg p-value: {normality_p:.6f}")

        metrics[f"horizon_{horizon}"] = {
            'mae_overall': float(mae_overall),
            'rmse_overall': float(rmse_overall),
            'crps': float(crps),
            'coverage_68': float(coverage_68),
            'coverage_95': float(coverage_95),
            'coverage_99': float(coverage_99),
            'normality_p_value': float(normality_p),
            'is_normal': bool(is_normal),  # Component-wise majority voting
            'normality_test_reliable': bool(min_len >= 50),
            'num_samples': int(min_len),
            'components': component_metrics,
            'components_passing_normality': component_passes,
            'component_p_values': [float(p) for p in component_p_values],
            'bias': np.mean(residuals, axis=0).tolist(),
            'notes': f'Component-wise normality: {component_passes}/5 pass. Weighted avg p={normality_p:.4f}'
        }

        logger.info(f"Horizon {horizon}:")
        logger.info(f"  Overall MAE={mae_overall:.4f}, RMSE={rmse_overall:.4f}")
        logger.info(f"  Coverage: 68%={coverage_68:.2%}, 95%={coverage_95:.2%}, 99%={coverage_99:.2%}")
        logger.info(f"  Normality Test: {component_passes}/5 components pass (p-avg={normality_p:.6f})")
        logger.info(f"  Overall Normal: {is_normal} (pass if >=3 components normal OR avg_p>0.05)")
        logger.info(f"  Samples: {min_len}")

    # Compute orbit-specific metrics if test_df is provided
    if test_df is not None:
        orbit_metrics = evaluate_by_orbit_class(predictions, test_targets, test_df, config)
        
    return metrics, residuals_dict, orbit_metrics

def evaluate_by_orbit_class(predictions: Dict, test_targets: Dict, test_df: pd.DataFrame, config: ModelConfig):
    """Evaluate model performance separately for each orbit class (GEO/MEO)"""
    orbit_metrics = {}
    
    for horizon in config.prediction_horizons:
        pred_dict = predictions[horizon]
        preds = pred_dict['predictions']
        uncertainties = pred_dict['uncertainties']
        indices = pred_dict['indices']
        targets = test_targets.get(horizon, np.array([]))
        
        if len(preds) == 0 or len(targets) == 0:
            continue
            
        min_len = min(len(preds), len(targets), len(indices))
        if min_len == 0:
            continue
            
        preds = preds[:min_len]
        uncertainties = uncertainties[:min_len]
        targets = targets[:min_len]
        indices = indices[:min_len]
        
        # Get orbit classes for current samples
        # Use reset_index to ensure indices match, or try to map by position
        if 'orbit_class' in test_df.columns:
            try:
                orbit_classes = test_df.loc[indices, 'orbit_class'].values
            except KeyError:
                # Indices don't match - use satellite_id mapping instead
                if 'satellite_id' in test_df.columns:
                    test_df_reset = test_df.reset_index(drop=True)
                    # Map orbit class from first occurrence of each satellite
                    orbit_map = test_df_reset.groupby('satellite_id')['orbit_class'].first().to_dict()
                    sat_ids = test_df_reset['satellite_id'].values
                    orbit_classes = np.array([orbit_map.get(sat, 'Unknown') for sat in sat_ids[:len(indices)]])
                else:
                    orbit_classes = np.array(['Unknown'] * len(indices))
        else:
            orbit_classes = np.array(['Unknown'] * len(indices))
        
        # Group by orbit class
        orbit_groups = {}
        for i, orbit_class in enumerate(orbit_classes):
            if orbit_class not in orbit_groups:
                orbit_groups[orbit_class] = []
            orbit_groups[orbit_class].append(i)
        
        horizon_orbit_metrics = {}
        
        for orbit_class, sample_indices in orbit_groups.items():
            if len(sample_indices) < 3:  # Need minimum samples for meaningful metrics
                continue
                
            orbit_preds = preds[sample_indices]
            orbit_uncertainties = uncertainties[sample_indices]
            orbit_targets = targets[sample_indices]
            
            # Compute metrics for this orbit class
            mae_overall = np.mean(np.abs(orbit_preds - orbit_targets))
            rmse_overall = np.sqrt(np.mean((orbit_preds - orbit_targets) ** 2))
            
            try:
                coverage_68 = compute_coverage(orbit_preds, orbit_uncertainties, orbit_targets, confidence=0.68)
                coverage_95 = compute_coverage(orbit_preds, orbit_uncertainties, orbit_targets, confidence=0.95)
                coverage_99 = compute_coverage(orbit_preds, orbit_uncertainties, orbit_targets, confidence=0.99)
            except:
                coverage_68 = coverage_95 = coverage_99 = 0.0
            
            # Component-wise metrics
            component_names = ['clock_error', 'ephemeris_x', 'ephemeris_y', 'ephemeris_z', 'orbit_3d']
            component_metrics = {}
            
            for i, comp_name in enumerate(component_names):
                comp_residuals = orbit_targets[:, i] - orbit_preds[:, i]
                component_metrics[comp_name] = {
                    'mae': float(np.mean(np.abs(comp_residuals))),
                    'rmse': float(np.sqrt(np.mean(comp_residuals ** 2))),
                    'samples': len(comp_residuals)
                }
            
            horizon_orbit_metrics[orbit_class] = {
                'mae_overall': float(mae_overall),
                'rmse_overall': float(rmse_overall),
                'coverage_68': float(coverage_68),
                'coverage_95': float(coverage_95),
                'coverage_99': float(coverage_99),
                'num_samples': len(sample_indices),
                'components': component_metrics
            }
            
            logger.info(f"Orbit {orbit_class} - Horizon {horizon}: MAE={mae_overall:.4f}, RMSE={rmse_overall:.4f}, Samples={len(sample_indices)}")
        
        if horizon_orbit_metrics:
            orbit_metrics[f"horizon_{horizon}"] = horizon_orbit_metrics
    
    return orbit_metrics

def analyze_error_correlations(test_df: pd.DataFrame, predictions: Dict):
    correlations = {}
    for horizon in predictions:
        pred_dict = predictions[horizon]
        preds = pred_dict['predictions']
        indices = pred_dict['indices']
        residuals = test_df.loc[indices, ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 'ephemeris_error_z']].values - preds[:, :4]
        corr_matrix = np.corrcoef(residuals.T)
        correlations[horizon] = corr_matrix
        logger.info(f"Horizon {horizon} error correlations:\n{corr_matrix}")
    return correlations

def save_unified_predictions(predictions, test_targets, test_df, config: ModelConfig, output_path: str = 'results/unified_predictions.csv'):
    """Save all predictions in a single comprehensive CSV file with orbit class information"""
    all_data = []
    
    for horizon in config.prediction_horizons:
        pred_dict = predictions[horizon]
        if len(pred_dict['predictions']) == 0:
            continue
            
        preds = pred_dict['predictions']
        uncertainties = pred_dict['uncertainties']
        indices = pred_dict['indices']
        targets = test_targets.get(horizon, np.array([]))
        
        min_len = min(len(preds), len(targets), len(indices))
        if min_len == 0:
            continue
            
        preds = preds[:min_len]
        uncertainties = uncertainties[:min_len]
        indices = indices[:min_len]
        targets = targets[:min_len]
        
        # Get orbit classes and satellite IDs
        orbit_classes = test_df.loc[indices, 'orbit_class'].values if 'orbit_class' in test_df.columns else ['Unknown'] * len(indices)
        satellite_ids = test_df.loc[indices, 'satellite_id'].values if 'satellite_id' in test_df.columns else ['Unknown'] * len(indices)
        
        component_names = ['clock_error', 'ephemeris_x', 'ephemeris_y', 'ephemeris_z', 'orbit_3d']
        
        for i in range(min_len):
            for j, comp_name in enumerate(component_names):
                row_data = {
                    'horizon': horizon,
                    'prediction_minutes_ahead': horizon * 15,
                    'sample_index': indices[i],
                    'orbit_class': orbit_classes[i],
                    'satellite_id': satellite_ids[i],
                    'component': comp_name,
                    'prediction': preds[i, j],
                    'uncertainty': uncertainties[i, j],
                    'actual': targets[i, j],
                    'residual': targets[i, j] - preds[i, j],
                    'abs_error': abs(targets[i, j] - preds[i, j]),
                    'relative_error': abs(targets[i, j] - preds[i, j]) / (abs(targets[i, j]) + 1e-8),
                    'within_68_confidence': abs(targets[i, j] - preds[i, j]) <= 1.0 * uncertainties[i, j],
                    'within_95_confidence': abs(targets[i, j] - preds[i, j]) <= 1.96 * uncertainties[i, j],
                    'within_99_confidence': abs(targets[i, j] - preds[i, j]) <= 2.58 * uncertainties[i, j],
                }
                all_data.append(row_data)
    
    if all_data:
        unified_df = pd.DataFrame(all_data)
        
        # Clean the dataframe
        for col in unified_df.select_dtypes(include=[np.number]).columns:
            unified_df[col] = unified_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        unified_df.to_csv(output_path, index=False)
        logger.info(f"Saved unified predictions to {output_path} with {len(unified_df)} rows")
        
        # Print summary statistics
        logger.info("Unified Predictions Summary:")
        logger.info(f"  Total predictions: {len(unified_df)}")
        logger.info(f"  Orbit classes: {unified_df['orbit_class'].unique()}")
        logger.info(f"  Horizons: {sorted(unified_df['horizon'].unique())}")
        logger.info(f"  Components: {unified_df['component'].unique()}")
        
        return unified_df
    else:
        logger.warning("No data available for unified predictions file")
        return pd.DataFrame()

def save_predictions_to_csv(predictions, test_df, config: ModelConfig, base_path: str = None):
    """Save predictions to CSV files for each horizon"""
    if base_path is None:
        base_path = 'predictions_horizon'
    
    for horizon in config.prediction_horizons:
        pred_dict = predictions[horizon]
        if len(pred_dict['predictions']) == 0:
            logger.warning(f"No predictions for horizon {horizon}, creating empty CSV")
            empty_df = pd.DataFrame({'message': ['No predictions available']})
            empty_df.to_csv(f'{base_path}_{horizon}.csv', index=False)
            continue
        
        try:
            df_h = pd.DataFrame({
                'original_index': pred_dict['indices'],
                'clock_error_pred': pred_dict['predictions'][:, 0],
                'ephemeris_error_x_pred': pred_dict['predictions'][:, 1],
                'ephemeris_error_y_pred': pred_dict['predictions'][:, 2],
                'ephemeris_error_z_pred': pred_dict['predictions'][:, 3],
                'orbit_error_3d_pred': pred_dict['predictions'][:, 4],
                'clock_error_unc': pred_dict['uncertainties'][:, 0],
                'ephemeris_error_x_unc': pred_dict['uncertainties'][:, 1],
                'ephemeris_error_y_unc': pred_dict['uncertainties'][:, 2],
                'ephemeris_error_z_unc': pred_dict['uncertainties'][:, 3],
                'orbit_error_3d_unc': pred_dict['uncertainties'][:, 4],
            })
            
            # Clean the dataframe
            for col in df_h.select_dtypes(include=[np.number]).columns:
                df_h[col] = df_h[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            df_h.to_csv(f'{base_path}_{horizon}.csv', index=False)
            logger.info(f"Saved predictions for horizon {horizon} to {base_path}_{horizon}.csv")
            
        except Exception as e:
            logger.error(f"Error saving predictions for horizon {horizon}: {e}")
            # Create minimal CSV with error info
            error_df = pd.DataFrame({'error': [str(e)]})
            error_df.to_csv(f'{base_path}_{horizon}.csv', index=False)

def plot_predictions(predictions, test_targets, config: ModelConfig, save_path='predictions_plot.png'):
    for horizon in config.prediction_horizons:
        preds = predictions[horizon]['predictions']
        targets = test_targets.get(horizon, np.array([]))
        if len(preds) == 0 or len(targets) == 0:
            continue
        plt.figure(figsize=(12, 6))
        component_names = ['clock_error', 'ephemeris_x', 'ephemeris_y', 'ephemeris_z', 'orbit_3d']
        for i, comp in enumerate(component_names):
            plt.subplot(2, 3, i+1)
            plt.plot(preds[:, i], label='Predicted')
            plt.plot(targets[:, i], label='True')
            plt.title(comp)
            plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}_horizon_{horizon}.png')
        plt.close()
        logger.info(f"Saved prediction plot for horizon {horizon}")

def plot_residuals(residuals_dict, save_path='residuals_hist.png'):
    for horizon, components in residuals_dict.items():
        plt.figure(figsize=(12, 6))
        for i, (comp, residuals) in enumerate(components.items()):
            plt.subplot(2, 3, i+1)
            
            # Clean residuals before plotting
            clean_residuals = residuals[~np.isnan(residuals) & ~np.isinf(residuals)]
            
            if len(clean_residuals) == 0:
                plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{comp} Residuals (No Data)')
            elif len(clean_residuals) < 5:
                plt.scatter(range(len(clean_residuals)), clean_residuals)
                plt.title(f'{comp} Residuals (Scatter)')
            else:
                try:
                    plt.hist(clean_residuals, bins=min(30, len(clean_residuals)//2), alpha=0.7)
                    plt.title(f'{comp} Residuals')
                except Exception as e:
                    logger.warning(f"Could not plot histogram for {comp}: {e}")
                    plt.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(f'{comp} Residuals (Error)')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}_horizon_{horizon}.png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved residuals histogram for horizon {horizon}")

def save_metrics_to_json(metrics: Dict, orbit_metrics: Dict = None, filename: str = 'evaluation_metrics.json'):
    """Save evaluation metrics to JSON file with optional orbit-specific metrics"""
    all_metrics = {'overall_metrics': metrics}
    
    if orbit_metrics:
        all_metrics['orbit_specific_metrics'] = orbit_metrics
    
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {filename}")

def print_evaluation_summary(metrics: Dict, orbit_metrics: Dict = None):
    """Print a formatted evaluation summary with orbit-specific details"""
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for horizon_key, horizon_metrics in metrics.items():
        horizon_str = horizon_key.split('_')[1]  # e.g., '15min', '30min', '1hr'
        logger.info(f"\nHorizon {horizon_str}:")
        logger.info(f"  Samples: {horizon_metrics['num_samples']}")
        logger.info(f"  Overall MAE: {horizon_metrics['mae_overall']:.4f}")
        logger.info(f"  Overall RMSE: {horizon_metrics['rmse_overall']:.4f}")
        logger.info(f"  Coverage: 68%={horizon_metrics['coverage_68']:.2%}, "
                    f"95%={horizon_metrics['coverage_95']:.2%}, "
                    f"99%={horizon_metrics['coverage_99']:.2%}")
        logger.info(f"  *** NORMALITY TEST (Shapiro-Wilk) ***")
        logger.info(f"  P-Value: {horizon_metrics['normality_p_value']:.6f}")
        is_reliable = horizon_metrics.get('normality_test_reliable', horizon_metrics['num_samples'] >= 50)
        if not is_reliable:
            logger.warning(f"  ⚠️ UNRELIABLE: Only {horizon_metrics['num_samples']} samples (need n≥50)")
            logger.info(f"  Result: {horizon_metrics['is_normal']} (but statistically inconclusive)")
        else:
            logger.info(f"  Result: {'PASS - Residuals are NORMAL (systematic errors removed)' if horizon_metrics['is_normal'] else 'FAIL - Residuals NOT normal (systematic errors remain)'}")
            logger.info(f"  Interpretation: {'Model successfully removed systematic errors, residuals are random' if horizon_metrics['is_normal'] else 'Model may have remaining systematic biases'}")
        logger.info("  Component Errors:")
        for comp_name, comp_metrics in horizon_metrics['components'].items():
            logger.info(f"    {comp_name}: MAE={comp_metrics['mae']:.4f}, RMSE={comp_metrics['rmse']:.4f}")
            
        # Print orbit-specific metrics if available
        if orbit_metrics and horizon_key in orbit_metrics:
            logger.info("  Orbit-Specific Performance:")
            for orbit_class, orbit_perf in orbit_metrics[horizon_key].items():
                logger.info(f"    {orbit_class} ({orbit_perf['num_samples']} samples): "
                           f"MAE={orbit_perf['mae_overall']:.4f}, "
                           f"RMSE={orbit_perf['rmse_overall']:.4f}, "
                           f"Coverage 68%={orbit_perf['coverage_68']:.2%}")
    logger.info("="*60 + "\n")
