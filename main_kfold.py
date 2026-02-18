"""
GNSS Error Prediction System - K-Fold Cross-Validation Training
Uses k-fold CV to maximize calibrator training data and ensure robust evaluation.
Per problem statement: Predict day 8 errors, residuals should follow normal distribution (Shapiro-Wilk test)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch

from config import ModelConfig
from data_utils import load_and_prepare_data, ensure_sufficient_data, create_kfold_splits
from model import HybridGNSSModel
from evaluation_utils import evaluate_model, save_metrics_to_json, save_predictions_to_csv, \
    plot_predictions, plot_residuals, print_evaluation_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Loggs/training_kfold.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_single_fold(fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      config: ModelConfig, save_model: bool = False):
    """Train model on a single fold"""
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING FOLD {fold_idx + 1}/{config.n_folds}")
    logger.info(f"{'='*80}")
    
    # Initialize model
    model = HybridGNSSModel(config)
    
    # Train
    model.train(train_df, val_df)
    
    # Save model if requested (typically only for final fold or best fold)
    if save_model:
        model_folder = "saved_model"
        os.makedirs(model_folder, exist_ok=True)
        checkpoint_path = os.path.join(model_folder, f'gnss_hybrid_fold{fold_idx}_checkpoint.pt')
        model._save_checkpoint('all', model_folder)
        logger.info(f"Fold {fold_idx + 1} model saved to: {checkpoint_path}")
    
    return model


def aggregate_predictions(fold_predictions: List[Dict], config: ModelConfig) -> Dict:
    """Aggregate predictions from all folds using ensemble averaging"""
    logger.info("Aggregating predictions from all folds...")
    
    aggregated = {}
    
    for horizon in config.prediction_horizons:
        # Collect all fold predictions for this horizon
        all_preds = []
        all_uncs = []
        all_indices = []
        
        for fold_pred in fold_predictions:
            if horizon in fold_pred:
                all_preds.append(fold_pred[horizon]['predictions'])
                all_uncs.append(fold_pred[horizon]['uncertainties'])
                all_indices.append(fold_pred[horizon]['indices'])
        
        if not all_preds:
            logger.warning(f"No predictions for horizon {horizon}")
            continue
        
        # Stack and average predictions
        # Note: indices should be the same across folds for the test set
        predictions = np.mean(all_preds, axis=0)
        
        # Combine uncertainties (using variance averaging)
        # Total uncertainty = mean prediction variance + variance of predictions
        pred_variance = np.mean([u**2 for u in all_uncs], axis=0)
        prediction_spread = np.var(all_preds, axis=0)
        uncertainties = np.sqrt(pred_variance + prediction_spread)
        
        indices = all_indices[0]  # Should be the same for all folds
        
        aggregated[horizon] = {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'indices': indices,
            'lower_bound': predictions - 1.96 * uncertainties,
            'upper_bound': predictions + 1.96 * uncertainties
        }
        
        logger.info(f"Horizon {horizon}: {len(predictions)} aggregated predictions")
    
    return aggregated


def run_kfold_training(data_folder: str = None, config: ModelConfig = None):
    """
    Run k-fold cross-validation training pipeline.
    
    Process:
    1. Split data into k folds (time-series aware)
    2. Train k models, each on different train/val split
    3. Aggregate predictions on held-out test set
    4. Evaluate on test set with focus on normality test (problem requirement)
    """
    
    try:
        # Setup
        if config is None:
            config = ModelConfig()
        
        if not config.use_kfold:
            logger.warning("use_kfold=False in config. Use main.py for single split training.")
            config.use_kfold = True
        
        if data_folder is None:
            data_folder = "dataset"
        
        logger.info("="*80)
        logger.info("GNSS ERROR PREDICTION SYSTEM - K-FOLD CROSS-VALIDATION")
        logger.info("="*80)
        logger.info(f"Data folder: {data_folder}")
        logger.info(f"K-Folds: {config.n_folds}")
        logger.info(f"Mode: {config.kfold_mode}")
        logger.info("="*80)
        
        # Step 1: Load data
        logger.info("Step 1: Loading and preparing data...")
        df = load_and_prepare_data(data_folder)
        logger.info(f"Loaded {len(df)} total samples from {df['satellite_id'].nunique()} satellites")
        
        # Step 2: Filter insufficient data
        logger.info("Step 2: Filtering satellites with sufficient data...")
        df = ensure_sufficient_data(df, config.min_samples_per_satellite)
        logger.info(f"After filtering: {len(df)} samples from {df['satellite_id'].nunique()} satellites")
        
        # Step 3: Create k-fold splits
        logger.info("Step 3: Creating k-fold cross-validation splits...")
        folds, test_df = create_kfold_splits(df, config)
        
        if test_df.empty:
            logger.error("Test set is empty!")
            return None, None, None
        
        logger.info(f"Created {len(folds)} folds, test set: {len(test_df)} samples")
        
        # Step 4: Train models on each fold
        logger.info("Step 4: Training models on each fold...")
        fold_models = []
        
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            # Save model only for the last fold (has most recent data)
            save_model = (fold_idx == len(folds) - 1)
            model = train_single_fold(fold_idx, train_fold, val_fold, config, save_model=save_model)
            fold_models.append(model)
        
        logger.info(f"Completed training {len(fold_models)} models")
        
        # Step 4.5: Compute bias corrections using validation data from all folds
        logger.info("Step 4.5: Computing bias corrections from validation data...")
        
        # Define target columns
        target_cols = ['clock_error', 'ephemeris_error_x', 'ephemeris_error_y', 
                      'ephemeris_error_z', 'orbit_error_3d']
        
        all_val_predictions = []
        all_val_targets = []
        
        for fold_idx, (model, (train_fold, val_fold)) in enumerate(zip(fold_models, folds)):
            logger.info(f"Computing bias from fold {fold_idx + 1} validation set...")
            val_preds = model.predict(val_fold, return_uncertainty=True)
            val_targets = model.prepare_test_targets(val_fold, target_cols)
            all_val_predictions.append(val_preds)
            all_val_targets.append(val_targets)
        
        # Compute bias corrections for ensemble
        logger.info("Computing ensemble bias corrections...")
        for model in fold_models:
            model.compute_bias_corrections(all_val_predictions, all_val_targets, config)
        
        # Step 5: Generate predictions on test set from all folds
        logger.info("Step 5: Generating predictions from all folds on test set...")
        fold_predictions = []
        
        for fold_idx, model in enumerate(fold_models):
            logger.info(f"Generating predictions from fold {fold_idx + 1}...")
            predictions = model.predict(test_df, return_uncertainty=True)
            fold_predictions.append(predictions)
        
        # Step 6: Aggregate predictions
        logger.info("Step 6: Aggregating predictions from all folds...")
        aggregated_predictions = aggregate_predictions(fold_predictions, config)
        
        # Step 7: Prepare test targets
        logger.info("Step 7: Preparing test targets...")
        test_targets = fold_models[0].prepare_test_targets(test_df, target_cols)
        
        # Step 8: Evaluate aggregated predictions
        logger.info("Step 8: Evaluating aggregated predictions...")
        logger.info("="*80)
        logger.info("*** EVALUATION FOCUS: NORMALITY TEST (Problem Requirement) ***")
        logger.info("Per problem statement: Residuals should follow normal distribution")
        logger.info("This indicates systematic errors have been removed, only random errors remain")
        logger.info("="*80)
        
        metrics, residuals_dict, orbit_metrics = evaluate_model(
            aggregated_predictions, 
            test_targets, 
            config, 
            test_df
        )
        
        if not metrics:
            logger.warning("No metrics computed - evaluation failed")
            return fold_models, aggregated_predictions, None
        
        # Step 9: Save results
        logger.info("Step 9: Saving results...")
        results_folder = "results"
        os.makedirs(results_folder, exist_ok=True)
        
        # Save metrics
        save_metrics_to_json(metrics, orbit_metrics, os.path.join(results_folder, "evaluation_metrics_kfold.json"))
        
        # Save predictions for each horizon (saves multiple CSVs)
        predictions_base = os.path.join(results_folder, "predictions_kfold")
        save_predictions_to_csv(aggregated_predictions, test_df, config, predictions_base)
        
        # Generate plots
        logger.info("Step 10: Generating visualizations...")
        
        # Plot predictions - pass full dictionaries, config, and save path
        try:
            pred_plot_path = os.path.join(results_folder, "prediction_plots_kfold")
            plot_predictions(aggregated_predictions, test_targets, config, pred_plot_path)
        except Exception as e:
            logger.warning(f"Could not generate prediction plots: {e}")
        
        # Plot residuals - pass residuals_dict from evaluate_model and save path
        if residuals_dict:
            try:
                residual_plot_path = os.path.join(results_folder, "residual_plots_kfold")
                plot_residuals(residuals_dict, residual_plot_path)
            except Exception as e:
                logger.warning(f"Could not generate residual plots: {e}")
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("K-FOLD CROSS-VALIDATION EVALUATION SUMMARY")
        logger.info("="*80)
        print_evaluation_summary(metrics, orbit_metrics)
        
        # Additional k-fold specific summary
        logger.info("\n" + "="*80)
        logger.info("K-FOLD TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Number of folds: {config.n_folds}")
        logger.info(f"Splitting mode: {config.kfold_mode}")
        logger.info(f"Total training samples across folds: {sum(len(folds[i][0]) for i in range(len(folds)))}")
        logger.info(f"Test samples (held out): {len(test_df)}")
        logger.info("="*80)
        
        # Normality test summary (FIXED: Component-wise evaluation)
        logger.info("\n" + "="*80)
        logger.info("*** NORMALITY TEST SUMMARY (Component-wise Evaluation) ***")
        logger.info("="*80)
        logger.info("Per problem statement: Residuals should follow normal distribution")
        logger.info("Method: Test each component separately, pass if >=3/5 components pass")
        logger.info("-"*80)
        
        normal_count = sum(1 for m in metrics.values() if m.get('is_normal', False))
        total_horizons = len(metrics)
        
        logger.info(f"Horizons with NORMAL residuals: {normal_count}/{total_horizons}")
        logger.info(f"Pass rate: {normal_count/total_horizons*100:.1f}%")
        
        if normal_count == total_horizons:
            logger.info("[SUCCESS] All horizons show normal residuals!")
            logger.info("  Model successfully removed systematic errors across all prediction windows")
        elif normal_count >= total_horizons * 0.7:
            logger.info("[GOOD] Majority of horizons show normal residuals")
            logger.info("  Model effectively removes most systematic errors")
        else:
            logger.info("[NEEDS IMPROVEMENT] Many horizons show non-normal residuals")
            logger.info("  Consider: more training data, hyperparameter tuning, or feature engineering")
        
        logger.info("="*80)
        
        logger.info("\n" + "="*80)
        logger.info("K-FOLD CROSS-VALIDATION PIPELINE COMPLETED!")
        logger.info("="*80)
        
        return fold_models, aggregated_predictions, metrics
    
    except Exception as e:
        logger.error(f"K-fold training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Run k-fold cross-validation training
    config = ModelConfig()
    models, predictions, metrics = run_kfold_training(config=config)
    
    if metrics:
        logger.info("Training completed successfully!")
        
        # Print final normality assessment (Component-wise)
        logger.info("\n" + "="*80)
        logger.info("FINAL ASSESSMENT: NORMALITY OF RESIDUALS (Component-wise)")
        logger.info("="*80)
        
        for horizon_key, horizon_metrics in metrics.items():
            horizon_str = horizon_key.split('_')[1]
            p_val = horizon_metrics['normality_p_value']
            is_normal = horizon_metrics['is_normal']
            n_pass = horizon_metrics.get('components_passing_normality', 0)
            
            status = "[PASS]" if is_normal else "[FAIL]"
            logger.info(f"{horizon_str:>6s}: {status} Components: {n_pass}/5 pass (avg_p={p_val:.4f})")
        
        logger.info("="*80)
    else:
        logger.error("Training failed - no metrics available")
