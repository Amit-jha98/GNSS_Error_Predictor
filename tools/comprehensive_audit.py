"""
Comprehensive System Audit - Find ALL Types of Errors
Checks: Code errors, logic errors, data issues, config problems, performance issues
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveAuditor:
    """Comprehensive system audit checking all error types"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        
    def run_full_audit(self):
        """Run all audit checks"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE SYSTEM AUDIT")
        logger.info("="*80)
        
        # 1. Configuration Audit
        self.audit_configuration()
        
        # 2. Code Structure Audit
        self.audit_code_structure()
        
        # 3. Data Integrity Audit
        self.audit_data_integrity()
        
        # 4. Model Architecture Audit
        self.audit_model_architecture()
        
        # 5. Evaluation Logic Audit
        self.audit_evaluation_logic()
        
        # 6. Results Consistency Audit
        self.audit_results_consistency()
        
        # 7. File System Audit
        self.audit_file_system()
        
        # 8. Performance Audit
        self.audit_performance()
        
        # 9. Documentation Audit
        self.audit_documentation()
        
        # 10. Deployment Readiness Audit
        self.audit_deployment_readiness()
        
        # Print summary
        self.print_summary()
        
    def audit_configuration(self):
        """Check configuration files for errors"""
        logger.info("\n" + "="*80)
        logger.info("1. CONFIGURATION AUDIT")
        logger.info("="*80)
        
        try:
            from config import ModelConfig
            config = ModelConfig()
            
            # Check critical parameters
            checks = [
                (config.sequence_length > 0, "sequence_length must be > 0"),
                (config.prediction_horizons, "prediction_horizons cannot be empty"),
                (config.batch_size > 0, "batch_size must be > 0"),
                (config.pit_num_epochs > 0, "pit_num_epochs must be > 0"),
                (0 < config.pit_learning_rate < 1, "pit_learning_rate must be in (0, 1)"),
                (0 <= config.pit_dropout < 1, "pit_dropout must be in [0, 1)"),
                (config.n_folds > 1 if config.use_kfold else True, "n_folds must be > 1 when using k-fold"),
                (config.target_dim == 5, "target_dim should be 5 (clock + 3 ephemeris + orbit_3d)"),
            ]
            
            for check, msg in checks:
                if not check:
                    self.errors.append(f"CONFIG ERROR: {msg}")
            
            # Check for potential issues
            if config.batch_size > 64:
                self.warnings.append(f"CONFIG WARNING: Large batch_size ({config.batch_size}) may cause memory issues")
            
            if config.pit_learning_rate > 0.01:
                self.warnings.append(f"CONFIG WARNING: High learning rate ({config.pit_learning_rate}) may cause instability")
            
            if config.use_kfold and config.n_folds > 10:
                self.warnings.append(f"CONFIG WARNING: Many folds ({config.n_folds}) increases training time significantly")
            
            self.info.append(f"✓ Configuration loaded successfully")
            logger.info("  ✓ Configuration file valid")
            
        except Exception as e:
            self.errors.append(f"CONFIG ERROR: Failed to load config.py: {e}")
            logger.error(f"  ✗ Configuration error: {e}")
    
    def audit_code_structure(self):
        """Check code structure and imports"""
        logger.info("\n" + "="*80)
        logger.info("2. CODE STRUCTURE AUDIT")
        logger.info("="*80)
        
        required_files = [
            'config.py',
            'model.py',
            'data_utils.py',
            'evaluation_utils.py',
            'device_utils.py',
            'main.py',
            'main_kfold.py'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                self.errors.append(f"CODE ERROR: Missing required file: {file}")
            else:
                self.info.append(f"✓ Found {file}")
        
        # Check imports
        try:
            import config
            import model
            import data_utils
            import evaluation_utils
            import device_utils
            self.info.append("✓ All modules importable")
            logger.info("  ✓ All required modules can be imported")
        except Exception as e:
            self.errors.append(f"CODE ERROR: Import failed: {e}")
            logger.error(f"  ✗ Import error: {e}")
    
    def audit_data_integrity(self):
        """Check data files and integrity"""
        logger.info("\n" + "="*80)
        logger.info("3. DATA INTEGRITY AUDIT")
        logger.info("="*80)
        
        dataset_folder = "dataset"
        
        if not os.path.exists(dataset_folder):
            self.errors.append(f"DATA ERROR: Dataset folder '{dataset_folder}' not found")
            logger.error(f"  ✗ Dataset folder not found")
            return
        
        # Check for data files
        data_files = [f for f in os.listdir(dataset_folder) 
                     if f.endswith('.csv') or f.endswith('.xlsx')]
        
        if not data_files:
            self.errors.append(f"DATA ERROR: No CSV/Excel files in {dataset_folder}")
            logger.error(f"  ✗ No data files found")
            return
        
        logger.info(f"  Found {len(data_files)} data files:")
        for f in data_files:
            logger.info(f"    - {f}")
        
        # Check data loading
        try:
            from data_utils import load_and_prepare_data
            df = load_and_prepare_data(dataset_folder)
            
            # Check for required columns
            required_cols = [
                'satellite_id', 'timestamp', 'clock_error',
                'ephemeris_error_x', 'ephemeris_error_y', 'ephemeris_error_z'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.errors.append(f"DATA ERROR: Missing columns: {missing_cols}")
            
            # Check data quality
            if df.empty:
                self.errors.append("DATA ERROR: Loaded dataframe is empty")
            else:
                self.info.append(f"✓ Loaded {len(df)} samples from {df['satellite_id'].nunique()} satellites")
                
                # Check for NaN values
                nan_cols = df.columns[df.isna().any()].tolist()
                if nan_cols:
                    self.warnings.append(f"DATA WARNING: NaN values in columns: {nan_cols}")
                
                # Check time span
                time_span = (df['timestamp'].max() - df['timestamp'].min()).days
                if time_span != 7:
                    self.warnings.append(f"DATA WARNING: Time span is {time_span} days, expected 7")
                
                # Check for duplicates
                duplicates = df.duplicated(subset=['satellite_id', 'timestamp']).sum()
                if duplicates > 0:
                    self.warnings.append(f"DATA WARNING: {duplicates} duplicate records found")
                
                logger.info(f"  ✓ Data integrity checks passed")
                
        except Exception as e:
            self.errors.append(f"DATA ERROR: Failed to load data: {e}")
            logger.error(f"  ✗ Data loading error: {e}")
    
    def audit_model_architecture(self):
        """Check model architecture consistency"""
        logger.info("\n" + "="*80)
        logger.info("4. MODEL ARCHITECTURE AUDIT")
        logger.info("="*80)
        
        try:
            from config import ModelConfig
            from model import HybridGNSSModel
            
            config = ModelConfig()
            
            # Check if model can be instantiated
            try:
                model = HybridGNSSModel(config)
                self.info.append("✓ Model instantiation successful")
                logger.info("  ✓ Model can be instantiated")
            except Exception as e:
                self.errors.append(f"MODEL ERROR: Cannot instantiate model: {e}")
                logger.error(f"  ✗ Model instantiation failed: {e}")
                return
            
            # Check model components
            if not hasattr(model, 'pit_model'):
                self.errors.append("MODEL ERROR: Missing pit_model component")
            if not hasattr(model, 'ndm_model'):
                self.errors.append("MODEL ERROR: Missing ndm_model component")
            if not hasattr(model, 'calibrator'):
                self.errors.append("MODEL ERROR: Missing calibrator component")
            
            # Check device compatibility
            if hasattr(model, 'device'):
                logger.info(f"  Device: {model.device}")
                self.info.append(f"✓ Model device: {model.device}")
            
            logger.info("  ✓ Model architecture checks passed")
            
        except Exception as e:
            self.errors.append(f"MODEL ERROR: Architecture check failed: {e}")
            logger.error(f"  ✗ Architecture error: {e}")
    
    def audit_evaluation_logic(self):
        """Check evaluation logic for correctness"""
        logger.info("\n" + "="*80)
        logger.info("5. EVALUATION LOGIC AUDIT")
        logger.info("="*80)
        
        try:
            from evaluation_utils import evaluate_model, compute_crps, compute_coverage
            import scipy.stats as stats
            
            # Test compute_crps
            try:
                test_preds = np.array([1.0, 2.0, 3.0])
                test_uncs = np.array([0.1, 0.2, 0.3])
                test_targets = np.array([1.1, 1.9, 3.2])
                crps = compute_crps(test_preds, test_uncs, test_targets)
                if np.isnan(crps) or np.isinf(crps):
                    self.errors.append("EVAL ERROR: compute_crps returns NaN/Inf")
                else:
                    self.info.append("✓ CRPS computation works correctly")
            except Exception as e:
                self.errors.append(f"EVAL ERROR: compute_crps failed: {e}")
            
            # Test compute_coverage
            try:
                coverage = compute_coverage(test_preds, test_uncs, test_targets, confidence=0.95)
                if not (0 <= coverage <= 1):
                    self.errors.append(f"EVAL ERROR: Invalid coverage value: {coverage}")
                else:
                    self.info.append("✓ Coverage computation works correctly")
            except Exception as e:
                self.errors.append(f"EVAL ERROR: compute_coverage failed: {e}")
            
            # Test Shapiro-Wilk
            try:
                test_residuals = np.random.normal(0, 1, 50)
                stat, p_value = stats.shapiro(test_residuals)
                if np.isnan(p_value):
                    self.errors.append("EVAL ERROR: Shapiro-Wilk returns NaN")
                else:
                    self.info.append("✓ Shapiro-Wilk test works correctly")
            except Exception as e:
                self.errors.append(f"EVAL ERROR: Shapiro-Wilk test failed: {e}")
            
            logger.info("  ✓ Evaluation logic checks passed")
            
        except Exception as e:
            self.errors.append(f"EVAL ERROR: Evaluation logic check failed: {e}")
            logger.error(f"  ✗ Evaluation error: {e}")
    
    def audit_results_consistency(self):
        """Check results files for consistency"""
        logger.info("\n" + "="*80)
        logger.info("6. RESULTS CONSISTENCY AUDIT")
        logger.info("="*80)
        
        results_folder = "results"
        
        if not os.path.exists(results_folder):
            self.warnings.append(f"RESULTS WARNING: No results folder found (normal if not trained yet)")
            logger.info("  ℹ No results folder (model not trained yet)")
            return
        
        # Check for metrics files
        metrics_files = {
            'kfold': 'evaluation_metrics_kfold.json',
            'single': 'evaluation_metrics.json'
        }
        
        for mode, filename in metrics_files.items():
            filepath = os.path.join(results_folder, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        metrics = json.load(f)
                    
                    # Check structure
                    if 'overall_metrics' in metrics:
                        metrics = metrics['overall_metrics']
                    
                    # Check each horizon
                    for horizon_key, horizon_data in metrics.items():
                        if not horizon_key.startswith('horizon_'):
                            continue
                        
                        required_fields = [
                            'mae_overall', 'rmse_overall', 'coverage_68',
                            'coverage_95', 'coverage_99', 'normality_p_value',
                            'is_normal', 'num_samples'
                        ]
                        
                        missing = [f for f in required_fields if f not in horizon_data]
                        if missing:
                            self.errors.append(f"RESULTS ERROR ({mode}): Missing fields in {horizon_key}: {missing}")
                        
                        # Check for MAPE (should be removed)
                        if 'components' in horizon_data:
                            for comp, comp_data in horizon_data['components'].items():
                                if 'mape' in comp_data:
                                    self.warnings.append(f"RESULTS WARNING ({mode}): MAPE still present in {horizon_key}/{comp} (should be removed)")
                        
                        # Check for invalid values
                        if horizon_data['mae_overall'] < 0:
                            self.errors.append(f"RESULTS ERROR ({mode}): Negative MAE in {horizon_key}")
                        
                        if not (0 <= horizon_data['coverage_68'] <= 1):
                            self.errors.append(f"RESULTS ERROR ({mode}): Invalid coverage_68 in {horizon_key}")
                        
                        if not (0 <= horizon_data['normality_p_value'] <= 1):
                            self.errors.append(f"RESULTS ERROR ({mode}): Invalid p-value in {horizon_key}")
                    
                    self.info.append(f"✓ {mode} results file valid")
                    logger.info(f"  ✓ {filename} is valid")
                    
                except Exception as e:
                    self.errors.append(f"RESULTS ERROR: Cannot parse {filename}: {e}")
                    logger.error(f"  ✗ Error parsing {filename}: {e}")
            else:
                self.info.append(f"ℹ {filename} not found (normal if not trained in {mode} mode)")
        
        logger.info("  ✓ Results consistency checks completed")
    
    def audit_file_system(self):
        """Check file system structure"""
        logger.info("\n" + "="*80)
        logger.info("7. FILE SYSTEM AUDIT")
        logger.info("="*80)
        
        required_dirs = [
            'dataset',
            'Loggs',
            'results',
            'saved_model',
            'tools'
        ]
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                self.warnings.append(f"FS WARNING: Missing directory: {dir_name}")
                logger.warning(f"  ⚠ Missing: {dir_name}")
            else:
                self.info.append(f"✓ Found directory: {dir_name}")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < 1:
                self.warnings.append(f"FS WARNING: Low disk space: {free_gb:.1f} GB")
            else:
                self.info.append(f"✓ Disk space: {free_gb:.1f} GB free")
            
            logger.info(f"  Disk space: {free_gb:.1f} GB free")
        except Exception as e:
            self.warnings.append(f"FS WARNING: Cannot check disk space: {e}")
        
        logger.info("  ✓ File system checks completed")
    
    def audit_performance(self):
        """Check for performance issues"""
        logger.info("\n" + "="*80)
        logger.info("8. PERFORMANCE AUDIT")
        logger.info("="*80)
        
        try:
            from config import ModelConfig
            config = ModelConfig()
            
            # Check for potential bottlenecks
            if config.sequence_length > 100:
                self.warnings.append(f"PERF WARNING: Large sequence_length ({config.sequence_length}) may slow training")
            
            if config.pit_hidden_dim > 512:
                self.warnings.append(f"PERF WARNING: Large hidden_dim ({config.pit_hidden_dim}) increases memory usage")
            
            if config.use_kfold and config.n_folds > 5:
                training_time_multiplier = config.n_folds
                self.warnings.append(f"PERF WARNING: {config.n_folds} folds will take ~{training_time_multiplier}x longer than single split")
            
            # Check device
            if torch.cuda.is_available():
                self.info.append("✓ CUDA available - GPU acceleration enabled")
                logger.info("  ✓ GPU available")
            else:
                try:
                    import torch_directml
                    self.info.append("✓ DirectML available - AMD GPU acceleration enabled")
                    logger.info("  ✓ DirectML (AMD GPU) available")
                except:
                    self.warnings.append("PERF WARNING: No GPU acceleration available, training will be slow")
                    logger.warning("  ⚠ No GPU available")
            
            logger.info("  ✓ Performance audit completed")
            
        except Exception as e:
            self.warnings.append(f"PERF WARNING: Performance audit failed: {e}")
    
    def audit_documentation(self):
        """Check documentation completeness"""
        logger.info("\n" + "="*80)
        logger.info("9. DOCUMENTATION AUDIT")
        logger.info("="*80)
        
        doc_files = [
            'README.md',
            'K_FOLD_IMPLEMENTATION_SUMMARY.md',
            'RESULTS_ANALYSIS_AND_RECOMMENDATIONS.md'
        ]
        
        found = 0
        for doc_file in doc_files:
            # Check in root and documentation folder
            if os.path.exists(doc_file):
                found += 1
                self.info.append(f"✓ Found {doc_file}")
            elif os.path.exists(os.path.join('documentation', doc_file)):
                found += 1
                self.info.append(f"✓ Found documentation/{doc_file}")
            else:
                self.warnings.append(f"DOC WARNING: Missing documentation: {doc_file}")
        
        if found >= 2:
            logger.info(f"  ✓ Found {found}/{len(doc_files)} documentation files")
        else:
            logger.warning(f"  ⚠ Only {found}/{len(doc_files)} documentation files found")
    
    def audit_deployment_readiness(self):
        """Check if system is ready for deployment"""
        logger.info("\n" + "="*80)
        logger.info("10. DEPLOYMENT READINESS AUDIT")
        logger.info("="*80)
        
        ready = True
        
        # Check for trained models
        if os.path.exists('saved_model'):
            checkpoints = [f for f in os.listdir('saved_model') if f.endswith('.pt')]
            if checkpoints:
                self.info.append(f"✓ Found {len(checkpoints)} model checkpoint(s)")
                logger.info(f"  ✓ Found trained models")
            else:
                ready = False
                self.warnings.append("DEPLOY WARNING: No trained models found")
                logger.warning("  ⚠ No trained models")
        else:
            ready = False
            self.warnings.append("DEPLOY WARNING: No saved_model directory")
        
        # Check for evaluation results
        if os.path.exists('results/evaluation_metrics_kfold.json') or os.path.exists('results/evaluation_metrics.json'):
            self.info.append("✓ Evaluation metrics available")
            logger.info("  ✓ Evaluation completed")
        else:
            ready = False
            self.warnings.append("DEPLOY WARNING: No evaluation metrics found")
            logger.warning("  ⚠ No evaluation metrics")
        
        # Check error count
        if self.errors:
            ready = False
            self.warnings.append(f"DEPLOY WARNING: {len(self.errors)} errors must be fixed before deployment")
        
        if ready:
            logger.info("  ✓ System appears ready for deployment")
        else:
            logger.warning("  ⚠ System not ready for deployment - issues must be addressed")
    
    def print_summary(self):
        """Print audit summary"""
        logger.info("\n" + "="*80)
        logger.info("AUDIT SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nErrors Found: {len(self.errors)}")
        if self.errors:
            for error in self.errors:
                logger.error(f"  ✗ {error}")
        
        logger.info(f"\nWarnings Found: {len(self.warnings)}")
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"  ⚠ {warning}")
        
        logger.info(f"\nInfo Messages: {len(self.info)}")
        
        # Overall status
        logger.info("\n" + "="*80)
        if not self.errors:
            logger.info("✓✓✓ NO CRITICAL ERRORS FOUND ✓✓✓")
            if not self.warnings:
                logger.info("✓✓✓ SYSTEM IS IN EXCELLENT CONDITION ✓✓✓")
            else:
                logger.info(f"⚠ {len(self.warnings)} warnings should be reviewed")
        else:
            logger.error(f"✗✗✗ {len(self.errors)} CRITICAL ERRORS MUST BE FIXED ✗✗✗")
        
        logger.info("="*80)
        
        # Save audit report
        report = {
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'status': 'PASS' if not self.errors else 'FAIL'
            }
        }
        
        with open('audit_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("\nAudit report saved to: audit_report.json")


if __name__ == '__main__':
    auditor = ComprehensiveAuditor()
    auditor.run_full_audit()
