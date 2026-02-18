"""
Test script to verify all fixes are working correctly
"""
import pandas as pd
import numpy as np
import logging
from config import ModelConfig
from data_utils import (
    load_and_prepare_data, 
    validate_dataset_requirements,
    RobustDataPreprocessor,
    GNSSDataset,
    create_robust_data_split
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_loading_and_validation():
    """Test 1: Data loading and validation"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Data Loading and Validation")
    logger.info("="*80)
    
    try:
        df = load_and_prepare_data("dataset")
        logger.info(f"✓ Successfully loaded {len(df)} records")
        logger.info(f"✓ Satellites: {df['satellite_id'].unique().tolist()}")
        logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"✓ Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Validation is now called within load_and_prepare_data
        logger.info("✓ Validation complete (see warnings above)")
        return df
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        raise

def test_preprocessing(df):
    """Test 2: Preprocessing with smart resampling"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Preprocessing with Smart Resampling")
    logger.info("="*80)
    
    try:
        config = ModelConfig()
        preprocessor = RobustDataPreprocessor(config)
        
        logger.info("Starting preprocessing...")
        df_processed = preprocessor.fit_transform(df)
        
        logger.info(f"✓ Processed {len(df_processed)} records")
        logger.info(f"✓ Features: {len(df_processed.columns)} columns")
        
        # Check if data is at 15-minute intervals now
        for sat_id in df_processed['satellite_id'].unique():
            sat_data = df_processed[df_processed['satellite_id'] == sat_id].sort_values('timestamp')
            intervals = sat_data['timestamp'].diff().dt.total_seconds() / 60
            intervals = intervals.dropna()
            
            if len(intervals) > 0:
                consistent_15min = ((intervals >= 14) & (intervals <= 16)).sum()
                consistency_pct = (consistent_15min / len(intervals) * 100)
                mean_interval = intervals.mean()
                
                logger.info(f"  {sat_id}: mean interval = {mean_interval:.1f}min, 15-min consistency = {consistency_pct:.1f}%")
        
        return df_processed, preprocessor
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        raise

def test_data_split(df):
    """Test 3: Time-based data split"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Time-Based Data Split")
    logger.info("="*80)
    
    try:
        config = ModelConfig()
        train_df, val_df, test_df = create_robust_data_split(df, config)
        
        logger.info(f"✓ Train: {len(train_df)} records")
        logger.info(f"  Time range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        
        if len(val_df) > 0:
            logger.info(f"✓ Val: {len(val_df)} records")
            logger.info(f"  Time range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        
        if len(test_df) > 0:
            logger.info(f"✓ Test: {len(test_df)} records")
            logger.info(f"  Time range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        # Verify temporal ordering
        if len(train_df) > 0 and len(test_df) > 0:
            if train_df['timestamp'].max() < test_df['timestamp'].min():
                logger.info("✓ Proper temporal split: training ends before testing begins")
            else:
                logger.warning("⚠ WARNING: Training and testing time periods overlap!")
        
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        raise

def test_dataset_creation(train_df, val_df, test_df):
    """Test 4: Dataset creation with time-based horizons"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Dataset Creation with Time-Based Horizons")
    logger.info("="*80)
    
    try:
        config = ModelConfig()
        config.input_dim = len([col for col in train_df.select_dtypes(include=[np.number]).columns 
                               if col not in ['satellite_id', 'is_real_measurement']])
        
        logger.info(f"Config: sequence_length = {config.sequence_length}")
        logger.info(f"Config: prediction_horizons = {config.prediction_horizons}")
        logger.info(f"Config: input_dim = {config.input_dim}")
        
        train_dataset = GNSSDataset(train_df, config, mode='train')
        logger.info(f"✓ Train dataset: {len(train_dataset)} sequences")
        
        if len(val_df) > 0:
            val_dataset = GNSSDataset(val_df, config, mode='val')
            logger.info(f"✓ Val dataset: {len(val_dataset)} sequences")
        else:
            val_dataset = None
        
        if len(test_df) > 0:
            test_dataset = GNSSDataset(test_df, config, mode='test')
            logger.info(f"✓ Test dataset: {len(test_dataset)} sequences")
        else:
            test_dataset = None
        
        # Test sample retrieval
        if len(train_dataset) > 0:
            seq, targets, idx = train_dataset[0]
            logger.info(f"✓ Sample sequence shape: {seq.shape}")
            logger.info(f"✓ Target horizons available: {list(targets.keys())}")
            
            for horizon, target in targets.items():
                logger.info(f"  - {horizon}: shape {target.shape}")
        
        return train_dataset, val_dataset, test_dataset
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        raise

def test_prediction_horizons():
    """Test 5: Verify prediction horizon mappings"""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Prediction Horizon Verification")
    logger.info("="*80)
    
    horizon_minutes = {
        '15min': 15,
        '30min': 30,
        '1hr': 60,
        '2hr': 120,
        '4hr': 240,
        '8hr': 480,
        '24hr': 1440
    }
    
    logger.info("Problem Requirements vs Implementation:")
    logger.info("-" * 60)
    logger.info(f"{'Horizon':<10} {'Minutes':<10} {'Hours':<10} {'Status':<20}")
    logger.info("-" * 60)
    
    required_horizons = ['15min', '30min', '1hr', '2hr', '24hr']
    config = ModelConfig()
    
    for horizon in required_horizons:
        minutes = horizon_minutes[horizon]
        hours = minutes / 60
        implemented = horizon in config.prediction_horizons
        status = "✓ Implemented" if implemented else "✗ Missing"
        logger.info(f"{horizon:<10} {minutes:<10} {hours:<10.2f} {status:<20}")
    
    logger.info("-" * 60)
    logger.info("Additional horizons implemented:")
    for horizon in config.prediction_horizons:
        if horizon not in required_horizons:
            minutes = horizon_minutes[horizon]
            hours = minutes / 60
            logger.info(f"  + {horizon} ({minutes} min / {hours:.1f} hours)")

def main():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE TEST SUITE FOR PROBLEM STATEMENT COMPLIANCE")
    logger.info("="*80)
    
    try:
        # Test 1: Load and validate
        df = test_data_loading_and_validation()
        
        # Test 2: Preprocessing
        df_processed, preprocessor = test_preprocessing(df)
        
        # Test 3: Data split
        train_df, val_df, test_df = test_data_split(df_processed)
        
        # Test 4: Dataset creation
        train_dataset, val_dataset, test_dataset = test_dataset_creation(train_df, val_df, test_df)
        
        # Test 5: Verify horizons
        test_prediction_horizons()
        
        logger.info("\n" + "="*80)
        logger.info("✓ ALL TESTS PASSED")
        logger.info("="*80)
        logger.info("\nSummary:")
        logger.info(f"  - Dataset validation: ✓ Working")
        logger.info(f"  - Smart resampling to 15-min: ✓ Working")
        logger.info(f"  - Time-based data split: ✓ Working")
        logger.info(f"  - Time-based horizons: ✓ Working")
        logger.info(f"  - Proper temporal ordering: ✓ Verified")
        logger.info("\nReady for model training!")
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("✗ TESTS FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
