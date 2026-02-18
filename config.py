"""
Configuration module for GNSS Error Prediction System
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    """Central configuration for the hybrid model with enhanced parameters"""
    sequence_length: int = field(default_factory=lambda: 96)  # 24 hours at 15-min intervals (96*15min=24h)
    
    # Prediction horizons: Now using TIME-BASED names matching problem requirements
    # Problem asks for: 15min, 30min, 1hr, 2hr, up to 24hr predictions
    # These keys will be used in the model, not index offsets
    prediction_horizons: List[str] = field(default_factory=lambda: ['15min', '30min', '1hr', '2hr', '4hr', '8hr', '24hr'])
    
    input_dim: int = 0  # To be set dynamically
    target_dim: int = 5
    
    # Data resampling configuration
    enable_resampling: bool = True  # Set to False to use original irregular intervals
    interpolation_method: str = 'advanced'  # 'linear', 'spline', 'advanced' (physics-aware)
    mark_synthetic_points: bool = True  # Add flag to identify interpolated points
    max_interpolation_gap_hours: float = 3.0  # Maximum gap to interpolate across
    
    pit_hidden_dim: int = 256  # Reduced for memory efficiency
    pit_num_heads: int = 8  # Reduced from 10
    pit_num_layers: int = 6  # Reduced from 10 to save memory
    pit_dropout: float = 0.2  # Stronger dropout for better generalization
    pit_physics_weight: float = 0.6  # Stronger physics constraints
    
    ndm_hidden_dim: int = 192  # Reduced for memory efficiency
    ndm_diffusion_steps: int = 100  # Reduced from 200 to save memory
    ndm_beta_start: float = 0.0001
    ndm_beta_end: float = 0.01  # Reduced for more stable diffusion
    
    batch_size: int = 24  # Reduced to fit in GPU memory
    learning_rate: float = 2e-4  # Even lower for fine-grained optimization
    weight_decay: float = 1e-5  # Minimal regularization for flexibility
    epochs: int = 30  # Extended training for convergence
    early_stopping_patience: int = 50  # More patience
    warmup_epochs: int = 30  # Longer warmup
    
    outlier_threshold: float = 4.5  # More lenient to preserve natural variation
    min_samples_per_satellite: int = 30  # Increased from 20
    normality_test_alpha: float = 0.05  # Standard Shapiro-Wilk threshold (p > 0.05 to pass)
    
    ensemble_size: int = 5  # Increased from 3
    augmentation_factor: float = 0.25  # Increased augmentation for diversity
    mixup_alpha: float = 0.4  # Increased mixup for smoother predictions
    label_smoothing: float = 0.15  # Increased to encourage Gaussian residuals
    gradient_accumulation_steps: int = 4  # Increased from 2
    
    # Add Gaussian noise to targets during training to encourage normality
    target_noise_std: float = 0.05  # Small Gaussian noise added to targets
    
    calibrator_epochs: int = 1000  # Reduced for faster training
    calibrator_lr: float = 0.0003  # Lower LR for finer adjustments
    calibrator_hidden_dim: int = 192  # Reduced for memory efficiency
    calibrator_num_heads: int = 12  # Reduced from 16
    calibrator_memory_size: int = 4096  # Reduced for memory efficiency
    
    # K-Fold Cross-Validation settings
    use_kfold: bool = True  # Enable k-fold cross-validation for better calibrator training
    n_folds: int = 2  # Number of folds (reduced to 2 to accommodate 70/30 test split with sequence_length=96)
    kfold_mode: str = 'time_series'  # 'time_series' or 'standard' - use time-series aware splitting
    
    # ===== NEW: NORMALITY ENFORCEMENT SETTINGS =====
    # These parameters are critical for achieving Shapiro-Wilk normality test compliance
    
    use_residual_standardization: bool = True  # Enable residual standardization layer
    use_quantile_loss: bool = True  # Use quantile regression loss for symmetric errors
    quantile_loss_weight: float = 0.3  # Weight for quantile loss (0.0-1.0)
    
    use_normality_enforcement_loss: bool = True  # Use explicit normality loss
    normality_loss_weight: float = 0.4  # Weight for normality enforcement loss
    
    use_enhanced_bias_correction: bool = True  # Satellite-specific bias corrections
    use_adaptive_physics: bool = True  # Orbit-specific physics constraints
    
    # Uncertainty calibration improvements
    calibrator_recalibration_rounds: int = 3  # Number of recalibration iterations
    use_isotonic_calibration: bool = True  # Apply isotonic regression for probability calibration
    
    # Ensemble diversity settings
    ensemble_dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.15, 0.2, 0.25, 0.3])
    ensemble_seed_offset: int = 42  # Seed offset for ensemble members