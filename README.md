# GNSS Error Prediction System

## Overview

This project implements a hybrid Physics-Informed Neural Network (PINN) system for predicting GNSS (Global Navigation Satellite System) errors with uncertainty quantification. The system combines advanced machine learning techniques with domain knowledge from satellite orbital mechanics to provide accurate and reliable error predictions.

## Key Features

- **Hybrid Architecture**: Combines Physics-Informed Transformer, Neural Diffusion Model, and Attention Calibrator
- **Multi-GPU Support**: AMD DirectML, NVIDIA CUDA, and Apple MPS acceleration
- **Multi-horizon Prediction**: Predicts errors from 15 minutes to 24 hours ahead
- **Uncertainty Quantification**: Provides confidence intervals and uncertainty estimates
- **Robust Preprocessing**: Handles outliers, missing data, and temporal gaps
- **Physics Integration**: Incorporates orbital dynamics, clock stability, and atmospheric effects

## Project Structure

```
d:\SIH_FINAL_MODEL\dt\
├── config.py              # Configuration settings and hyperparameters
├── device_utils.py        # GPU/CPU device setup and management
├── data_utils.py          # Data loading, preprocessing, and dataset utilities
├── evaluation_utils.py    # Model evaluation and visualization tools
├── model.py               # Main AI model components
├── main.py                # Training and execution pipeline
├── README.md              # This documentation file
└── requirements.txt       # Python dependencies
```

##  Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA 11.8+ (for NVIDIA GPUs) or DirectML (for AMD GPUs)

### Step-by-step Installation

1. **Clone or download the project files**
   ```bash
   cd d:\SIH_FINAL_MODEL\dt
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv gnss_env
   gnss_env\Scripts\activate  # Windows
   # source gnss_env/bin/activate  # Linux/Mac
   ```

3. **Install core dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install numpy pandas scipy scikit-learn matplotlib
   pip install openpyxl xlrd  # For Excel file support
   ```

4. **Install DirectML for AMD GPUs (optional)**
   ```bash
   pip install torch-directml
   ```

## Data Format

### Input Data Structure

The system expects GNSS data in CSV or Excel format with the following columns:

| Column Name | Description | Units | Example |
|-------------|-------------|-------|---------|
| `timestamp` or `utc_time` | Time stamp | YYYY-MM-DD HH:MM:SS | 2023-01-01 00:00:00 |
| `satclockerror (m)` | Satellite clock error | meters | 1.23e-6 |
| `x_error (m)` | X-component ephemeris error | meters | 0.45 |
| `y_error (m)` | Y-component ephemeris error | meters | -0.32 |
| `z_error (m)` | Z-component ephemeris error | meters | 0.67 |

### File Naming Convention

Files should be named to indicate satellite type:
- `*GEO*` - Geostationary satellites
- `*MEO*` - Medium Earth Orbit satellites  
- `*GSO*` - Geosynchronous satellites

### Data Requirements

- **Time span**: 7 days of continuous data recommended
- **Sampling rate**: 15-minute intervals
- **Minimum samples**: 20 data points per satellite
- **File formats**: CSV (.csv) or Excel (.xlsx)


### Basic Training

```bash
python main.py --data-folder "path/to/your/data"
```

### Advanced Options

```bash
# Training with custom data folder
python main.py --data-folder "d:\your_data\gnss_files"

# Prediction only (using pre-trained model)
python main.py --predict-only "gnss_hybrid_all_checkpoint.pt" --new-data "new_data.csv"

# Custom configuration
python main.py --data-folder "data" --config-file "custom_config.json"
```

### Python API Usage

```python
from config import ModelConfig
from main import main, load_trained_model
from model02 import HybridGNSSModel

# Custom training
config = ModelConfig()
config.epochs = 100
config.learning_rate = 0.001

model, predictions, metrics = main("path/to/data", config)

# Load pre-trained model
model = load_trained_model("checkpoint.pt")
predictions = model.predict(test_data, return_uncertainty=True)
```

## Configuration

### ModelConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 6 | Input sequence length |
| `prediction_horizons` | [1,2,4,8,12,24,48,96] | Prediction steps ahead |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.01 | Learning rate |
| `epochs` | 50 | Training epochs |
| `pit_hidden_dim` | 16 | Transformer hidden dimension |
| `ndm_diffusion_steps` | 50 | Diffusion model steps |

### Customizing Configuration

```python
from config import ModelConfig

config = ModelConfig()
config.epochs = 100
config.learning_rate = 0.001
config.batch_size = 64
config.prediction_horizons = [1, 6, 12, 24, 48]
```

## Model Architecture

### 1. Physics-Informed Transformer (PIT)

**Purpose**: Primary prediction engine incorporating physical constraints

**Components**:
- Multi-head attention mechanism
- Orbital dynamics layer (Kepler's laws, perturbations)
- Clock dynamics layer (Allan variance, drift modeling)
- Atmospheric effects layer (ionospheric and tropospheric delays)

**Physics Integration**:
- Orbital period constraints based on satellite type
- Gravitational effects modeling
- Solar radiation pressure proxy
- Clock stability enforcement

### 2. Neural Diffusion Model (NDM)

**Purpose**: Uncertainty quantification and residual error modeling

**Process**:
1. Forward diffusion: Gradually adds noise to residuals
2. Reverse diffusion: Learns to denoise and generate uncertainty samples
3. Sampling: Produces multiple realizations for uncertainty estimation

**Benefits**:
- Captures complex error distributions
- Provides calibrated uncertainty estimates
- Handles non-Gaussian error patterns

### 3. Attention Calibrator

**Purpose**: Post-hoc calibration of predictions and uncertainties

**Method**:
- Memory-based attention mechanism
- Learns correction patterns from validation data
- Improves prediction accuracy and uncertainty calibration

## Evaluation Metrics

### Accuracy Metrics

- **MAE (Mean Absolute Error)**: Average magnitude of prediction errors
- **RMSE (Root Mean Square Error)**: Penalizes large errors more heavily
- **MAPE (Mean Absolute Percentage Error)**: Relative error measure

### Uncertainty Metrics

- **CRPS (Continuous Ranked Probability Score)**: Evaluates probabilistic forecasts
- **Coverage**: Percentage of true values within prediction intervals
- **Calibration**: How well uncertainty estimates match actual errors

### Statistical Tests

- **Shapiro-Wilk Test**: Tests normality of residuals
- **Correlation Analysis**: Examines error dependencies between components

## Output Files

### Generated Files

| Filename | Description |
|----------|-------------|
| `evaluation_metrics.json` | Comprehensive evaluation results |
| `predictions_horizon_X.csv` | Predictions for each horizon |
| `prediction_plots_horizon_X.png` | Visualization of predictions |
| `residual_plots_horizon_X.png` | Residual analysis plots |
| `gnss_hybrid_all_checkpoint.pt` | Trained model checkpoint |
| `gnss_training.log` | Detailed training logs |

### Prediction Output Format

```csv
original_index,clock_error_pred,ephemeris_error_x_pred,ephemeris_error_y_pred,ephemeris_error_z_pred,orbit_error_3d_pred,clock_error_unc,ephemeris_error_x_unc,ephemeris_error_y_unc,ephemeris_error_z_unc,orbit_error_3d_unc
0,1.23e-06,0.45,-0.32,0.67,0.89,2.1e-07,0.08,0.06,0.09,0.12
```

## Troubleshooting

### Common Issues

**1. ImportError: attempted relative import with no known parent package**
```
Solution: Ensure you're running from the dt directory and using absolute imports
```

**2. CUDA/DirectML not detected**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# For AMD GPUs, install DirectML
pip install torch-directml
```

**3. Memory errors during training**
```python
# Reduce batch size in config
config.batch_size = 16
config.ndm_diffusion_steps = 25
```

**4. Poor prediction accuracy**
```
- Check data quality and completeness
- Ensure proper timestamp formatting
- Verify satellite type identification in filenames
- Increase training epochs or adjust learning rate
```

### Performance Optimization

**For Large Datasets**:
- Increase `batch_size` to 64 or 128
- Use multiple GPU if available
- Reduce `sequence_length` for faster training

**For Small Datasets**:
- Decrease `min_samples_per_satellite` to 10
- Increase `augmentation_factor` to 0.2
- Use fewer prediction horizons

## Technical Details

### Preprocessing Pipeline

1. **Data Loading**: Multi-format support (CSV, Excel)
2. **Column Mapping**: Automatic standardization of column names
3. **Timestamp Handling**: Multiple format detection and conversion
4. **Outlier Detection**: Modified Z-score and IQR methods
5. **Missing Value Imputation**: Spline interpolation and forward/backward fill
6. **Normalization**: Per-satellite robust scaling
7. **Feature Engineering**: Physics-based and temporal features
8. **Data Augmentation**: Gaussian noise and time warping

### Training Process

**Stage 1: Physics-Informed Transformer Training**
- MSE loss on multi-horizon predictions
- Physics-based regularization terms
- Early stopping with validation monitoring

**Stage 2: Neural Diffusion Model Training**
- Trains on residuals from Stage 1
- Denoising objective with diffusion process
- Context-aware conditioning

**Stage 3: Attention Calibrator Fitting**
- Memory-based attention mechanism
- Learns correction patterns
- Improves calibration of uncertainties

### Prediction Horizons

| Horizon | Time Ahead | Use Case |
|---------|------------|----------|
| 1 | 15 minutes | Real-time corrections |
| 2-4 | 30-60 minutes | Short-term planning |
| 8-12 | 2-3 hours | Medium-term operations |
| 24-48 | 6-12 hours | Daily planning |
| 96 | 24 hours | Long-term forecasting |

##  Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

### Code Style

- Follow PEP 8 conventions
- Add docstrings for all functions
- Include type hints where appropriate
- Add logging for important operations


## Support

For questions or issues:

1. Check this README and troubleshooting section
2. Review the training logs in `gnss_training.log`
3. Examine the evaluation metrics in `evaluation_metrics.json`
4. Create detailed issue reports with:
   - Error messages
   - Data characteristics
   - System specifications
   - Steps to reproduce

## Version History

- **v1.0**: Initial modular implementation
- **v1.1**: Fixed import issues and improved documentation
- **v1.2**: Enhanced error handling and logging

## Future Enhancements

- [ ] Real-time streaming prediction capability
- [ ] Integration with live GNSS data feeds
- [ ] Web-based visualization dashboard
- [ ] Model ensemble techniques
- [ ] Automated hyperparameter optimization
- [ ] Support for additional satellite constellations

---

*This system is designed to advance GNSS error prediction capabilities through the integration of physics-informed machine learning and uncertainty quantification techniques.*
