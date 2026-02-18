"""
Device utility functions for GPU/CPU setup
"""

import torch
import logging

logger = logging.getLogger(__name__)

def setup_directml_device():
    """Setup DirectML for AMD GPU support"""
    device = None
    device_name = "cpu"
    
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = "CUDA GPU"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            device_name = "Apple MPS"
            logger.info("Using Apple MPS")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                device_name = "AMD GPU (DirectML)"
                logger.info("Successfully initialized DirectML for AMD GPU")
            except ImportError:
                logger.warning("torch-directml not installed. Falling back to CPU")
                device = torch.device('cpu')
                device_name = "CPU"
    except Exception as e:
        logger.warning(f"Device setup error: {e}. Using CPU")
        device = torch.device('cpu')
        device_name = "CPU"
    
    logger.info(f"Using device: {device_name}")
    return device
