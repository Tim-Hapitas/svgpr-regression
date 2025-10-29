"""
SVGPR: Stochastic Variational Gaussian Process Regression with Input-Dependent Noise

A python package for scalable Gaussian process regression that simultaneously models both the latent function and input-dependent noise
governing an observational dataset. 

Based on: Hapitas et al. (2025, ApJ) - "Gaussian Process Methods for Very Large Astrometric Datasets"
"""

__version__ = "0.1.0"

# Main API classes
from .regressor import SVGPRegressor

# Regression configuration classes
from .config.settings import (
    DeviceConfig,
    OptimizerConfig,
    NoiseMeanInitConfig,
)

# Utility functions
from .utils.preprocessing import opt_init_disp_params, disp_model_tanh

__all__ = [
    # Main API classes
    'SVGPRegressor',

    # User config
    'DeviceConfig',
    'OptimizerConfig',
    'NoiseMeanInitConfig',

    # Utils
    'opt_init_disp_params',
    'disp_model_tanh',
]



