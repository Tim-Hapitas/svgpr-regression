"""
Utility functions for data preprocessing and initializing a noise model with a parametric mean function.
Also includes the parametric mean function used for modelling vertical stellar velocity dispersion as in Hapitas et. al.
"""

from .preprocessing import disp_model_tanh, opt_init_disp_params, bin_data_1D

__all__ = [
    'disp_model_tanh',
    'opt_init_disp_params',
    'bin_data_1D',
]