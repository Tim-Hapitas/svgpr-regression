"""
GP model definitions for the latent function and noise used in Hapitas et. al.
"""

from .latent import LatentGPModel
from .noise import DispGPModel, VelocityDispersionMean

__all__ = ['LatentGPModel', 'DispGPModel', 'VelocityDispersionMean']

