"""
Custom Gaussian likelihood function that allows for GP inference of input-dependent noise
"""

from .vector_noise import VectorNoiseGaussianLikelihood

__all__ = ['VectorNoiseGaussianLikelihood']