from typing import List, Optional
from typing import Any
from typing import Union
import warnings
import gpytorch
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise, FixedGaussianNoise
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
import torch

from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.distributions import Normal
from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.utils.warnings import GPInputWarning



class VectorNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
          self,
          noise: Tensor,
          noise_model: gpytorch.models.GP,
          batch_shape: Optional[torch.Size] = torch.Size(),
          **kwargs: Any,
      ) -> None:

      #First we take the observed noise and assign it to the first noise covariance
      super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

      # These parameters are required by the HeteroskedasticNoise class, along with the input
      # Noise model (i.e. GP)
      # So, for example, when you instantiate it you can pass
      # likelihood = VectorNoiseGaussianLikelihood( ..., noise_constraint = gpytorch.constraints.GreaterThan(1e-7))
      # or something similar
      noise_constraint = kwargs.get("noise_constraint", None)
      noise_indices = kwargs.get("noise_indices", None)
      self.second_noise_covar: HeteroskedasticNoise = None

      # and we can initialise the additional noise now we have everything.
      # Here is where the GP noise model is incoporated to the heteroskedastic noise model
      self.second_noise_covar = HeteroskedasticNoise(noise_model,
                                                     noise_indices = noise_indices,
                                                     noise_constraint = noise_constraint)

    # The forward function needs to be customised because the GP noise model needs input coordinates
    # We might be able to use the standard foward function if we ensure that the input coordinates are propageted through
    def forward(self, function_samples: Tensor, inputs: Tensor,
                *params: Any, **kwargs: Any) -> Normal:
      noise = self._shaped_noise_covar(function_samples.shape, inputs, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
      return base_distributions.Normal(function_samples, noise.sqrt())

    #noise is the total noise
    @property
    def noise(self, inputs) -> Tensor:
        return self.noise_covar.noise + self.second_noise(inputs)

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    #second_noise is the additional heteroskedastic noise model
    @property
    def second_noise(self, input) -> Union[float, Tensor]:
        if self.second_noise_covar is None:
            return 0.0
        else:
            return self.second_noise_covar.forward(input)

    # To return the covariance matrix of the total noise model (noise + second_noise)
    # Puts the two covariances together and makes sure it's shaped correctly for any batching, etc.
    def _shaped_noise_covar(self, base_shape: torch.Size, inputs, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(inputs, *params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a VectorNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res

    # The marginal log likelihood
    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)