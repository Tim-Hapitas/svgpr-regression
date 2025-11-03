from typing import List, Union
import numpy as np
import torch
import gpytorch
from gpytorch.means.mean import Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy


class VelocityDispersionMean(Mean):
    """Parametric mean function using a tanh-based functional form.
    
    Implements the tanh vertical velocity dispersion mean function discussed in Hapitas et al.
    This is included for those wishing to reproduce the paper results and is meant to be used
    in conjuction with DispGPModel.
    """
    def __init__(self, batch_shape=torch.Size(), 
                 bias_constraint=None, 
                 outscale_constraint=None,
                 zbias_constraint=None,
                 hscale_constraint=None,
                 learn_mean_params=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # register all parameters
        self.register_parameter(name="raw_bias", parameter=torch.nn.Parameter(torch.ones(batch_shape),
                                requires_grad=learn_mean_params))
        self.register_parameter(name="raw_outscale", parameter=torch.nn.Parameter(torch.ones(batch_shape),
                                requires_grad=learn_mean_params))
        self.register_parameter(name="raw_zbias", parameter=torch.nn.Parameter(torch.zeros(batch_shape),
                                requires_grad=learn_mean_params))
        self.register_parameter(name="raw_hscale", parameter=torch.nn.Parameter(torch.ones(batch_shape),
                                requires_grad=learn_mean_params))
        
    
    # setting up actual parameters and their setters
    @property
    def bias(self):
        return self.raw_bias
    
    @bias.setter
    def bias(self, value):
        return self._set_bias(value)
        
    @property
    def outscale(self):
        return self.raw_outscale
    
    @outscale.setter
    def outscale(self, value):
        return self._set_outscale(value)
        
    @property
    def zbias(self):
        return self.raw_zbias
    
    @zbias.setter
    def zbias(self, value):
        return self._set_zbias(value)
    
    @property
    def hscale(self):
        return self.raw_hscale
    
    @hscale.setter
    def hscale(self, value):
        return self._set_hscale(value)
    
    
    def _set_bias(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_bias)
               
        self.initialize(raw_bias=value) 
           
    def _set_outscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outscale)
            
        self.initialize(raw_outscale=value)
        
    def _set_zbias(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_zbias)
            
        self.initialize(raw_zbias=value)
        
    def _set_hscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_hscale)
            
        self.initialize(raw_hscale=value)
        
    # WHEN LOADING INTO DISPERSION MODEL, THE MEAN FUNCTION OUTPUT MUST BE SQUARED (this will depend on the astronomical def of dispersion)
    # computes the dispersion mean function
    def forward(self, x):
        arg = torch.abs(x - self.zbias) / self.hscale
        res = (self.bias + self.outscale * torch.tanh(arg)) ** 2
        return res.squeeze(-1)
    


class DispGPModel(ApproximateGP):
    """
    The specific Sparse variational GP model used to model
    input dependent velocity dispersion in Hapitas et al. 2025.
    
    Models the log-variance of heteroscedastic noise as a function of the
    input location. Uses a parametric mean function (VelocityDispersionMean)
    initialized from binned data statistics, combined with an RQ kernel.
    
    Parameters
    ----------
    inducing_points : torch.Tensor, shape (n_inducing, n_features)
        Initial locations of the inducing points in the input space.
    
    Attributes
    ----------
    mean_module : VelocityDispersionMean
        Parametric mean function with optinally learnable parameters.
    covar_module : gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel)
        Scaled RQ kernel with trainable lengthscale, relative lengthscale weighting, outputscale.
    
    Notes
    -----
    This model predicts the raw (unnormalized) noise variance. A softplus
    transformation is applued in the inference path to ensure
    positivity. This is specifically designed for the velocity
    dispersion modeling approach in Hapitas et al. (2025). For designing custom
    GP models please consult the appropriate tutorial (UNDER CONSTRUCTION)
    """
    def __init__(self, inducing_points):
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations = True)

        super(DispGPModel, self).__init__(variational_strategy)

        self.mean_module = VelocityDispersionMean(learn_mean_params=False)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def set_mean_parameters(self, params: Union[List, np.ndarray, torch.Tensor]):
        self.mean_module.bias = params[0]
        self.mean_module.outscale = params[1]
        self.mean_module.zbias = params[2]
        self.mean_module.hscale = params[3]

    def set_noise_lengthscale(self, lengthscale: float):
        self.covar_module.base_kernel.lengthscale = lengthscale