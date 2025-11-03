import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy

class LatentGPModel(ApproximateGP):
    """Sparse variational GP model for the latent mean function.
    
    This models the latent GP prior using the standard zero mean prescription and a simple RBF Kernel, 
    which is suitable for many use cases. This inherets from gpytorch.models.AproximateGP which
    provides the necessary functionality for performing stochastic variational inference.
    
    Parameters
    ----------
    inducing_points : torch.Tensor, shape (n_inducing, n_features)
        Initial locations of the inducing points in the input space. These
        are variational parameters that will be optimized during training.
    
    Attributes
    ----------
    mean_module : gpytorch.means.ConstantMean
        Zero mean function for the GP prior.
    covar_module : gpytorch.kernels.ScaleKernel
        Scaled RBF kernel with learnable lengthscale and outputscale parameters.
    
    Notes
    -----
    This implementation follows the sparse variational GP framework described
    in Hensman et al. (2013) and uses the variational strategy from GPyTorch.
    Please consult the appropriate tutorial for constructing custom GP models (UNDER CONSTRUCTION)
    """
    def __init__(self, inducing_points):
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations = True)

        super(LatentGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)