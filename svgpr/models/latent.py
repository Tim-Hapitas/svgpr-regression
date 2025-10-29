import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy

# We use variational inference with a simple RBF kernel for now
class LatentGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations = True)

        super(LatentGPModel, self).__init__(variational_strategy)

        #self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)