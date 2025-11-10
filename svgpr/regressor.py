import torch
import gpytorch
import tqdm

from svgpr.likelihoods.vector_noise import VectorNoiseGaussianLikelihood
from svgpr.models.latent import LatentGPModel
from svgpr.models.noise import DispGPModel
from svgpr.config.settings import DeviceConfig, OptimizerConfig, NoiseMeanInitConfig
from svgpr.utils.preprocessing import opt_init_disp_params

from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Optional, Union

torch.set_default_dtype(torch.float64)

class SVGPRegressor:
    """
    Implements Stochastic Variational Gaussian Process Regression, performing latent and input-dependent
    noise function inference on a provided dataset

    Given a 1D dataset (including measurement errors) and GP models for the latent function and heteroskedastic noise processes,
    provides an object that performs SVGPR using by optimizing the evidence-lower bound for the data. 

    Parameters
    ----------
    data_x : torch.Tensor (1D)
        Training inputs, (independent variable of the data)
    data_y : torch.Tensor (1D)
        Training targets, (dependet variable of the data)
    latent_model : svgpr.models.LatentGPModel
        The variational sparse GP model for the latent function.
    noise_model : gpytorhc.models.ApproximateGP subclass
        The variational sparse GP model for the input-dependent noise function.
    y_err : torch.Tensor, optional
        The observational uncertainty of the input dataset. In the current version, if this is not provided, SVGPRegressor instances generate 
        "fake" measurement uncertainties of 0.1 * data_y. This will be changed to a required parameter in the next update.
    device_config : DeviceConfig (see config.settings for class definiton)
        Configuration dataclass specifying whether to perform inference explictly on the CPU or the GPU. If not provided, an SVGPRegressor instance attempts GPU inference 
        if CUDA and a compatible graphics card is detected. If not, it falls back to the CPU.
    optimizer_config : OptimizerConfig (see config.settings for class definiton)
        Configuration dataclass allowing customization of both the parameters and specific gpytorch optimizer used to perform stochastic gradient descent. 
        If not provided, SVGPR employs the optimization setup used in Hapitas et al. 2025.
    noise_init_config: NoiseMeanInitConfig (see config.settings for class defintion)
        Configuration dataclass that specifies whether or not to fit a parametric function to the heteroskedastic noise prior to performing SVGPR.
        If not provided, SVGPRegressor assumes nothing about the prior on the noise function apart from what is specified in the noise_model GP.
        This technique was employed in Hapitas et al. to capture the prior informaton on the specific noise function they were trying to fit for, and is not required unless your
        dataset would benefit from a similar treatment. For this use-case, this dataclass explictly tells SVGPRegressor instances how to perform this pre-fitting to the data. 
        Please consult the corresponding tutorial for how this works (UNDER CONSTRUCTION).
        ** Note, if you are using DispGPModel for re-producing the results in Hapitas et al. 2025, this class requires the Hapitas et al. 2025 configurations for this parameter,
           which can be set by passing in noise_init_model = NoiseMeanInitConfig(enabled = True, settings = NoiseConfig())
    """
    def __init__(self, 
                 data_x: torch.Tensor, 
                 data_y: torch.Tensor, 
                 latent_model: gpytorch.models.ApproximateGP,
                 noise_model: gpytorch.models.ApproximateGP,
                 y_err: Optional[torch.Tensor] = None,
                 device_config: Optional[DeviceConfig] = None, 
                 optimizer_config: Optional[OptimizerConfig] = None,
                 noise_init_config: Optional[NoiseMeanInitConfig] = None,
        ):

        # validate config typing
        if device_config is not None:
            if not isinstance(device_config, DeviceConfig):
                raise ValueError(f"device_config must be of type DeviceConfig, instead got {type(device_config)}")

        if optimizer_config is not None:
            if not isinstance(optimizer_config, OptimizerConfig):
                raise ValueError(f"optimizer_config must be of type OptimizerConfig, instead got {type(optimizer_config)}")
            
        if noise_init_config is not None:
            if not isinstance(noise_init_config, NoiseMeanInitConfig):
                raise ValueError(f"noise_init_config must be of type NoiseMeanInitConfig, instead got {type(noise_init_config)}")

        # validate GP model typing
        if not isinstance(latent_model, gpytorch.models.ApproximateGP):
            raise ValueError(f"latent_model must be a subclass of gpytorch.models.ApproximateGP")
        
        if not isinstance(noise_model, gpytorch.models.ApproximateGP):
            raise ValueError(f"noise_model must be a subclass of gpytorch.models.ApproximateGP")

        if isinstance(noise_model, DispGPModel) and (noise_init_config == NoiseMeanInitConfig()):
            raise ValueError(
                "DispGPModel requires noise_init_config to match the settings used in Hapitas et al. 2025. "
                "Please pass in a noise_init_config with enabled set to True and noise_config set to NoiseConfig()"
            )
        
        elif isinstance(noise_model, DispGPModel) and (noise_init_config == None):
            raise ValueError(
                "DispGPModel requires noise_init_config to match the settings used in Hapitas et al. 2025. "
                "Please pass in a noise_init_config with enabled set to True and noise_config set to NoiseConfig()"
            )

        # Validate data input
        if not isinstance(data_x, torch.Tensor) or not isinstance(data_y, torch.Tensor):
            raise ValueError(f"data_x and data_y must be of type torch.Tensor, instead got {type(data_x)} and {type(data_y)}")

        if data_x.ndim != 1 or data_y.ndim != 1:
            raise ValueError(
                f"data_x and data_y must be 1D tensors, instead got {data_x.ndim} and {data_y.ndim}"
            )
        
        if len(data_x) != len(data_y):
            raise ValueError(f"data_x and data_y must be the same length, instead got {len(data_x)} and {len(data_y)}")

        if y_err is not None:
            if not isinstance(y_err, torch.Tensor):
                raise ValueError(f"y_err must be of type torch.Tensor, instead got {type(y_err)}")
            
            if y_err.ndim != 1:
                raise ValueError(f"y_err must be a 1D tensor, instead got {y_err.ndim}")
            
            if len(y_err) != len(data_y):
                raise ValueError(f"length of y_err must match that of data_y, instead got {len(y_err)}")


        self.device_config = device_config or DeviceConfig()
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.noise_init_config = noise_init_config or NoiseMeanInitConfig()

        data = self._prepare_data(data_x, data_y, y_err=y_err)

        self.data_x = data['x']
        self.data_y = data['y']
        self.data_y_err = data['y_err']

        self.data_y_mean = data['y_mean']
        self.data_y_std = data['y_std']

        self.latent_model = latent_model
        self.noise_model = noise_model

        self.likelihood = VectorNoiseGaussianLikelihood(
            noise = data['y_err'],
            noise_model = self.noise_model,
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)
        )

    def _prepare_data(self, x: torch.Tensor, y: torch.Tensor, y_err: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_mean = torch.mean(y)
        y_std = torch.std(y)

        y_norm = (y - y_mean) / y_std

        if y_err is not None:
            y_err_norm = y_err / y_std

        else:
            y_err_norm = 0.1 * torch.abs(y)

        return {
            'x': x.contiguous(),
            'y': y_norm.contiguous(),
            'y_err': y_err_norm.contiguous(),
            'y_mean': y_mean,
            'y_std': y_std
        }
    
    def fit(self): 
        """
        Fits the constructed SVGP model to the training data

        Performs stochastic variational GPR by optimizing the evidence lower bound (ELBO)
        using mini-batch gradient descent. The method trains both the latent GP and the noise GP simultaneously.
        
        The fitting process consists of three steps:
        1. Initialize the noise model internal parameters (if noise_init_config is provided and enables)
        2. Prepare data and GP models for the training process (deivce loading and building the optimization procedure)
        3. Run the main optimization loop for the specified number of training epochs

        Returns
        -------
        None
            The method modifies the GP models in place. Trained hyperparameters are stored internally in latent_model and noise_model.

        Notes
        -----
        Training progress and the value of the loss function (negative value of the ELBO) are reported via tqdm

        """
        self._initialize_noise_parameters()
        self._prepare_training()
        self._run_training_loop()


    def predict(self, test_x: torch.tensor, get_uncerts: bool = True) -> Union[torch.tensor, List[torch.tensor]]:
        """
        Compute predictions for the latent and noise function on provided test inputs using the fitted SVGP model

        Returns the posterior mean predictions for the latent and noise functions, along with uncertainty estimates for just the latent process.
        (The computation of noise uncertainty as discussed in Hapitas et al. 2025 is done using a separate manual procedure. Please consult the 
        corresponding tutorial (UNDER CONSTRUCTION))

        Parameters
        ----------
        test_x : torch.Tensor, shape (n_test,)
            Test coordinates at which to predict the latnet and noise functions
        get_uncerts : bool
            Whether or not to return uncertainties (not yet implemented, for now this will always return uncertainty information
            for the latent function)

        Returns
        -------
        mean : torch.Tensor, length (n_test)
            Posterior mean predictions of the latent function at the test coordinates.
        latent_lower : torch.Tensor, length (n_test)
            Lower bound of the 95% confidence region for the latent function prediction
        latent_upper : torch.Tensor, length (n_test)
            Upper bound of the 95% confidence region for the latent function prediction
        noise_preds : torch.Tensor, length (n_test)
            Predicted noise values at the test coordinates

        Notes
        -----
        This method uses gpytorch's fast predictive variance settings for efficiency
        """
        sp = torch.nn.Softplus()

        self.latent_model.eval()
        self.likelihood.eval()
        self.likelihood.second_noise_covar.noise_model.eval()

        test_x = test_x.to(self.device_config.get_device())

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_preds = self.latent_model(test_x)
            mean = latent_preds.mean
            latent_lower, latent_upper = latent_preds.confidence_region()

            raw_noise_preds = self.likelihood.second_noise_covar.noise_model(test_x)
            noise_preds = torch.sqrt(sp(raw_noise_preds.mean))

            del latent_preds
            del raw_noise_preds

            mean = (mean * self.data_y_std) + self.data_y_mean
            latent_lower = (latent_lower * self.data_y_std) + self.data_y_mean
            latent_upper = (latent_upper * self.data_y_std) + self.data_y_mean
            noise_preds = noise_preds * self.data_y_std

            return mean, latent_lower, latent_upper, noise_preds
        
    def _initialize_noise_parameters(self):

        # if the user is not fitting a parametric mean function to the noise model, we want this
        # function to do nothing
        if not self.noise_init_config.enabled:
            return
        
        if self.noise_init_config.settings is None:
            raise ValueError(
                "NoiseMeanInitConfig.enabled is True but settings is None. "
                "Please provide a NoiseConfig instance for the settings parameter."
            )
        
        opt_disp_params = opt_init_disp_params(
            coords = self.data_x,
            data_to_bin = self.data_y,
            disp_model = self.noise_init_config.settings.mean_function,
            init_guess = self.noise_init_config.settings.init_guess,
            bins = self.noise_init_config.settings.bins,
            bin_range = self.noise_init_config.settings.bin_range,
            bin_size = self.noise_init_config.settings.bin_size, 
            debug = self.noise_init_config.settings.debug
        )

        # hard coding here needs to be modified in next stage of refactoring
        self.noise_model.set_mean_parameters(opt_disp_params)
        self.noise_model.set_noise_lengthscale(0.1)

    def _prepare_training(self):

        self.data_x = self.data_x.to(self.device_config.get_device())
        self.data_y = self.data_y.to(self.device_config.get_device())
        self.data_y_err = self.data_y_err.to(self.device_config.get_device())
        self.latent_model = self.latent_model.to(self.device_config.get_device())
        self.likelihood = self.likelihood.to(self.device_config.get_device())

        batch_size = int(self.data_x.size(0) / self.optimizer_config.batch_ratio)

        training_set = TensorDataset(self.data_x, self.data_y)
        self.train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

        self.latent_model.train()
        self.likelihood.train()
        self.likelihood.second_noise_covar.noise_model.train()

        optimizer = self.optimizer_config.optimizer_class([
            {'params': self.latent_model.parameters(), 'lr': self.optimizer_config.latent_lr},
            {'params': self.likelihood.parameters(), 'lr': self.optimizer_config.noise_lr},
        ])

        mll = self.optimizer_config.mll_class(self.likelihood, self.latent_model, num_data=self.data_y.size(0))

        self.optimizer = optimizer
        self.mll = mll

    def _run_training_loop(self):

        t = tqdm.trange(self.optimizer_config.n_epochs, desc='loss', leave=True)
        for k in t:
            for x_batch, y_batch in self.train_loader:

                self.optimizer.zero_grad()
                output = self.latent_model(x_batch)
                loss = -self.mll(output, y_batch, inputs=x_batch)
                loss.backward()
                self.optimizer.step()
                t.set_postfix({'loss value': loss.item()})

# -------------------------------------------------------------------------------------------------    




    
