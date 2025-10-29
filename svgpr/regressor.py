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
    def __init__(self, 
                 data_x: torch.Tensor, 
                 data_y: torch.Tensor, 
                 latent_model: LatentGPModel, 
                 noise_model: DispGPModel,
                 y_err: Optional[torch.Tensor] = None,
                 device_config: Optional[DeviceConfig] = None, 
                 optimizer_config: Optional[OptimizerConfig] = None,
                 noise_init_config: Optional[NoiseMeanInitConfig] = None,
        ):

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

        self._initialize_noise_parameters()
        self._prepare_training()
        self._run_training_loop()


    def predict(self, test_x: torch.tensor, get_uncerts: bool = True) -> Union[torch.tensor, List[torch.tensor]]:
        # toggle eval mode
        # load data to cpu
        # extract predictions
        # return predictions
        sp = torch.nn.Softplus()

        self.latent_model.eval()
        self.likelihood.eval()
        self.likelihood.second_noise_covar.noise_model.eval()

        test_x = test_x.to(self.device)

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
        
        opt_disp_params = opt_init_disp_params(
            coords = self.data_x,
            data_to_bin = self.data_y,
            disp_model = self.noise_init_config.mean_function,
            init_guess = self.noise_init_config.init_guess,
            bins = self.noise_init_config.bins,
            bin_range = self.noise_init_config.bin_range,
            bin_size = self.noise_init_config.bin_size, 
            debug = self.noise_init_config.debug
        )

        # hard coding here needs to be modified in next stage of refactoring
        self.noise_model.set_mean_params(opt_disp_params)
        self.noise_model.set_noise_lengthscale(0.1)

    def _prepare_training(self):

        self.data_x = self.data_x.to(self.device)
        self.data_y = self.data_y.to(self.device)
        self.data_y_err = self.data_y_err.to(self.device)
        self.latent_model = self.latent_model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

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




    
