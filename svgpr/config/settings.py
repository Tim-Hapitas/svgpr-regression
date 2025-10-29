from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type
from svgpr.utils.preprocessing import disp_model_tanh
import gpytorch
import numpy
import torch

@dataclass
class DeviceConfig:
    """Configuration for cpu/gpu inference"""
    device: Optional[torch.device] = None
    use_cpu: bool = False

    def get_device(self) -> torch.device:
        if self.use_cpu:
            return torch.device('cpu')
        return self.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class OptimizerConfig:
    """User specification for gpytorch optimizer backend and associated parameters"""
    batch_ratio: float = 100
    latent_lr: float = 1.0
    noise_lr: float = 0.1
    n_epochs: float = 300
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.SGD
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    mll_class: Type = gpytorch.mlls.VariationalELBO


@dataclass
class NoiseMeanInitConfig:
    """Configuration for parametric noise mean initialization"""
    enabled: bool = True
    mean_function: Optional[Callable] = disp_model_tanh
    init_guess: Optional[numpy.ndarray] = field(default_factory = lambda: numpy.array([0.1, 1.0, 0.0, 1.0]))
    bins: float = 0.1
    bin_range: tuple = (-2.5, 2.5)
    bin_size: bool = True
    debug: bool = True