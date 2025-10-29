# computes dispersion according to a tanh profile
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import torch


def disp_model_tanh(z, mean_params, t_tensor=False):
    bias, out_scale, z_bias, lengthscale = mean_params[0], mean_params[1], mean_params[2], mean_params[3]
    
    if not t_tensor:
        z = np.asarray(z)
        return (bias + out_scale * np.tanh(np.abs(z - z_bias) / lengthscale))
    
    else:
        return (bias + out_scale * torch.tanh(torch.abs(z - z_bias) / lengthscale))


# utility function for binning and computing a statistic on a 1D data set
def bin_data_1D(coords, data_to_bin, statistic, bin_range, bins=10, bin_size=False):
    
    if bin_size:
        num_bins = int(abs(np.min(coords) - np.max(coords)) / bins)
        
    else:
        num_bins = bins
        
    statistic_in_bins, bin_edges, _ = stats.binned_statistic(x=coords, values=data_to_bin, statistic=statistic, bins=num_bins, range=bin_range)
    bin_coords = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_counts = np.histogram(coords, num_bins)[0]
    
    # returns ndarrays
    return statistic_in_bins, bin_coords, bin_counts


# MLE function for performing non-linear regression
def mle_likelihood(params, coords, data, data_err, model):
    arg = ((data - model(coords, params)) / data_err) ** 2 + np.log(2 * np.pi * (data_err ** 2))
    return -0.5 * np.sum(arg)

def opt_init_disp_params(coords, data_to_bin, disp_model, init_guess, statistic='std', bins=10, bin_range=None, bin_size=False, debug=False):
    np.random.seed(25)
    
    # coords, data_to_bin are torch tensors on input
    coords = np.asarray(coords)
    data_to_bin = np.asarray(data_to_bin)
    
    # bin data
    disp_data, bin_coords, bin_counts = bin_data_1D(coords, 
                                                    data_to_bin, 
                                                    statistic,
                                                    bin_range,
                                                    bins, 
                                                    bin_size)
    
    disp_err = (1 / (2 * disp_data)) * (disp_data ** 2) * np.sqrt(2 / (bin_counts - 1))
    
    
    # perform mle fit to binned data using provided disp_model and scipy minimize
    nll = lambda *args: -mle_likelihood(*args)
    start_guess = init_guess + 0.1*np.random.randn(len(init_guess))
    
    fit_result = minimize(nll, start_guess, args=(bin_coords, disp_data, disp_err, disp_model), method='L-BFGS-B')
    opt_init_params = fit_result.x
    print(opt_init_params)
    
    # plot binned data if debug flag is on
    if debug:
        plt.figure()
        plt.plot(bin_coords, disp_data, 'k.', label='binned dispersion')
        plt.plot(coords, disp_model(coords, opt_init_params), label='mle tanh dispersion fit')
        plt.xlabel("Galactic $z$ (pc)")
        plt.ylabel("$\sigma_{v_R}$ (km $s^{-1}$)")
        _ = plt.title("Binned Vertical Velocity Dispersion")
        plt.legend()
        plt.show()
    
    return opt_init_params