# SVGPR: Stochastic Variational Gaussian Process Regression

Tutorials and python implementation for performing Stochastic Variational Gaussian Process Regression on large datasets.

## About

A python package for scalable Gaussian process regression, allowing for simultaneous inference of both a dataset's latent function and input-dependent noise profile. Originally developed for applications in data-driven Galactic Dynamics but is applicable to any large datset with heteroskedastic noise. This package acts as both a wrapper and extension to the GPyTorch package (https://github.com/cornellius-gp/gpytorch).

## Status

* Under development! Installation instructions, code examples, full tutorials, and final citation information will be added following paper publication. 

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2.2
- GPyTorch ≥ 1.14
- See `pyproject.toml` for complete dependency list

## Installation

### From source

Clone the repository into a directory of your choosing.
```bash
git clone https://github.com/Tim-Hapitas/svgp-regression.git
```
Once complete, cd into the cloned folder and create a clean virtual environment (recommended so that there are no package conflicts with
your other working environments).
```bash
cd svgp-regression
pip -m venv <environment-name>
```
Activate the environment and install with pip.
```bash
venv\Scripts\activate
pip install .
```

## Citation

Based on the method presented in "Gaussian Process Methods for Very Large Astrometric Data Sets (Hapitas et al. 2025) - accepted for publication in ApJ.

If you use this code in your research, please cite:

@article{Hapitas2025,
  author = {Hapitas, Timothy and Widrow, Lawrence M. and Dharmawardena, Thavisha E. and Foreman-Mackey, Daniel},
  title = {Gaussian Process Methods for Very Large Astrometric Datasets},
  journal = {The Astrophysical Journal},
  year = {2025},
  note = {In press}
}



