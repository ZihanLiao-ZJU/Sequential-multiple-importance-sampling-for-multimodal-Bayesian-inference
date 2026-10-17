# Sequential Multiple Importance Sampling for Multimodal Bayesian Inference

MATLAB implementation of the **Sequential Multiple Importance Sampling (SeMIS)** algorithm for multimodal Bayesian inference, with subset simulation (SuS) included for comparison. This code accompanies the paper:

> **Sequential multiple importance sampling for multimodal Bayesian inference.**
> *Mechanical Systems and Signal Processing*.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Benchmark likelihoods](#benchmark-likelihoods)
- [Outputs](#outputs)
- [Citation](#citation)
- [License](#license)

## Overview

SeMIS is a multiple importance sampling (MIS) based algorithm for Bayesian inference with multimodal and high-dimensional posterior distributions. It builds a sequence of **softly truncated, prior-based proposal distributions** that evolve smoothly from the prior to the posterior, which improves mixing across separated modes while still allowing efficient estimation of the Bayesian evidence.

At each iteration, SeMIS:

1. constructs a proposal distribution by softly truncating the prior using likelihood information;
2. adaptively determines the proposal parameter based on a target acceptance rate;
3. generates samples with parallel MCMC chains started from accepted seeds (elliptical slice sampling by default);
4. accumulates all samples through multiple importance sampling with the balance heuristic.

After the last iteration, the Bayesian evidence is estimated and posterior samples are obtained by resampling. All weights are computed in the log domain for numerical stability.

## Requirements

- MATLAB
- Statistics and Machine Learning Toolbox (`makedist` and probability distribution objects)

## Repository structure

```
.
├── drive_SuS_SeMIS.m                    % driver: runs SeMIS and SuS on a benchmark problem
├── SeMIS_Bay_Nataf.m                    % SeMIS intermediate distributions (soft truncation) in standard normal space
├── SuS_Bay_Nataf.m                      % subset simulation for Bayesian updating (comparison)
├── SeMCS.m                              % sequential Monte Carlo loop shared by SeMIS and SuS
├── MCMC-samplers/
│   ├── ES.m                             % elliptical slice sampling
│   └── aCS.m                            % adaptive conditional sampling
├── Likelihood-functions/                % benchmark likelihoods with reference log-evidence
├── support functions/                   % MIS estimator and posterior resampling (MIS.m), Nataf transformation, weights, ...
├── Operations on Logarithmic values/    % log-domain arithmetic (logsum, logplus, logmean, ...)
├── LICENSE
└── README.md
```

## Quick start

1. Add the repository and all its subfolders to the MATLAB path, for example from the repository root:

   ```matlab
   >> addpath(genpath(pwd))
   ```

2. Run the driver:

   ```matlab
   >> drive_SuS_SeMIS
   ```

The driver defines the benchmark problem (by default `LKF_nD_GaussianLogGamma` with `Ndim = 2`), runs SeMIS and SuS for 100 independent runs, and computes the evidence and efficiency statistics described below. To change the problem, edit the `Tfun = ...` line and `Ndim` at the top of `drive_SuS_SeMIS.m`. The sample size per level is set by `smc_SeMIS.NumSam` (default 1700) and `smc_SuS.NumSam` (default 1000).

## Benchmark likelihoods

| File | Description | Reference log-evidence |
|---|---|---|
| `LKF_2D_Eggbox.m` | 2D eggbox likelihood, uniform prior on [0, 10π]² | 235.856 |
| `LKF_nD_GaussianLogGamma.m` | Gaussian + LogGamma, n-dimensional | −8.19 (n = 2), −20.47 (n = 5), −40.94 (n = 10), −81.887 (n = 20) |
| `LKF_nD_GaussianShells.m` | two Gaussian shells, n-dimensional | −1.75 (n = 2) to −168.16 (n = 70); see the file header |

Each likelihood class provides the prior (`Pdis`), the log-likelihood and the reference value `logz_ref` used by the driver.

## Outputs

During the run, each iteration prints the sample size, the number of chains, the samples per chain and the number of likelihood evaluations. After all runs, the following variables are available in the workspace:

| Variable | Meaning |
|---|---|
| `logz_SeMIS_MIS_bias`, `logz_SeMIS_MIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, MIS estimator |
| `logz_SeMIS_SIS_bias`, `logz_SeMIS_SIS_cv` | relative bias and CoV (%) of the SeMIS log-evidence, SIS estimator |
| `logz_SuS_bias`, `logz_SuS_cv` | relative bias and CoV (%) of the SuS log-evidence |
| `Npos_Ncal_SeMIS_mu`, `Npos_Ncal_SuS_mu` | posterior samples per likelihood evaluation (%) |
| `Ncal_SeMIS_mu`, `Ncal_SuS_mu` | average number of likelihood evaluations per run |
| `x_pos_SeMIS`, `x_pos_SuS` | posterior samples in the physical parameter space (last run) |

## Citation

If you use this code in your research, please cite:

> Sequential multiple importance sampling for multimodal Bayesian inference. *Mechanical Systems and Signal Processing*.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
