# Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference

This repository provides a MATLAB implementation of the **Sequential Multiple Importance Sampling (SeMIS)** algorithm for **multimodal Bayesian inference**.

The code accompanies the paper:

> **Sequential multiple importance sampling for multimodal Bayesian inference**  
> *Mechanical Systems and Signal Processing*

---

## Description

SeMIS is a multiple importance sampling (MIS)–based Bayesian inference algorithm designed for **multimodal and high-dimensional posterior distributions**.  
The algorithm constructs a sequence of **softly truncated prior–based proposal distributions** that smoothly evolve from the prior to the posterior. This strategy improves sample mixing across separated modes while retaining efficient Bayesian evidence estimation.

The implementation integrates:
- Sequentially adapted proposal distributions
- Balance heuristic weighting for MIS
- Parallel MCMC sampling
- Elliptical slice sampling (ESS)
- Bayesian evidence estimation and posterior resampling

---

## Algorithm Summary

At each iteration, SeMIS:
1. Constructs a proposal distribution by softly truncating the prior using likelihood information
2. Adaptively determines the proposal parameter based on a target acceptance rate
3. Generates samples using parallel MCMC chains initialized from accepted seeds
4. Accumulates all samples through multiple importance sampling
5. Estimates Bayesian evidence and resamples posterior samples

The proposal sequence gradually transitions from the prior distribution to the posterior distribution, ensuring robustness for multimodal inference problems.

---

## Requirements

- MATLAB (tested with recent versions)
- No external toolboxes are required beyond standard MATLAB functionality

---

## How to Run

To execute the algorithm and reproduce the numerical examples reported in the paper, simply run:

```matlab

drive_SuS_SeMIS

```
The driver script defines the Bayesian inference problem, executes the SeMIS algorithm (and comparison algorithms when enabled), and outputs Bayesian evidence estimates, posterior samples, and performance metrics.

---

## Main Files

- **`SeMIS_Bay_Nataf.m`**  
  Core implementation of the Sequential Multiple Importance Sampling (SeMIS) algorithm.  
  This file includes:
  - Construction of sequential proposal distributions based on softly truncated priors  
  - Adaptive determination of proposal hyperparameters using a target acceptance rate   
  - Bayesian evidence estimation and posterior resampling  

- **`drive_SuS_SeMIS.m`**  
  Main driver script for running SeMIS and benchmark examples.  
  It is used to reproduce the numerical studies and comparisons reported in the paper.

Auxiliary functions provide support for likelihood evaluation, probabilistic transformations (e.g., Nataf transformation), benchmark problem definitions, and sampling utilities.

---

## Citation

If you use this code in your research, please cite the corresponding paper:

> Sequential multiple importance sampling for multimodal Bayesian inference  
> *Mechanical Systems and Signal Processing*
