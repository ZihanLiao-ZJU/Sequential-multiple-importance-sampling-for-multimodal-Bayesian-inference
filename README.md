# Sequential Multiple Importance Sampling for Bayesian Inference

MATLAB toolbox for estimating Bayesian evidence and drawing posterior samples.

## Quick start

Open MATLAB in this repository. The common SeMCS core is included; no separate core checkout is needed.

```matlab
addpath(genpath(pwd));
setup_semcs();
rng(1);
result = run_example('semis_bayes_loggamma2');
logZ = result.logZ;
Ncal = result.Ncal;
```

To inspect the example settings before sampling:

```matlab
settings = run_example('semis_bayes_loggamma2', true);
```

Available example names: `semis_bayes_eggbox2`, `semis_bayes_5Gaussian2`, `semis_bayes_loggamma2`, `semis_bayes_loggamma5`, `semis_bayes_loggamma10`, `semis_bayes_loggamma20`.

## Architecture

- `+semcs/`: common sequential engine and MCMC implementations.
- `MCMC-samplers/`: public sampler classes backed by the common core.
- `Likelihood-functions/`: example likelihoods and prior distributions.
- `support functions/`: numerical integration, weights and supporting operations.
- `SeMIS_Bay_Nataf.m`: intermediate distributions, seed selection and stopping rules.
- `G`: evolving method parameters and sampler settings.

The example runner uses coordinate elliptical slice sampling and `source_legacy_mis` evidence integration. Intermediate-distribution construction stays in the method class; the common engine controls the sequence.

The existing `drive_SuS_SeMIS.m` script and named integration options in `estimate_evidence.m` remain available.

## Reference

[Source paper](https://doi.org/10.1016/j.ymssp.2025.113788).

MIT license: see `LICENSE`; the bundled common core retains its notice in `LICENSE_SEMCS_CORE`.
