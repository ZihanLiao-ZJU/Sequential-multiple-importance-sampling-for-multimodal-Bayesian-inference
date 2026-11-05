# SeMIS Bayesian — unified SeMCS (matlab)

This update retains the tested numerical algorithms and uses the common SeMCS engine and common MCMC implementation. Method-specific state and policies remain in the method classes; state evolves in G. The common core is bundled byte for byte so installation does not depend on an unreleased reliability repository.

## Qualified entrypoint

The qualified profile is `author_latest with explicit coordinate-pi ES each_step overrides`, with estimator `source_legacy_mis`. The existing drivers and optional policies remain available for compatibility; use `run_qualified` for the settings in this table. No default is silently presented as paper-qualified.

```matlab
addpath(pwd); addpath(genpath(pwd));
setup_semcs();
settings = run_qualified('semis_bayes_5Gaussian2', true); % construct/inspect only
% result = run_qualified('semis_bayes_5Gaussian2'); % one actual run
```

## Source-paper comparison

Each row uses its declared independent repeat count. Elog is mean Ncal × [SD(log Z)/|printed source log Z reference|]², with sample SD (ddof=1). R=Elog/source Elog; the fixed acceptance limit is R≤1.10.

| Case | Per-level N | Repeats | Mean log Z | SD log Z | Mean Ncal | Elog | Paper Elog | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| semis_bayes_5Gaussian2 | 2606 | 1000 | -5.5083341 | 0.076726762 | 34440.177 | 6.6725744 | 6.160799 | 1.0830697 |
| semis_bayes_eggbox2 | 1418 | 1000 | 235.85055 | 0.14247466 | 32386.808 | 0.011817751 | 0.011196 | 1.0555333 |
| semis_bayes_loggamma10 | 1742 | 1000 | -40.971625 | 0.36140125 | 181065.12 | 14.109722 | 13.024356 | 1.0833335 |
| semis_bayes_loggamma2 | 1702 | 100 | -8.1899691 | 0.087534026 | 20335.43 | 2.3229473 | 2.459367 | 0.94453057 |
| semis_bayes_loggamma20 | 1841 | 1000 | -82.085657 | 0.74239119 | 444095 | 36.498898 | 33.965184 | 1.0745974 |
| semis_bayes_loggamma5 | 1813 | 1000 | -20.467593 | 0.19052941 | 86992.985 | 7.5365499 | 7.337456 | 1.0271339 |

Source papers: [DOI 10.1016/j.ymssp.2025.113788](https://doi.org/10.1016/j.ymssp.2025.113788).

## Scope and limitations

- Qualification applies to the listed profile, estimator, targets and budgets; it does not cover every optional setting.
- No global finite-sample unbiasedness or universal posterior accuracy claim is made.
- The migration bias check uses100 same-seed native/shared pairs per case, not1000 paired runs; the1000 cohorts are independent precision studies.
- Paper efficiency points are treated as fixed; their Monte Carlo uncertainty and rounding are not included. Confidence intervals do not waive the1.10 point gate.
- Existing100-run cases are listed as100, never relabelled1000. Prior100 and1000 cohorts are not pooled.
- MATLAB LogGamma10 mean(Z)/Zref is about1.0364 (~3.09 sample standard errors); this inherited-versus-new-bias attribution is unresolved. Exact migration pairs rule out differences on their tested histories, not systematic bias outside them.
- Negative mean(log Z) shifts, especially LogGamma20, cannot by themselves establish evidence-scale bias because log is nonlinear.
- The qualified estimator is legacy MIS. paper_ars_20261103, paper_raw_mis and raw_density_ratio_mis remain optional inherited interfaces and are not qualified by this release.

See `VALIDATION_SUMMARY.json` for exact profile settings and source-report SHA bindings; `SEMCS_CORE_MANIFEST.json` binds every bundled core file. Existing licenses are retained, with the core notice in `LICENSE_SEMCS_CORE`. No private papers, per-run research outputs or local machine paths are bundled.

## Open evidence-scale calibration

MATLAB LogGamma20 mean(Z)/Zref is1.07657, with bootstrap95% interval[1.02154,1.13417]. This evidence-scale calibration question remains OPEN; the Elog criterion and exact migration pairs do not establish unbiased evidence. The literal fixed source N1700 experiment has not been executed; listed N values are the registered cost-matched budgets.
