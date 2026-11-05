function result=run_example(caseName,configurationOnly)
% Configure and run a Bayesian example using the common SeMCS engine.
if nargin<2,configurationOnly=false;end
root=fileparts(mfilename('fullpath'));addpath(genpath(root));setup_semcs();
switch char(caseName)
 case 'semis_bayes_5Gaussian2', target=LKF_2D_FivePeak(); N=2606;
 case 'semis_bayes_eggbox2', target=LKF_2D_Eggbox(); N=1418;
 case 'semis_bayes_loggamma10', target=LKF_nD_GaussianLogGamma(10); N=1742;
 case 'semis_bayes_loggamma2', target=LKF_nD_GaussianLogGamma(2); N=1702;
 case 'semis_bayes_loggamma20', target=LKF_nD_GaussianLogGamma(20); N=1841;
 case 'semis_bayes_loggamma5', target=LKF_nD_GaussianLogGamma(5); N=1813;
 otherwise,error('SeMCS:Case','Choose one of the example cases listed in README.md.');
end
model=SeMIS_Bay_Nataf(target,'author_latest');
sampler=ES(model);sampler.ESMode='coordinate';sampler.AcceptArithmetic='exp';sampler.SeedUpdateMode='each_step';sampler.Ntry=20;sampler.SedSav=false;simulation=SeMCS(sampler);
simulation.NumSam=N;simulation.MaxIte=10000;simulation.ConPrb=.1;
result=struct('case',char(caseName),'N',N,'profile','author_latest with explicit coordinate-pi ES each_step overrides','estimator','source_legacy_mis','modelG',model.G,'samplerG',sampler.BayStc.G);
if configurationOnly,return;end
raw=simulation.RunIte();post=estimate_evidence(raw,'Estimator','source_legacy_mis','Diagnostics',false);logZ=post.log_estimates;
result.logZ=logZ;result.Ncal=sum(raw.Ncal);disp(result.logZ);disp(result.Ncal);
end
