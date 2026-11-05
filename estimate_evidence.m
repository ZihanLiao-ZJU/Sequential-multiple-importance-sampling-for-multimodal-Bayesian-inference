function result=estimate_evidence(raw,varargin)
% Named estimator selection; same-history diagnostics never add repetitions.
parser=inputParser;addParameter(parser,'Estimator','primary',@(v)ischar(v)||isstring(v));
addParameter(parser,'Diagnostics',true,@(v)islogical(v)&&isscalar(v));parse(parser,varargin{:});
which_estimator=char(parser.Results.Estimator);
if strcmp(which_estimator,'primary')
    if isfield(raw,'primary_estimator_id'),which_estimator=raw.primary_estimator_id;
    else,which_estimator='source_legacy_mis';end
end
before=rng;diagnostics=struct();
switch which_estimator
    case 'paper_raw_mis'
        primary=PaperRawMIS(raw);
        if parser.Results.Diagnostics,diagnostics.source_legacy_mis=primary.LegacyDiagnostic;end
    case 'raw_density_ratio_mis'
        primary=RawDensityRatioMIS(raw);
        if parser.Results.Diagnostics,diagnostics.source_legacy_mis=primary.LegacyDiagnostic;end
    case 'source_legacy_mis'
        primary=MIS(raw);
    otherwise,error('SeMISBay:Estimator','Choose primary, paper_raw_mis, raw_density_ratio_mis or source_legacy_mis.');
end
result=struct('primary_estimator_id',which_estimator,'primary',primary,'diagnostics',diagnostics, ...
    'Z',primary.Z,'C',primary.C,'estimates',exp(primary.Z),'log_estimates',primary.Z, ...
    'independent_runs',1,'same_history_diagnostics',true,'model_calls',0,'rng_draws',0);
assert(isequaln(before,rng),'SeMISBay:EstimateRNG','Evidence estimation must not alter RNG.');
end
