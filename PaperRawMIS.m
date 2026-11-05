classdef PaperRawMIS < MIS
    % Explicit opt-in raw-density normalizer; source MIS/ResRat stay unchanged.
    % Inherits posterior/query methods; they use the new primary weights.
    properties(Constant)
        EstimatorId='paper_raw_mis'
    end
    properties
        LegacyDiagnostic
    end
    properties(Dependent)
        estimates
        log_estimates
    end
    methods
        function obj=PaperRawMIS(raw,initial_log_normalizer)
            obj@MIS(raw); % Exact old estimator is computed once as a diagnostic.
            if nargin<2,initial_log_normalizer=0;end
            validateattributes(initial_log_normalizer,{'numeric'},{'scalar','real','finite'});
            assert(ismethod(raw.intBay,'raw_log_ratio'),'PaperRawMIS:Protocol','Model must provide its cached raw ratio.');
            obj.LegacyDiagnostic=struct('estimator_id','source_legacy_mis','Z',obj.Z,'C',obj.C, ...
                'estimates',exp(obj.Z),'log_estimates',obj.Z,'same_history',true,'model_calls',0,'rng_draws',0);
            previous_rng=rng;K=raw.Nite;working=raw.y;C=zeros(K+1,1);C(1)=initial_log_normalizer;wr=cell(K,1);
            model=raw.intBay;
            for k=1:K
                block=reshape(raw.y{k},size(raw.y{k},1),[]);
                ratio=model.raw_log_ratio(block(3,:),raw.g{k},raw.g{k+1});
                assert(~any(isnan(ratio)),'PaperRawMIS:Ratio','Undefined raw density ratio.');
                wr{k}=reshape(ratio,size(raw.y{k},2:3));C(k+1)=C(k)+logmean(ratio,'all');
            end
            for k=1:K,working{k}(1,:)=working{k}(1,:)-C(k);end
            weights=WgtMis_Blc(model,raw.g,C,working);target=WgtPos(working);
            Zi=MisInt(weights,target);Z=logsum(Zi,'all');
            for k=1:K,target{k}=target{k}-Z;end
            obj.C=C;obj.Wmis=weights;obj.Wpos=target;obj.Wres=wr;obj.Zi=Zi;obj.Z=Z;
            assert(isequaln(previous_rng,rng),'PaperRawMIS:RNG','Cached estimation must not draw random numbers.');
        end
        function value=get.estimates(obj),value=exp(obj.Z);end
        function value=get.log_estimates(obj),value=obj.Z;end
    end
end
