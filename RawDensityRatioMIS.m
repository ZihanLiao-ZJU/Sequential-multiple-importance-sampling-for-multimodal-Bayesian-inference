classdef RawDensityRatioMIS < MIS
    % Explicit cached source-SIR diagnostic, separate from paper_raw_mis.
    properties(Constant)
        EstimatorId='raw_density_ratio_mis'
        ProposalConvention='source_sir'
    end
    properties
        LegacyDiagnostic
    end
    properties(Dependent)
        estimates
        log_estimates
    end
    methods
        function obj=RawDensityRatioMIS(raw)
            assert(~SeMISPaperARSPolicy.enabled(raw.intBay),'RawMIS:Scope', ...
                'source_sir only; preserve PaperRawMIS for the explicit paper ARS policy.');
            assert(ismethod(raw.intBay,'proposal_log_ratio'),'RawMIS:Protocol','A pure cached source proposal ratio is required.');
            K=raw.Nite;assert(isscalar(K)&&isfinite(K)&&K>=1&&K==floor(K)&& ...
                numel(raw.y)==K&&numel(raw.g)==K+1&&numel(raw.Nsam)==K, ...
                'RawMIS:History','Need at least one sampled state plus next state.');
            assert(isequaln(raw.g{1}.parameters,[-inf,-inf]),'RawMIS:Initial','Only known normalized prior h0=1,C0=0 is supported.');
            for k=1:K
                assert(size(raw.y{k},1)==4,'RawMIS:PackedRows','Expected four raw packed rows.');
                assert(raw.Nsam(k)==numel(raw.y{k})/4,'RawMIS:Counts','Use actual packed sample count.');
            end
            obj@MIS(raw);
            obj.LegacyDiagnostic=struct('estimator_id','source_legacy_mis','Z',obj.Z,'C',obj.C, ...
                'estimates',exp(obj.Z),'log_estimates',obj.Z,'same_history',true,'model_calls',0,'rng_draws',0);
            before=rng;C=zeros(K+1,1);wr=cell(K,1);working=raw.y;model=raw.intBay;
            for k=1:K
                block=reshape(raw.y{k},4,[]);ll=block(3,:);g=raw.g{k}.parameters;
                if isinf(g(1)),expected=zeros(size(ll));else,expected=min(ll-g(2)-g(1),0);end
                assert(isequaln(block(1,:),expected),'RawMIS:Prefix','Stored prefix differs from the source soft factor.');
                assert(isequaln(block(2,:),block(4,:)),'RawMIS:Prior','Known fixed normalized Gaussian prior rows must agree.');
                ratio=model.proposal_log_ratio(ll,raw.g{k},raw.g{k+1});
                assert(~any(isnan(ratio))&&any(isfinite(ratio)),'RawMIS:Ratio','Undefined or zero raw density ratio.');
                wr{k}=reshape(ratio,size(raw.y{k},2:3));C(k+1)=C(k)+logmean(ratio,'all');
            end
            assert(all(isfinite(C)),'RawMIS:Normalizer','Finite soft-proposal normalizers required.');
            for k=1:K,working{k}(1,:)=working{k}(1,:)-C(k);end
            weights=WgtMis_Blc(model,raw.g,C,working);target=WgtPos(working);
            Zi=MisInt(weights,target);Z=logsum(Zi,'all');
            for k=1:K,target{k}=target{k}-Z;end
            obj.C=C;obj.Wmis=weights;obj.Wpos=target;obj.Wres=wr;obj.Zi=Zi;obj.Z=Z;
            assert(isequaln(before,rng),'RawMIS:RNG','Cached evidence estimation must not change RNG.');
        end
        function value=get.estimates(obj),value=exp(obj.Z);end
        function value=get.log_estimates(obj),value=obj.Z;end
    end
end
