classdef SeMISPaperARSPolicy
    % Client-owned opt-in method policy. No sampling loop or kernel body.
    methods(Static)
        function yes=enabled(obj)
            yes=isfield(obj.G,'method_policy')&&strcmp(obj.G.method_policy,'paper_ars_20261103');
        end
        function obj=initialize(obj)
            d=obj.Ndim;
            obj.G.method_policy='paper_ars_20261103';
            obj.G.parameters=[-inf,-inf];
            obj.G.paper_controls=struct('beta_mode','definition_new_over_old', ...
                'seed_selection','bernoulli_ars','seed_reselection','uniform_without_replacement', ...
                'max_conditional_iterations',10*d,'max_stored_levels',10*d+1, ...
                'legacy_MaxIte',10*d+2,'log_r_stop',-1e-4,'stop_before_next_seed_selection',true);
            obj.G.solver=struct('method','monotone_cached_bisection','xtol',1e-8, ...
                'logprob_tol',1e-10,'max_iterations',200);
            obj.G.p=.1;obj.G.run.NumSam=1000;obj.G.run.MaxIte=10*d+2;
            obj.G.run.level_offset=1;obj.G.run.select_at_cap=false;
            obj.G.run.stop_before_seeds=true;obj.G.run.schedule={};
            obj.G.estimator_policy=struct('primary','paper_raw_mis','diagnostics',{{'source_legacy_mis'}});
            obj.G.paper_runtime=struct('previous_parameters',[-inf,-inf], ...
                'sampled_levels',0,'conditional_iterations',0,'chain_plan',[]);
        end
        function [y,g,obj,Ncal]=advance(obj,x,y,p)
            obj.G.paper_runtime.sampled_levels=obj.G.paper_runtime.sampled_levels+1;
            obj.G.paper_runtime.conditional_iterations=obj.G.paper_runtime.sampled_levels-1;
            [g,obj]=obj.UpdObj(y,p);Ncal=0;
        end
        function [xs,ys,steps]=select_seeds(obj,x,y,N)
            plan=obj.G.paper_runtime.chain_plan;
            assert(~isempty(plan)&&plan.source_stored_level==obj.G.paper_runtime.sampled_levels, ...
                'SeMISPaperARS:Plan','No current G-owned chain plan.');
            assert(plan.target_N==N,'SeMISPaperARS:Budget','Requested budget differs from G plan.');
            xx=reshape(x,obj.Ndim,[]);yy=reshape(y,obj.Nfun,[]);
            xs=xx(:,plan.selected_indices);ys=yy(:,plan.selected_indices);steps=plan.Ns;
            if ~isempty(plan.selected_indices),ys=obj.UpdY(ys);end
        end
        function yes=stop(obj)
            yes=obj.G.paper_runtime.conditional_iterations>=1&&obj.FlgCvg;
        end
        function out=annotate(out)
            state=out.intBay.G.paper_runtime.termination;
            out.primary_estimator_id=out.intBay.G.estimator_policy.primary;
            out.model_converged=state.model_converged;
            out.engine_completed=out.flg_cvg;out.raw_engine_completed=out.flg_cvg;
            out.paper_cap_reached=state.paper_cap_reached;
            out.terminated_by_paper_cap=state.terminated_by_paper_cap;
            out.empty_seed_before_cap=state.empty_seed_before_cap;
            out.paper_procedure_completed=state.paper_procedure_completed;
        end
        function [g,obj]=UpdObj(obj,y,p)
            assert(p==obj.G.p,'PaperARS:Probability','G.p owns expected acceptance.');
            cached=reshape(y,4,[]);ll=cached(3,:);old=obj.G.parameters;
            assert(~any(isnan(ll))&&~any(ll==inf)&&any(isfinite(ll)),'PaperARS:LL','Invalid cached likelihood pool.');
            maximum=max(max(ll,[],'all'),old(2));
            [parameters,diagnostic]=SeMISPaperARSPolicy.threshold_root(ll,old,maximum,p,obj.G.solver);
            obj.G.paper_runtime.previous_parameters=old;obj.G.parameters=parameters;
            lb=SeMISPaperARSPolicy.raw_beta(ll,old,obj.G.parameters);
            diagnostic.logmean_raw_beta=logmean(lb,'all');
            diagnostic.max_raw_beta=exp(max(lb,[],'all'));
            diagnostic.objective_log_error=abs(diagnostic.logmean_raw_beta-log(p));
            diagnostic.probability_converged=diagnostic.objective_log_error<=obj.G.solver.logprob_tol;
            diagnostic.interval_converged=diff(diagnostic.final_bracket)<=obj.G.solver.xtol;
            diagnostic.bounded_optimum=strcmp(diagnostic.reason,'upper_bound')&&~diagnostic.probability_converged;
            if diagnostic.probability_converged,diagnostic.status='root_logprob_tolerance';
            elseif diagnostic.bounded_optimum,diagnostic.status='upper_bound_optimum';
            elseif diagnostic.interval_converged,diagnostic.status='interval_tolerance_only';
            else,diagnostic.status='iteration_limit';end
            obj.G.paper_runtime.last_parameter_search=diagnostic;
            g=obj.G;
        end
        function [obj,Ncal]=after_advance(obj,x,y,g)
            % This callback returns G. The value-class select_seeds ABI cannot
            % return modified model state, so prepare stochastic indices here.
            % The draw order is still update -> stop/cap check -> next seeds.
            Ncal=0;
            runtime=obj.G.paper_runtime;model_converged=obj.stop();
            cap=runtime.sampled_levels>=obj.G.paper_controls.max_stored_levels;
            obj.G.paper_runtime.termination=struct('model_converged',model_converged, ...
                'paper_cap_reached',cap,'terminated_by_paper_cap',cap&&~model_converged, ...
                'empty_seed_before_cap',false,'paper_procedure_completed',model_converged||cap);
            if model_converged||cap,obj.G.paper_runtime.chain_plan=[];return;end
            old=obj.G.paper_runtime.previous_parameters;yy=reshape(y{end},4,[]);
            lb=SeMISPaperARSPolicy.raw_beta(yy(3,:),old,obj.G.parameters);prob=exp(lb);
            accepted=find(rand(size(prob))<prob);M=numel(accepted);N=obj.G.run.NumSam;
            validateattributes(N,{'numeric'},{'scalar','integer','positive','finite'});
            if M
                candidate_counts=round(N./(1:N));
                Nc=max(candidate_counts(candidate_counts<=M));Ns=round(N/Nc);
                positions=randperm(M,Nc);chosen=accepted(positions);
            else,Nc=0;Ns=0;positions=[];chosen=[];end
            obj.G.paper_runtime.last_seed_selection=struct('pool_size',numel(prob),'accepted_count',M, ...
                'Nc',Nc,'Ns',Ns,'sampled_next_count',Nc*Ns,'accepted_indices',accepted, ...
                'selected_indices',chosen,'accepted_subset_positions',positions, ...
                'logmean_raw_beta',logmean(lb,'all'),'max_raw_beta',max(prob,[],'all'), ...
                'selection','U<raw_beta; equal subset without replacement');
            % chain_plan alone is live; last_seed_selection is its event receipt.
            obj.G.paper_runtime.chain_plan=struct('Nc',Nc,'Ns',Ns,'actual_count',Nc*Ns, ...
                'source_stored_level',runtime.sampled_levels,'target_N',N,'selected_indices',chosen);
            if Nc==0,obj.G.paper_runtime.termination.empty_seed_before_cap=true;end
        end
        function value=factor(ll,parameters)
            if isinf(parameters(1))&&parameters(1)<0,value=zeros(size(ll));
            else,value=min(ll-(parameters(1)+parameters(2)),0);end
        end
        function ratio=raw_beta(ll,old,new)
            if isinf(old(1))&&old(1)<0,tau_old=-inf;else,tau_old=old(1)+old(2);end
            tau_new=new(1)+new(2);
            assert(isfinite(tau_new)&&tau_new>=tau_old,'PaperARS:Threshold','Threshold must be finite and nondecreasing.');
            ratio=min(max(ll,tau_old)-tau_new,0);
            assert(~any(isnan(ratio))&&all(ratio<=0),'PaperARS:Beta','Probability exceeds analytic bound; no empirical rescaling.');
        end
        function [parameters,info]=threshold_root(ll,old,maximum,p,solver)
            assert(strcmp(solver.method,'monotone_cached_bisection'),'PaperARS:Solver','Unknown cached solver.');
            if isinf(old(1))&&old(1)<0,tau_old=-inf;else,tau_old=old(1)+old(2);end
            finite=ll(isfinite(ll));lo=max(tau_old,min(finite));hi=maximum;
            assert(isfinite(lo)&&isfinite(hi)&&lo<=hi,'PaperARS:Bracket','Invalid cached bracket.');
            bracket=[lo,hi];target=log(p);
            objective=@(tau) logmean(min(max(ll,tau_old)-tau,0),'all');
            upper=objective(hi);lower=objective(lo);iterations=0;
            if upper>=target||abs(upper-target)<=solver.logprob_tol
                tau=hi;reason='upper_bound';
            elseif lower<target-solver.logprob_tol
                error('PaperARS:TargetUnattainable','Target probability is unattainable at lower cached bracket.');
            elseif abs(lower-target)<=solver.logprob_tol
                tau=lo;reason='lower_endpoint_tolerance';
            else
                reason='max_iterations';
                for iterations=1:solver.max_iterations
                    tau=lo+(hi-lo)/2;value=objective(tau);
                    if abs(value-target)<=solver.logprob_tol,reason='logprob_tolerance';break;end
                    if hi-lo<=solver.xtol,reason='threshold_tolerance';break;end
                    if value>target,lo=tau;else,hi=tau;end
                end
            end
            parameters=[tau-maximum,maximum];
            info=struct('solver',solver.method,'tau_old',tau_old,'initial_bracket',bracket, ...
                'final_bracket',[lo,hi],'tau_requested',tau,'tau_effective',parameters(1)+parameters(2), ...
                'lower_logmean_beta',lower,'upper_logmean_beta',upper,'iterations',iterations,'reason',reason, ...
                'probability_converged',false,'interval_converged',hi-lo<=solver.xtol, ...
                'bounded_optimum',strcmp(reason,'upper_bound'));
        end
    end
end
