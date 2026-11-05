classdef SeMIS_Bay_Nataf
    % SeMIS Bayesian updating in Nataf-transformed (standard normal) space.
    %
    % Paper-aligned naming:
    %   - r_i    (implementation)  <->  lambda_i (theory)
    %   - r_max  (implementation)  <->  r^{max}_{0:i-1} (running max of log-likelihood)
    %
    % Key implementation note:
    %   EvlLKF() computes the SOFT TRUNCATION term (logLi):
    %       logLi(theta) = log( min( L(theta) / exp(r_max + r_i), 1 ) )
    %                    = min( logL(theta) - r_max - r_i, 0 )
    %   The ratio beta_i(theta) in Eq. (24) is constructed elsewhere
    %   (e.g., in ResRat) as a ratio of two logLi terms.

    properties
        LKF            % Likelihood/prior wrapper (must provide .Ndim, .EvlLKF, .Pdis.U2X)
        Nfun = 4;      % Reserved / kept for compatibility
        Ndim           % Parameter dimension
        Ncal           % #evaluations in the last EvlY call
        X              % Samples in U-space, size [Ndim, Nsam]
        Y              % Cached base terms associated with X (see EvlY / UpdY)
    end

    properties
        % Intermediate threshold parameters are stored in G.parameters = [r_i, r_max]
        %   r_i   : adaptive threshold controlling soft truncation (lambda_i)
        %   r_max : running maximum of log-likelihood (r^{max}_{0:i-1})
        G = struct('parameters', [-inf, -inf], 'lambda', 0.6);
    end

    properties(Dependent)
        FlgCvg         % Convergence flag based on r_i
    end

    methods
        function spec = sampler_profile(~,family,name)
            spec=method_sampler_profile(family,name);
        end
        function obj = SeMIS_Bay_Nataf(LKF, method_profile)
            if nargin < 2 || isempty(method_profile), method_profile='author_latest'; end
            % Construct with target model LKF defined in X-space
            obj.LKF  = LKF;
            obj.Ndim = LKF.Ndim;
            obj=semcs.configure_method_profile(obj,'semis_bayes',method_profile);
        end

        function logLi = EvlLKF(obj)
            % Evaluate SOFT TRUNCATION term logLi(theta).
            %
            % Inputs (from obj.Y):
            %   logL = obj.Y(1,:) : log-likelihood values
            %
            % Parameters (from obj.G):
            %   r_i   = G(1)      : adaptive threshold (lambda_i)
            %   r_max = G(2)      : running max log-likelihood
            %
            % Output:
            %   logLi(theta) = min( logL - r_max - r_i , 0 )

            y    = obj.Y;
            logL = y(1,:);

            r_i   = obj.G.parameters(1);
            r_max = obj.G.parameters(2);

            if isinf(r_i)
                logLi = zeros(size(logL));          % initialization: no truncation
            else
                logLi = min(logL - r_max - r_i, 0); % soft truncation (log domain)
            end
        end

        function logPi = EvlPDF(obj)
            % Return prior log-density logPi(theta).
            logPi = obj.Y(2,:);
        end

        function [G,obj] = UpdObj(obj,y,p)
            % Update intermediate parameters G = [r_i, r_max].
            %
            % Inputs:
            %   y : function table (log-likelihood in row 3: y(3,:))
            %   p : target conditional probability
            %
            % Output:
            %   G : updated [r_i, r_max]
            %
            % r_i is determined by solving Eq. (25) via 1D constrained search,
            % where beta_i(theta) is constructed in ResRat(...) from logLi terms.

            % --- update r_max (running maximum of log-likelihood) ---
            logL  = y(3,:);
            r_i   = obj.G.parameters(1);
            r_max = max(max(logL,[],"all"), obj.G.parameters(2));

            dr_max = r_max - obj.G.parameters(2);

            % --- search r_i such that logmean(beta_i) ~= log(p) ---
            maxi    = 100;
            err_min = inf;

            fun = @(ri) abs( ...
                logmean(ResRat(obj, y, [ri, r_max]), "all") - log(p) ...
            );

            % Bounds (kept identical to original implementation)
            rmin = max(-1e12, r_i - dr_max);
            rmax = 0;

            rbnd = (rmax - rmin) / maxi;
            for i = 1:maxi
                [r_tmp, err] = fminbnd(fun, rmin+(i-1)*rbnd, rmin+i*rbnd);
                if err < err_min
                    err_min = err;
                    r_i = r_tmp;
                end
            end

            G = [r_i, r_max];
            obj.G.parameters = G;
            G = obj.G;
        end

        function FlgCvg = get.FlgCvg(obj)
            % Convergence criterion (unchanged)
            FlgCvg = obj.G.parameters(1) >= -1e-4;
        end

        function obj = EvlY(obj,u)
            % Evaluate and cache base terms at samples u (U-space).
            %
            % Side effects:
            %   obj.Y = [logL; logPi]
            %   where:
            %     logL  = log-likelihood in X-space
            %     logPi = log-prior in U-space

            x     = U2X(obj,u);
            logL  = obj.LKF.EvlLKF(x);
            logPi = logGauss(u);

            obj.X    = u;
            obj.Y    = [logL; logPi];
            obj.Ncal = size(u,2);
        end

        function y = UpdY(obj,y)
            % Refresh and prepend intermediate terms.
            %
            % Output layout:
            %   y := [logLi; logPi; base_terms...]

            y = y(3:end,:);          % drop existing intermediate rows
            obj.Y = y;

            logPi = obj.EvlPDF;
            logLi = obj.EvlLKF;

            y = [logLi; logPi; y];
        end

        function x = U2X(obj,u)
            % Map U-space samples to X-space via Nataf transformation
            x = obj.LKF.Pdis.U2X(u);
        end
    end
    methods
        function [y,g,obj,Ncal]=advance(obj,x,y,p)
            [g,obj]=obj.UpdObj(y,p);Ncal=0;
        end
        function [xs,ys,steps]=select_seeds(obj,x,y,N)
            [xs,ys]=SedSlt(obj,x,y,N);steps=round(N/size(xs,2));
        end
        function yes=stop(obj),yes=obj.FlgCvg;end

    end

end

function spec = method_sampler_profile(family, name)
% Method-owned sampling settings; no budgets, thresholds or stops are changed.
% 'author_latest' follows the user's selected latest local author source policy.
% It does not authenticate a historical table-generation commit.
if nargin < 2 || isempty(name), name='author_latest'; end
family=char(family); name=char(name);
if strcmp(name,'paper_settings'), name='paper'; end
if ~any(strcmp(name,{'author_latest','paper','public_20261102'}))
    error('method_sampler_profile:Name','Choose author_latest, paper, paper_settings or public_20261102.');
end
if ~any(strcmp(family,{'semis_bayes','sus_bayes'}))
    error('method_sampler_profile:Family','Unsupported method family for this toolbox.');
end
options=struct();
options.aCS=struct('Sigma0Mode','unit','CarryLambda',true);
options.ES=struct('ESMode','coordinate','Ntry',20);
default_sampler='aCS';
support='public20261102 unit/carry aCS and coordinate ES replay';
if any(strcmp(family,{'gss_mis'})), default_sampler='CW_aMH'; end
if strcmp(family,'semis_bayes'), default_sampler='ES'; end
if strcmp(family,'resus'), default_sampler='ES'; end
if strcmp(name,'paper')
    switch family
        case 'semis_rel'
            options.aCS=struct('Sigma0Mode','unit','CarryLambda',false);
            support='retained SeMIS reliability src v1 aCS: unit sigma0, lambda reset';
        case 'sus_bayes'
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            support='CS-MH article family supplemented by retained author aCS seed std/reset';
        case 'semis_bayes'
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            options.ES=struct('ESMode','global','Ntry',[]);
            support='paper global vector-angle ESS, until accepted; optional aCS author-source supplement';
        case 'resus'
            default_sampler='aCS';
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            options.ES=struct('ESMode','global','Ntry',[]);
            support='paper permits seed std or unit; retained author aCS seed std/reset; paper global ESS';
        case 'gss_mis'
            support='retained author CW_aMH seed std; no lambda setting applies to CW_aMH';
        case 'sus_rel'
            support='optional scalar SuS keeps public aCS settings; no method-paper kernel certification';
    end
end
if strcmp(name,'author_latest')
    switch family
        case 'semis_rel'
            options.aCS=struct('Sigma0Mode','unit','CarryLambda',false);
            support='latest reliability src v1: unit aCS, lambda reset';
        case 'sus_bayes'
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            support='latest Bayesian SuS author aCS: seed std, lambda reset';
        case 'semis_bayes'
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            options.ES=struct('ESMode','coordinate','Ntry',20);
            support='latest SeMIS Bayesian author ES: coordinate, Ntry20; optional aCS seed std/reset';
        case 'resus'
            default_sampler='ES';
            options.aCS=struct('Sigma0Mode','seed_std','CarryLambda',false);
            options.ES=struct('ESMode','coordinate','Ntry',20);
            support='latest ReSuS author driver: ES coordinate/Ntry20 and hybrid midpoint-strict threshold; optional aCS seed std/reset';
        case 'gss_mis'
            support='latest coherent src_MIS: CW seed std; active-history balance MIS';
        case 'sus_rel'
            support='optional scalar SuS retains public aCS settings; outside certified latest SeMIS method scope';
    end
end
options.aCS.AcceptArithmetic='log';options.aCS.CoordinateMode='identity';
options.aCS.GroupMode='available';options.aCS.AcceptanceRateMode='attempts';options.aCS.AdaptEmptyGroups=false;
options.ES.AcceptArithmetic='log';options.ES.CoordinateMode='identity';options.ES.SeedUpdateMode='each_step';
options.CW_aMH=struct('AcceptArithmetic','log','PriorMode','gaussian','CoordinateMode','identity');
if strcmp(name,'author_latest')
    options.aCS.AcceptArithmetic='exp';options.ES.AcceptArithmetic='exp';options.CW_aMH=struct('AcceptArithmetic','exp');
    options.aCS.GroupMode='fixed';options.aCS.AcceptanceRateMode='attempts';options.aCS.AdaptEmptyGroups=true;
    if strcmp(family,'semis_rel')
        options.aCS.GroupMode='available';options.aCS.AcceptanceRateMode='chain_steps';options.aCS.AdaptEmptyGroups=false;
    end
    if strcmp(family,'resus')
        options.aCS.CoordinateMode='model';options.ES.CoordinateMode='model';options.ES.SeedUpdateMode='all_accepted';
    end
end
evidence=struct('support',support, ...
    'historical_table_generator_commit_authenticated',false, ...
    'unknown_paper_budget_or_tolerance_not_filled',true, ...
    'reference_factor_separate_from_kernel_base_scale',true, ...
    'author_source_arithmetic_not_asserted_identical',true);
% Field-level evidence describes support, not historical table authentication.
recorded=struct('status','public_20261102_recorded_policy', ...
    'detail','Retained public defaults; not automatically an article setting');
author=struct('status','author_source_not_explicit_in_paper', ...
    'detail','Retained author aCS source: Lambda .6, NumIte 5, SedSav false, AcpOpt .44; exact table-generation commit unverified');
notcertified=struct('status','optional_kernel_not_method_paper_certified', ...
    'detail','Optional retained kernel policy, outside the method-paper benchmark kernel');
na=struct('status','not_applicable','detail','No lambda adaptation/target/groups in CW_aMH');
evidence.aCS=struct('Sigma0Mode',recorded,'CarryLambda',recorded, ...
    'Lambda',author,'NumIte',author,'SedSav',author,'AcpOpt',author);
evidence.ES=struct('ESMode',recorded,'Ntry',recorded);
evidence.CW_aMH=struct('Sigma0Mode',struct('status','author_source', ...
    'detail','Retained CW_aMH recomputes seed standard deviation per sampling call'), ...
    'Lambda',na,'CarryLambda',na,'AcpOpt',na,'NumIte',na);
if strcmp(name,'paper')
    switch family
        case 'semis_rel'
            evidence.aCS.Sigma0Mode=struct('status','author_source_not_explicit_in_paper', ...
                'source','SeMIS reliability src v1/Support-Fun/aCS.m L58', ...
                'detail','Unit proposal base scale; reference covariance inflation is separate');
            evidence.aCS.CarryLambda=struct('status','author_source_not_explicit_in_paper', ...
                'source','SeMIS reliability src v1/Support-Fun/aCS.m L30,L103', ...
                'detail','Lambda starts from object option at each call; no adapted value written back');
        case 'sus_bayes'
            evidence.aCS.Sigma0Mode=struct('status','article_family_plus_author_source_supplement', ...
                'detail','Seed standard deviation in whitened reference coordinates; exact table-generating implementation unverified');
            evidence.aCS.CarryLambda=struct('status','author_source_not_explicit_in_paper', ...
                'detail','Reset from configured Lambda each call; no cross-level carry claimed from paper text');
        case 'semis_bayes'
            evidence.aCS.Sigma0Mode=notcertified;evidence.aCS.CarryLambda=notcertified;
        case 'resus'
            evidence.aCS.Sigma0Mode=struct('status','paper_allows_choice_author_source_selects', ...
                'source','Li et al. 2025 p3 Sec2.2.1; retained src/github/MCMC-samplers/aCS.m L76', ...
                'detail','Paper permits seed std or unit; this profile selects author-source seed std');
            evidence.aCS.CarryLambda=struct('status','author_source_not_explicit_in_paper', ...
                'source','retained src/github/MCMC-samplers/aCS.m L55,L113', ...
                'detail','Local lambda resets each call; final adaptation not returned to object');
            evidence.aCS.AcpOpt=struct('status','paper_explicit', ...
                'source','Li et al. 2025 p3 Sec2.2.1','detail','Target acceptance approximately .44');
        otherwise
            evidence.aCS.Sigma0Mode=notcertified;evidence.aCS.CarryLambda=notcertified;
    end
    if any(strcmp(family,{'semis_bayes','resus'}))
        if strcmp(family,'resus'),source='Li et al. 2025 p4 Eq12';
        else,source='SeMIS Bayesian paper global elliptical slice sampling description';end
        evidence.ES.ESMode=struct('status','paper_explicit','source',source, ...
            'detail','One angle for full vector, full 2pi bracket, shrink until accepted');
        evidence.ES.Ntry=struct('status','not_applicable','detail','Global paper ESS has no finite rejection-attempt cap');
    end
end
if strcmp(name,'author_latest')
    evidence.policy=struct('status','user_selected_latest_author_source', ...
        'detail','Selected by runnable entry/algorithm file mtime; not historical table identity');
    if any(strcmp(family,{'semis_rel','sus_bayes','semis_bayes','resus'}))
        evidence.aCS.Sigma0Mode=struct('status','latest_author_source', ...
            'detail',['Configured base scale: ' options.aCS.Sigma0Mode]);
        evidence.aCS.CarryLambda=struct('status','latest_author_source', ...
            'detail','Per-call lambda reset from configured option; final runtime value remains recorded');
    end
    if any(strcmp(family,{'semis_bayes','resus'}))
        evidence.ES.ESMode=struct('status','latest_author_source', ...
            'detail','Author coordinate-angle ES is retained instead of replacing it with a paper-formula policy');
        evidence.ES.Ntry=struct('status','latest_author_source','detail','Finite Ntry=20');
    end
end
if any(strcmp(name,{'paper','author_latest'}))&&strcmp(family,'sus_bayes')
    explicit=struct('status','paper_explicit', ...
        'source','SuS-Bay proof p6 Algorithm 2', ...
        'detail','Seed standard deviation; initial lambda .6; target acceptance .44. Exact five groups is an author-source choice within the printed adaptation-fraction range.');
    evidence.aCS.Sigma0Mode=explicit;evidence.aCS.Lambda=explicit;evidence.aCS.AcpOpt=explicit;
end
evidence.author_group_and_acceptance_arithmetic=struct('status','not_replayed_in_this_increment', ...
    'detail','author_latest selects retained source group/exp arithmetic; paper/public retain their recorded numerical policies');
spec=struct('family',family,'name',name,'default_sampler',default_sampler, ...
    'options',options,'evidence',evidence);
end
