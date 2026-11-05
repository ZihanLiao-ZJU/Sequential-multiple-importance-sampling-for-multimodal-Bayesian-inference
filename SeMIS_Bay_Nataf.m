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
        function obj=apply_method_profile(obj,spec)
            if SeMISPaperARSPolicy.enabled(obj)&&~strcmp(spec.name,'paper_ars_20261103')
                error('SeMISPaperARS:FreshModelRequired', ...
                    'Create a fresh model to change from the initialized paper ARS method policy.');
            end
            if strcmp(spec.name,'paper_ars_20261103'),obj=SeMISPaperARSPolicy.initialize(obj);end
        end
        function obj = SeMIS_Bay_Nataf(LKF, method_profile)
            if nargin < 2 || isempty(method_profile), method_profile='author_latest'; end
            setup_semcs();
            % Construct with target model LKF defined in X-space
            obj.LKF  = LKF;
            obj.Ndim = LKF.Ndim;
            obj=semcs.configure_method_profile(obj,'semis_bayes',method_profile);
        end

        function logLi = EvlLKF(obj)
            if SeMISPaperARSPolicy.enabled(obj)
                logLi=SeMISPaperARSPolicy.factor(obj.Y(1,:),obj.G.parameters);return;
            end
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
            if SeMISPaperARSPolicy.enabled(obj)
                [G,obj]=SeMISPaperARSPolicy.UpdObj(obj,y,p);return;
            end
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
            if SeMISPaperARSPolicy.enabled(obj)
                FlgCvg=obj.G.parameters(1)>=obj.G.paper_controls.log_r_stop;return;
            end
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
            if SeMISPaperARSPolicy.enabled(obj)
                [y,g,obj,Ncal]=SeMISPaperARSPolicy.advance(obj,x,y,p);return;
            end
            [g,obj]=obj.UpdObj(y,p);Ncal=0;
        end
        function [xs,ys,steps]=select_seeds(obj,x,y,N)
            if SeMISPaperARSPolicy.enabled(obj)
                [xs,ys,steps]=SeMISPaperARSPolicy.select_seeds(obj,x,y,N);return;
            end
            [xs,ys]=SedSlt(obj,x,y,N);steps=round(N/size(xs,2));
        end
        function yes=stop(obj)
            if SeMISPaperARSPolicy.enabled(obj),yes=SeMISPaperARSPolicy.stop(obj);
            else,yes=obj.FlgCvg;end
        end
        function [obj,Ncal]=after_advance(obj,x,y,g)
            Ncal=0;
            if SeMISPaperARSPolicy.enabled(obj),[obj,Ncal]=SeMISPaperARSPolicy.after_advance(obj,x,y,g);end
        end
        function ratio=proposal_log_ratio(~,ll,oldG,newG)
            % Pure source-SIR density ratio: cached terms, no ARS assumptions.
            assert(isnumeric(ll)&&isreal(ll)&&all(isfinite(ll)|ll==-inf,'all'), ...
                'RawMIS:LogLikelihood','Cached log likelihood must be finite or negative infinity.');
            old=oldG.parameters;next=newG.parameters;
            ratio=factor(next)-factor(old);
            assert(~any(isnan(ratio)|ratio==inf,'all'),'RawMIS:Ratio','Undefined or infinite positive density ratio.');
            function value=factor(g)
                assert(isnumeric(g)&&isreal(g)&&isvector(g)&&numel(g)==2&& ...
                    (isequal(g(:)',[-inf,-inf])||all(isfinite(g),'all')), ...
                    'RawMIS:Parameters','Expected [-Inf,-Inf] prior or finite [r,rmax].');
                if g(1)==-inf,value=zeros(size(ll));
                else,value=min(ll-g(2)-g(1),0);end
            end
        end
        function ratio=raw_log_ratio(obj,ll,oldG,newG)
            assert(SeMISPaperARSPolicy.enabled(obj),'SeMISPaperARS:Policy','Raw ratio API belongs to the explicit paper ARS policy.');
            ratio=SeMISPaperARSPolicy.raw_beta(ll,oldG.parameters,newG.parameters);
        end

    end

end

