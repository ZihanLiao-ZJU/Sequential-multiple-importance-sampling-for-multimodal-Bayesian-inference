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
        MethodProfile = 'author_latest'
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
        function obj = SeMIS_Bay_Nataf(LKF, method_profile)
            if nargin < 2 || isempty(method_profile)
                method_profile = 'author_latest';
            end
            % Construct with target model LKF defined in X-space
            obj.LKF  = LKF;
            obj.Ndim = LKF.Ndim;
            if ~any(strcmp(char(method_profile), {'author_latest', 'paper'}))
                error('method_sampler_profile:Name', 'Choose author_latest or paper.');
            end
            obj.MethodProfile = method_profile;
            obj.G.p = 0.1;
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

        function [y,G,obj,N_cal] = UpdObj(obj,~,y,p)
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
            % Prescribed G.p remains the live conditional-probability setting.
            p = obj.G.p;
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
            N_cal = 0;
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

        function options = sampler_options(obj, kernel)
            name = obj.MethodProfile;
            if isempty(name)
                name = 'author_latest';
            end
            name = char(name);
            if ~any(strcmp(name, {'author_latest', 'paper'}))
                error('method_sampler_profile:Name', 'Choose author_latest or paper.');
            end

            settings = struct();
            settings.aCS = struct('Sigma0Mode', 'seed_std', 'CarryLambda', false);
            settings.ES = struct('ESMode', 'coordinate', 'Ntry', 20);
            if strcmp(name, 'paper')
                settings.ES = struct('ESMode', 'global', 'Ntry', []);
            end

            settings.aCS.AcceptArithmetic = 'log';
            settings.aCS.CoordinateMode = 'identity';
            settings.aCS.GroupMode = 'available';
            settings.aCS.AcceptanceRateMode = 'attempts';
            settings.ES.AcceptArithmetic = 'log';
            settings.ES.CoordinateMode = 'identity';
            settings.ES.SeedUpdateMode = 'each_step';
            settings.CW_aMH = struct('AcceptArithmetic', 'log', 'PriorMode', 'gaussian', 'CoordinateMode', 'identity');
            if strcmp(name, 'author_latest')
                settings.aCS.AcceptArithmetic = 'exp';
                settings.ES.AcceptArithmetic = 'exp';
                settings.CW_aMH = struct('AcceptArithmetic', 'exp');
                settings.aCS.GroupMode = 'fixed';
            end

            options = struct();
            if isfield(settings, kernel)
                options = settings.(kernel);
            end
        end

    end
end
