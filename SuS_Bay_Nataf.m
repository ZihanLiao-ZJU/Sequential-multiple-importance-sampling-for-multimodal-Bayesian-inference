classdef SuS_Bay_Nataf
    % Subset Simulation (SuS) with Nataf transformation for Bayesian updating
    %
    % Naming convention (consistent with SeMIS_Bay_Nataf):
    %   - logLi : log-domain HARD truncation term (indicator)
    %   - logPi : log prior density
    %   - G = [r_i, r_max]
    %       r_i   : adaptive threshold (level parameter)
    %       r_max : running maximum of log-likelihood

    properties (Constant)
        RunOptions = struct('InitializeProbability', true, 'PresetMode', 'overlay', 'MinimumLevels', 1, 'SeedFunction', 'SedSlt')
    end

    properties
        MethodProfile = 'author_latest'
        LKF            % Likelihood/prior wrapper
        Nfun = 4;      % Reserved for compatibility
        Ndim           % Parameter dimension
        Ncal           % #evaluations in the last EvlY call
        X              % Samples in U-space, size [Ndim, Nsam]
        Y              % Cached base terms associated with X
    end

    properties
        % Intermediate threshold parameters are stored in G.parameters = [r_i, r_max]
        G = struct('parameters', [-inf, -inf], 'lambda', 0.6);
    end

    properties(Dependent)
        FlgCvg         % Convergence flag
    end

    methods
        function obj = SuS_Bay_Nataf(LKF, method_profile)
            if nargin < 2 || isempty(method_profile)
                method_profile = 'author_latest';
            end
            obj.LKF  = LKF;
            obj.Ndim = LKF.Ndim;
            if ~any(strcmp(char(method_profile), {'author_latest', 'paper'}))
                error('method_sampler_profile:Name', 'Choose author_latest or paper.');
            end
            obj.MethodProfile = method_profile;
            obj.G.p = 0.1;
        end

        function logLi = EvlLKF(obj)
            % Evaluate HARD TRUNCATION term logLi(theta).
            %
            % logLi(theta) = log( I( logL(theta) > r_max + r_i ) )

            y     = obj.Y;
            logL  = y(1,:);

            r_i   = obj.G.parameters(1);
            r_max = obj.G.parameters(2);

            if isinf(r_i)
                logLi = zeros(size(logL));   % initial level: no truncation
            else
                Lth   = r_max + r_i;
                logLi = log(double(logL > Lth));
            end
        end

        function logPi = EvlPDF(obj)
            % Return prior log-density logPi(theta)
            logPi = obj.Y(2,:);
        end

        function [y,G,obj,N_cal] = UpdObj(obj,~,y,p)
            % Update G = [r_i, r_max] using standard SuS quantile rule.

            % Prescribed G.p remains the live conditional-probability setting.
            p = obj.G.p;
            logL = y(3,:);
            Nsam = size(logL,2);

            % sort log-likelihoods in descending order
            l_sorted = sort(logL,"descend");

            Nsed = round(Nsam * p);
            r_max = max(max(logL,[],"all"), obj.G.parameters(2));

            if Nsed > 0
                fth = (l_sorted(Nsed) + l_sorted(Nsed+1)) / 2;
            else
                fth = l_sorted(1);
            end

            r_i = fth - r_max;
            G   = [r_i, r_max];

            obj.G.parameters = G;
            G = obj.G;
            N_cal = 0;
        end

        function [u_sed, y_sed] = SltSed(obj,u,y,Nsam)
            % Select seeds satisfying the hard truncation condition.

            u = u(1:obj.Ndim,:);
            y = y(1:obj.Nfun,:);

            logL = y(3,:);
            r_i  = obj.G.parameters(1);
            r_max = obj.G.parameters(2);
            Lth  = r_max + r_i;

            Ns   = round(Nsam ./ (1:Nsam));
            Nsed = round(Nsam ./ Ns);

            % candidates satisfying hard constraint
            ind_sed = logL > Lth;
            u_can   = u(:,ind_sed);
            y_can   = y(:,ind_sed);
            Ncan    = sum(ind_sed,"all");

            Nsed = Nsed(find(Ncan - Nsed >= 0, 1, "last"));

            if Ncan >= Nsed
                randind = randperm(Ncan, Nsed);
            elseif Ncan > 0
                randind = randi(Ncan, 1, Nsed);
            else
                randind = [];
            end

            u_sed = u_can(:,randind);
            y_sed = y_can(:,randind);

            if ~isempty(y_sed)
                y_sed = obj.UpdY(y_sed);
            end
        end

        function FlgCvg = get.FlgCvg(obj)
            FlgCvg = obj.G.parameters(1) >= -1e-1;
        end

        function out = BayInf(obj,in)
            x    = in.x;
            y    = in.y;
            g    = in.g;
            Nite = in.Nite;

            % ---- normalize weights across levels ----
            w_res = cell(Nite,1);
            c     = zeros(Nite,1);

            for ite = 1:Nite-1
                w_res{ite} = ResRat(obj, y{ite}, g{ite+1});
                c(ite+1)  = c(ite) + logmean(w_res{ite},"all");
                y{ite+1}(1,:) = y{ite+1}(1,:) - c(ite+1);
            end

            % ---- evidence estimation ----
            zi = zeros(Nite,1);
            for ite = 1:Nite
                if ite < Nite
                    Li  = g{ite}.parameters(1)  + g{ite}.parameters(2);
                    Li1 = g{ite+1}.parameters(1)+ g{ite+1}.parameters(2);
                    w_z = min(logminus(y{ite}(3,:),Li), logminus(Li1,Li));
                else
                    Li  = g{ite}.parameters(1) + g{ite}.parameters(2);
                    w_z = logminus(y{ite}(3,:),Li);
                end
                zi(ite) = logmean(w_z,"all") + c(ite);
            end
            z = logsum(zi,"all");

            % ---- posterior resampling ----
            w_pos = cell(Nite,1);
            for ite = 1:Nite
                Nsze = size(y{ite});
                Nsze = Nsze(2:end);

                if ite < Nite
                    w_tmp = y{ite}(3,:) + y{ite}(4,:) ...
                          - y{ite}(1,:) - y{ite}(2,:) ...
                          + log(double(y{ite}(3,:) < g{ite+1}.parameters(1)+g{ite+1}.parameters(2)));
                else
                    w_tmp = y{ite}(3,:) + y{ite}(4,:) ...
                          - y{ite}(1,:) - y{ite}(2,:);
                end

                w_pos{ite} = zeros([Nsze,1]);
                w_pos{ite}(:) = w_tmp(:);
                w_pos{ite}(isnan(w_pos{ite})) = -inf;
            end

            [x_all,y_all] = VabCmb(x,y);
            w_pos = VabCmb(w_pos);
            w_pos = exp(w_pos - max(w_pos,[],"all"));

            Npos = round(sum(w_pos)^2 / sum(w_pos.^2));
            ind_pos = randsample(1:size(y_all,2), Npos, true, w_pos);

            out.z     = z;
            out.x_pos = x_all(:,ind_pos);
            out.y_pos = y_all(:,ind_pos);
        end

        function obj = EvlY(obj,u)
            % Evaluate base terms:
            %   obj.Y = [logL; logPi]

            x     = U2X(obj,u);
            logL  = obj.LKF.EvlLKF(x);
            logPi = logGauss(u);

            obj.X    = u;
            obj.Y    = [logL; logPi];
            obj.Ncal = size(u,2);
        end

        function y = UpdY(obj,y)
            % Prepend intermediate terms:
            %   y := [logLi; logPi; base_terms...]

            y = y(3:end,:);
            obj.Y = y;

            logPi = obj.EvlPDF;
            logLi = obj.EvlLKF;

            y = [logLi; logPi; y];
        end

        function x = U2X(obj,u)
            % Nataf transformation from U-space to X-space
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
