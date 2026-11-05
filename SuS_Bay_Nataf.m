classdef SuS_Bay_Nataf
    % Subset Simulation (SuS) with Nataf transformation for Bayesian updating
    %
    % Naming convention (consistent with SeMIS_Bay_Nataf):
    %   - logLi : log-domain HARD truncation term (indicator)
    %   - logPi : log prior density
    %   - G = [r_i, r_max]
    %       r_i   : adaptive threshold (level parameter)
    %       r_max : running maximum of log-likelihood

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
        function spec = sampler_profile(~,family,name)
            spec=method_sampler_profile(family,name);
        end
        function obj = SuS_Bay_Nataf(LKF, method_profile)
            if nargin < 2 || isempty(method_profile), method_profile='author_latest'; end
            obj.LKF  = LKF;
            obj.Ndim = LKF.Ndim;
            obj.MethodProfile=method_profile;method_sampler_profile('sus_bayes',method_profile);obj.G.p=0.1;
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

        function [G,obj] = UpdObj(obj,y,p)
            % Update G = [r_i, r_max] using standard SuS quantile rule.

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
            spec = method_sampler_profile('sus_bayes', obj.MethodProfile);
            options = struct();
            if isfield(spec.options, kernel)
                options = spec.options.(kernel);
            end
        end

        function defaults = run_defaults(obj)
            defaults = struct('NumSam', 1000, 'ConPrb', 0.1, 'MaxIte', 10000, 'Verbose', true);
            if isprop(obj, 'Nsam')
                defaults.NumSam = obj.Nsam;
            end

            if isfield(obj.G, 'p')
                defaults.ConPrb = obj.G.p;
            end
        end

        function context = start_run(~, owner, sampler)
            owner.IntBay.G.p = owner.ConPrb;
            context = SeMCS.start_context(owner, sampler, max(1, owner.MaxIte - 1), 'overlay');
        end

        function context = advance_level(~, context, record, level)
            if level + 1 <= numel(context.owner.G)
                context.model.G = SeMCS.prescribed_state(context.model.G, context.owner.G{level+1}, false, 'overlay');
                extra = 0;
            else
                [~, context.model] = context.model.UpdObj(record.y, context.model.G.p);
                extra = 0;
            end

            context.Ncal(level) = context.Ncal(level) + extra;
            context.g{level+1, 1} = context.model.G;
            context.sampler.BayStc = context.model;
        end

        function context = select_run_seeds(~, context, record, level)
            if context.targets(level + 1) <= 0
                error('SeMCS:ScheduleExhausted', 'Sample budget exhausted before seed selection.');
            end

            target = context.targets(level + 1);
            [context.xs, context.ys] = SedSlt(context.model, record.x, context.y{level}, target);
            context.nc = size(context.xs, 2);
            context.steps = round(target / context.nc);
            if context.owner.Verbose
                fprintf('Level %d: N=%d, chains=%d, steps=%d, calls=%d\n', level, record.count, record.shape(1), record.shape(2), context.Ncal(level));
            end
        end

        function [yes, context] = run_complete(~, context, ~, phase)
            yes = false;
            if ~strcmp(phase, 'after_seeds')
                return;
            end

            if context.nc <= 0 || numel(context.x) == context.limit
                yes = true;
                return;
            end

            context.complete = context.model.FlgCvg;
            yes = context.complete;
        end

        function out = finish_run(~, context)
            out = SeMCS.pack_output(context);
        end

    end
end

function spec = method_sampler_profile(family, name)
    % Sampling settings for this method; thresholds, budgets and stopping are unchanged.
    if nargin < 2 || isempty(name)
        name = 'author_latest';
    end

    family = char(family);
    name = char(name);
    if ~any(strcmp(name, {'author_latest', 'paper'}))
        error('method_sampler_profile:Name', 'Choose author_latest or paper.');
    end

    if ~strcmp(family, 'sus_bayes')
        error('method_sampler_profile:Family', 'Unsupported method family for this model.');
    end

    options = struct();
    options.aCS = struct('Sigma0Mode', 'seed_std', 'CarryLambda', false);
    options.ES = struct('ESMode', 'coordinate', 'Ntry', 20);
    options.aCS.AcceptArithmetic = 'log';
    options.aCS.CoordinateMode = 'identity';
    options.aCS.GroupMode = 'available';
    options.aCS.AcceptanceRateMode = 'attempts';
    options.ES.AcceptArithmetic = 'log';
    options.ES.CoordinateMode = 'identity';
    options.ES.SeedUpdateMode = 'each_step';
    options.CW_aMH = struct('AcceptArithmetic', 'log', 'PriorMode', 'gaussian', 'CoordinateMode', 'identity');
    if strcmp(name, 'author_latest')
        options.aCS.AcceptArithmetic = 'exp';
        options.ES.AcceptArithmetic = 'exp';
        options.CW_aMH = struct('AcceptArithmetic', 'exp');
        options.aCS.GroupMode = 'fixed';
    end

    spec = struct('family', family, 'name', name, 'default_sampler', 'aCS', 'options', options);
end
