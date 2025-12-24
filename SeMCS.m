classdef SeMCS
    % Sequential Monte Carlo sampler for Bayesian sequential schemes
    %
    % Developers: Zihan Liao, Binbin Li, Hua-Ping Wan, Xiao He (Zhejiang University)
    % Init: Jan 2023 | Last modified: Apr 2025

    properties (Constant)
        MaxIte = 10000;   % Hard cap on iterations
    end

    properties
        IntBay           % Intermediate Bayesian state (updated each iteration)
        Sampler          % Sampling engine (requires .BayStc and .SamGen)
        ConPrb = 0.1;    % Target conditional probability (e.g., SuS level prob.)
        NumSam = 1000;   % Samples per iteration (scalar or per-iteration vector)
        G = [];          % Optional predefined intermediate parameters (cell/array)
        X = [];          % Optional storage hook (not used in RunIte output)
        Y = [];          % Optional storage hook (not used in RunIte output)
    end

    methods
        function obj = SeMCS(Sampler)
            % SeMCS(Sampler)
            % Sampler must provide:
            %   - Sampler.BayStc : initial IntBay state
            %   - Sampler.SamGen : sample generator interface
            if nargin < 1
                error('Sampler is required and must contain a valid BayStc field.');
            end
            obj.Sampler = Sampler;
            obj.IntBay  = Sampler.BayStc;
        end

        function out = RunIte(obj)
            % Run sequential iterations until convergence or termination.
            % Returns all per-iteration samples and bookkeeping.

            % Local handles (avoid repeated obj. dereference in loop)
            intBay   = obj.IntBay;
            sampler  = obj.Sampler;
            p        = obj.ConPrb;
            maxi     = obj.MaxIte;
            use_vecN = (numel(obj.NumSam) > 1);    % NumSam varies by iteration
            use_G    = ~isempty(obj.G);            % user-specified intermediate G

            % Preallocate outputs (trimmed at the end)
            Nsam = zeros(maxi,1);                  % total samples per iteration
            Nsze = zeros(maxi,2);                  % [#chains, samples/chain]
            Ncal = zeros(maxi,1);                  % model/likelihood evaluations
            x = cell(maxi,1);  y = cell(maxi,1);  g = cell(maxi,1);

            % Determine per-iteration sample budget
            if use_vecN
                Nite_sam = length(obj.NumSam);
                Nsam(1:Nite_sam) = obj.NumSam(:);
            else
                Nsam(:) = obj.NumSam;
            end
            Nsze(1,:) = [Nsam(1), 1];              % iteration 1: independent samples

            % Initialize intermediate parameter sequence g{ite}
            if use_G
                Nite_int = length(obj.G);
                g(1:Nite_int) = obj.G;
            else
                Nite_int = 0;                      % will be adaptively updated
                g{1} = intBay.G;                   % start from current IntBay.G
            end

            for ite = 1:maxi

                % 1) Generate samples (first iteration: from prior; later: from seeds)
                if all(Nsze(ite,:) > 0) && ~isnan(Nsze(ite,2))
                    if ite == 1
                        [x{ite}, y{ite}, Ncal(ite), intBay] = sampler.SamGen(Nsze(ite,1), Nsze(ite,2));
                    else
                        [x{ite}, y{ite}, Ncal(ite), intBay] = sampler.SamGen(x_sed, y_sed, Nsze(ite,2));
                    end
                end
                Nsam(ite) = prod(Nsze(ite,:));

                % 2) Update intermediate distribution parameter g
                if ite + 1 <= Nite_int
                    intBay.G = g{ite+1};            % use prescribed sequence
                else
                    [g{ite+1}, intBay] = intBay.UpdObj(y{ite}, p);  % adaptive update
                end
                sampler.BayStc = intBay;

                % 3) Select seeds and set chain configuration for next iteration
                [x_sed, y_sed] = SedSlt(intBay, x{ite}, y{ite}, Nsam(ite+1));
                Nsze(ite+1,1) = size(x_sed,2);      % number of chains
                Nsze(ite+1,2) = round(Nsam(ite+1) / Nsze(ite+1,1)); % samples per chain

                % 4) Console summary
                DspRst(ite, Nsam(ite), Nsze(ite,:), Ncal(ite));

                % 5) Termination conditions
                if Nsze(ite+1,1) <= 0 || ite + 1 == maxi
                    flg_cvg = false;
                    break
                elseif intBay.FlgCvg
                    flg_cvg = true;
                    break
                end
            end

            % Trim preallocated arrays to the actual number of iterations
            Nite = ite;
            [x, y, Nsam, Nsze, Ncal] = VabSrk(Nite, x, y, Nsam, Nsze, Ncal);
            g = VabSrk(Nite + 1, g);

            % Pack outputs
            out.x       = x;
            out.y       = y;
            out.g       = g;
            out.Nsam    = Nsam;
            out.Nsze    = Nsze;
            out.Ncal    = Ncal;
            out.Nite    = Nite;
            out.flg_cvg = flg_cvg;
            out.intBay  = intBay;
        end
    end
end
