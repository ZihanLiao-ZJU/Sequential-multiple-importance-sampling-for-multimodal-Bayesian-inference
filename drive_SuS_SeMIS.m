clear;

%% Problem definition and target density
Ndim = 2;                                      % Parameter dimension
% Tfun = LKF_2D_Eggbox;                        % Eggbox benchmark
% Tfun = LKF_2D_5Gaussian;                     % Gaussian mixture benchmark
Tfun = LKF_nD_GaussianLogGamma(Ndim);          % Gaussian–LogGamma benchmark
logz_ref = Tfun.logz_ref;                     % Reference log-evidence

% Bayesian inference objects in Nataf space
IntBay_SuS   = SuS_Bay_Nataf(Tfun);            % SuS-based inference
IntBay_SeMIS = SeMIS_Bay_Nataf(Tfun);          % SeMIS-based inference

% Sequential Monte Carlo samplers
smc_SuS   = SeMCS(ES(IntBay_SuS));
smc_SeMIS = SeMCS(ES(IntBay_SeMIS));

% Sampling budget per level
smc_SeMIS.NumSam = 1700;
smc_SuS.NumSam   = 1000;

%% Monte Carlo experiment settings
Nrun = 100;                                   % Number of independent runs

% Log-evidence estimates
logz_SeMIS_MIS = zeros(Nrun,1);               % SeMIS + MIS
logz_SeMIS_SIS = zeros(Nrun,1);               % SeMIS + SIS
logz_SuS       = zeros(Nrun,1);               % SuS

% Computational cost and posterior size
Ncal_SeMIS = zeros(Nrun,1);                   % Likelihood evaluations
Ncal_SuS   = zeros(Nrun,1);
Npos_SeMIS = zeros(Nrun,1);                   % Posterior samples
Npos_SuS   = zeros(Nrun,1);

%% Main Monte Carlo loop
for irun = 1:Nrun

    % ----- SeMIS -----
    out_sm = smc_SeMIS.RunIte;                 % SMC iterations
    Ncal_SeMIS(irun) = sum(out_sm.Ncal);

    out_mis = MIS(out_sm);                     % MIS post-processing
    logz_SeMIS_MIS(irun) = out_mis.Z;          % MIS evidence

    [u_pos,~] = out_mis.PosSim;                % Posterior samples (U-space)
    Npos_SeMIS(irun) = size(u_pos,2);
    logz_SeMIS_SIS(irun) = out_mis.C(end) ...
                         + out_sm.g{end}(2);   % SIS evidence

    x_pos_SeMIS = Tfun.Pdis.U2X(u_pos);        % Map to physical space

    % ----- SuS -----
    out_sm = smc_SuS.RunIte;
    Ncal_SuS(irun) = sum(out_sm.Ncal);

    out_sus = IntBay_SuS.BayInf(out_sm);
    logz_SuS(irun) = out_sus.z;

    u_pos = out_sus.x_pos;
    Npos_SuS(irun) = size(u_pos,2);
    x_pos_SuS = Tfun.Pdis.U2X(u_pos);
end

%% Performance and efficiency statistics
% Relative bias and coefficient of variation of evidence
logz_SeMIS_MIS_bias = mean(logz_SeMIS_MIS/logz_ref - 1) * 100;
logz_SeMIS_MIS_cv   = std (logz_SeMIS_MIS/logz_ref - 1) * 100;

logz_SeMIS_SIS_bias = mean(logz_SeMIS_SIS/logz_ref - 1) * 100;
logz_SeMIS_SIS_cv   = std (logz_SeMIS_SIS/logz_ref - 1) * 100;

logz_SuS_bias = mean(logz_SuS/logz_ref - 1) * 100;
logz_SuS_cv   = std (logz_SuS/logz_ref - 1) * 100;

% Posterior efficiency and average cost
Npos_Ncal_SeMIS_mu = mean(Npos_SeMIS ./ Ncal_SeMIS) * 100;
Npos_Ncal_SuS_mu   = mean(Npos_SuS   ./ Ncal_SuS)   * 100;

Ncal_SeMIS_mu = mean(Ncal_SeMIS);
Ncal_SuS_mu   = mean(Ncal_SuS);
