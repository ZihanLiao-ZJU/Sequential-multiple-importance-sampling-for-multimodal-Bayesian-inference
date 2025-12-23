clear;
% target function
% ----------------------------------------------------------------------------------
Ndim = 2;
% Tfun = LKF_2D_Eggbox; 
% Tfun = LKF_2D_5Gaussian;
Tfun = LKF_nD_GaussianLogGamma(Ndim);
logz_ref = Tfun.logz_ref;
% ----------------------------------------------------------------------------------

% select and initialize intermediate likelihood     function and prior distribution
% ----------------------------------------------------------------------------------
IntBay_SuS = SuS_Bay_Nataf(Tfun);
IntBay_SeMIS = SeMIS_Bay_Nataf(Tfun);
% ----------------------------------------------------------------------------------

% sequential MCS
smc_SuS = SeMCS(ES(IntBay_SuS));
smc_SeMIS = SeMCS(ES(IntBay_SeMIS));
% settings of SeMIS
smc_SeMIS.NumSam = 1700;
smc_SuS.NumSam = 1000;
%% run
Nrun = 100;

logz_SeMIS_MIS = zeros(Nrun,1);
logz_SeMIS_SIS = zeros(Nrun,1);
logz_SuS = zeros(Nrun,1);

Ncal_SeMIS = zeros(Nrun,1);
Ncal_SuS = zeros(Nrun,1);

Npos_SeMIS = zeros(Nrun,1);
Npos_SuS = zeros(Nrun,1);

% idim_ks = [1:2,3,11,12,20];
idim_ks = 1:2;
% idim_ks = 1:5;
Ndim_ks = length(idim_ks);
ks_SeMIS = zeros(Ndim_ks,Nrun);
ks_SuS = zeros(Ndim_ks,Nrun);
for irun=1:Nrun
    % Sequential Multiple importance sampling
    out_sm = smc_SeMIS.RunIte;
    Ncal_SeMIS(irun) = sum(out_sm.Ncal);
    out_mis = MIS(out_sm);
    logz_SeMIS_MIS(irun) = out_mis.Z;
    [u_pos,~] = out_mis.PosSim;% posterior samples
    Npos_SeMIS(irun) = size(u_pos,2);
    logz_SeMIS_SIS(irun) = out_mis.C(end)+out_sm.g{end}(2);
    x_pos_SeMIS =  Tfun.Pdis.U2X(u_pos);

    % SuS
    out_sm = smc_SuS.RunIte;
    Ncal_SuS(irun) = sum(out_sm.Ncal);
    out_sus = IntBay_SuS.BayInf(out_sm);
    logz_SuS(irun) = out_sus.z;
    u_pos = out_sus.x_pos;
    Npos_SuS(irun) = size(u_pos,2);
    x_pos_SuS =  Tfun.Pdis.U2X(u_pos);
end
logz_SeMIS_MIS_bias = mean(logz_SeMIS_MIS/logz_ref-1)*100;
logz_SeMIS_MIS_cv = std(logz_SeMIS_MIS/logz_ref-1)*100;
logz_SeMIS_SIS_bias = mean(logz_SeMIS_SIS/logz_ref-1)*100;
logz_SeMIS_SIS_cv = std(logz_SeMIS_SIS/logz_ref-1)*100;
logz_SuS_bias = mean(logz_SuS/logz_ref-1)*100;
logz_SuS_cv = std(logz_SuS/logz_ref-1)*100;

Npos_Ncal_SeMIS_mu = mean(Npos_SeMIS./Ncal_SeMIS)*100;
Npos_Ncal_SuS_mu = mean(Npos_SuS./Ncal_SuS)*100;

Ncal_SeMIS_mu = mean(Ncal_SeMIS);
Ncal_SuS_mu = mean(Ncal_SuS);