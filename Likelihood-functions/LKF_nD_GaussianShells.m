classdef LKF_nD_GaussianShells
    % two Gaussian Shells
    % Configuration
    % ---------------------------------------------------------------------
    % 1. log evidence:
    %    logz = -1.75; % (Ndim = 2)
    %    logz = -5.67; % (Ndim = 5)
    %    logz = -14.59; % (Ndim = 10)
    %    logz = -36.09; % (Ndim = 20)
    %    logz = -60.13; % (Ndim = 30)
    %    logz = -112.42; % (Ndim = 50)
    %    logz = -168.16; % (Ndim = 70)
    %    logz = -255.62; % (Ndim = 100)
    % ---------------------------------------------------------------------
    % Reference:
    % ---------------------------------------------------------------------
    % [1] Feroz F, Hobson M P. Multimodal nested sampling: an efficient and robust alternative to Markov Chain Monte Carlo methods for astronomical data analyses[J]. Monthly Notices of the Royal Astronomical Society, 2008, 384(2): 449-463.
    % ---------------------------------------------------------------------

    properties(Constant)
        % Fixed properties:
        % Name       Description                                    Type                 Size
        % -----------------------------------------------------------------------------------
        % LogL       output with logrithm                           logical             [1,1]
        % Nfun       length of function list                        double              [1,1]
        % -----------------------------------------------------------------------------------
        Nfun = 1;
        LogL = true;
    end

    properties
        % properties:
        % Name       Description                                      Type               Size
        % -----------------------------------------------------------------------------------
        % Ndim       number of dimensions                             double      [Ncha,Ndat]
        % Pdis       Prior distribution                               logical           [1,1]
        % -----------------------------------------------------------------------------------
        Ndim
        Pdis
    end

    methods
        function obj = LKF_nD_GaussianShells(Ndim)
            obj.Ndim = Ndim;
            obj.Pdis = Nataf(repmat(makedist("Uniform","lower",-6,"upper",6),Ndim,1),eye(Ndim));
        end

        function L = EvlLKF(obj,theta)
            % evaluate the function value
            % ----------------------------------------------------------------------
            % SYNTAX:
            % L = EvlFun(obj,theta)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj  : class constructed
            % theta: variable samples                                    [Ndim,Nsam]
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % L    : function value
            % ----------------------------------------------------------------------
            % REFERENCES:
            % ----------------------------------------------------------------------
            % [1].
            % ----------------------------------------------------------------------

            ndim = obj.Ndim;    % dimension
            c=zeros(ndim,2);
            c(1,1)=3.5;
            c(1,2)=-3.5;
            r=2;
            omega=0.1;
            X=-1*(sqrt(sum((theta-c(:,1)).^2,1))-r).^2./2./omega^2;
            Y=-1*(sqrt(sum((theta-c(:,2)).^2,1))-r).^2./2./omega^2;
            XY=max([X;Y],[],1);
            L = log(1/sqrt(2*pi*omega^2)) + XY + log(exp(X-XY)+exp(Y-XY));
        end

        function [theta_mle,L_mle] = MLE(obj)
            % Give the maximum likelihood estimation
            % ----------------------------------------------------------------------
            % SYNTAX:
            % L = MLE (obj,theta)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj       : constructed class
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % theta_mle : mle of likelihood function
            % L_mle     : corresponding likelihood value of theta_mle
            % ----------------------------------------------------------------------
            
            theta_mle = [];
            L_mle = obj.EvlLKF(theta_mle);
        end
    end
end