classdef ES
    % Elliptical slice sampling with Nataf transformation
    properties(Constant)
        % properties:
        % Name       Description                                      Type               Size
        % -----------------------------------------------------------------------------------
        % SedSav     Save the seeds or not (defalut:false)            logical           [1,1]
        % Ntry       Number of max shrink times (defalut:100)         double            [1,1]
        % -----------------------------------------------------------------------------------
        SedSav = false;
        Ntry = 20
    end

    properties
        % properties:
        % Name       Description                                      Type               Size
        % -----------------------------------------------------------------------------------
        % BayStc     A Bayesian structure                             /                 [1,1]
        % LogL       output with logrithm (likelihood)                logical           [1,1]
        % LogP       output with logrithm (prior)                     logical           [1,1]
        % Ndim       Number of dimensions                             double            [1,1]
        % Nfun       Number of functions                              double            [1,1]
        % -----------------------------------------------------------------------------------
        BayStc
        Ndim
        Nfun
    end

    methods
        function obj = ES(BayStc)
            % Initializatipon
            % ----------------------------------------------------------------------
            % SYNTAX:
            % obj = MCMC(Liklihood,Prior)
            % ----------------------------------------------------------------------
            % INPUTS:
            % Liklihood : Likelihood function
            % Prior     : Nataf based prior distribution
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % obj       : constructed class
            % ----------------------------------------------------------------------
            obj.BayStc = BayStc;
            obj.Nfun = BayStc.Nfun;
            obj.Ndim = BayStc.Ndim;
        end

        function [u,y,Ncal,baystc] = SamGen(obj,varargin)
            % Generate samples distributing as prior * likelihood
            % 1. With seeds, no "burn in"
            % 2. Without seeds, prior --> prior * likelihood
            % ----------------------------------------------------------------------
            % SYNTAX:
            % [x,y,Ncal] = SamGen(obj,Nc,Ns)
            % [x,y,Ncal] = SamGen(obj,x_sed,y_sed,Ns)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed
            % Ns    : length of Markov chains                                  [1,1]
            % Nc    : number of Markov chains                                  [1,1]
            % x_sed : initial samples                                      [Ndim,Nc]
            % y_sed : function list of x_sed                               [Nfun,Nc]
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % x     : generated samples                                 [Ndim,Nc,Ns]
            % y     : output function values                            [Nfun,Nc,Ns]
            %         --likelihood function L
            %         --...
            %         --prior distribution p
            %         --...
            % Ncal  : number of likelihood calls                               [1,1]
            % ----------------------------------------------------------------------

            % initialization
            baystc = obj.BayStc;
            sedsav = obj.SedSav || nargin<=3;
            nTry = obj.Ntry;
            nDim = obj.Ndim;
            nFun = obj.Nfun;
            if nargin<=3
                Nc = varargin{1};
                Ns = varargin{2};
                u_sed = randn(nDim,Nc);
                baystc = baystc.EvlY(u_sed);
                y_sed = [baystc.EvlLKF;baystc.EvlPDF;baystc.Y];
                Ncal = baystc.Ncal;
            else
                u_sed = varargin{1};
                y_sed = varargin{2};
                Ns = varargin{3};
                Nc = size(u_sed,2);
                Ncal = 0;
            end

            % allocate memory
            u = zeros(nDim,Nc,Ns);
            y = zeros(nFun,Nc,Ns);

            % generation of randn numbers
            du = randn(nDim,Nc,Ns);
            p = rand(1,Nc,Ns);

            for k=1:Ns
                if k == 1 && sedsav
                    u(:,:,1) = u_sed;
                    y(:,:,1) = y_sed;
                else
                    Rho = sqrt(u_sed.^2+du(:,:,k).^2); % [Ndim,Nc]
                    alpha_appro = acos(u_sed./Rho); % [Ndim,Nc]
                    ind_sam = 1:Nc;
                    Nrst = Nc;
                    alphamin = zeros(nDim,Nc);
                    alphamax = pi*ones(nDim,Nc);
                    for i=1:nTry
                        alpha = rand(nDim,Nrst).*(alphamax-alphamin)+alphamin;
                        u(:,ind_sam,k) = cos(alpha).*Rho(:,ind_sam);
                        baystc = baystc.EvlY(u(:,ind_sam,k));
                        y(:,ind_sam,k) = [baystc.EvlLKF;baystc.EvlPDF;baystc.Y];
                        Ncal = Ncal + baystc.Ncal;
                        ind_rej =  exp(y(1,ind_sam,k)-y_sed(1,ind_sam)) < p(1,ind_sam,k);
                        Nrst = sum(ind_rej);
                        if Nrst>0
                            ind_sam = ind_sam(ind_rej);
                            u(:,ind_sam,k) = u_sed(:,ind_sam);
                            y(:,ind_sam,k) = y_sed(:,ind_sam);
                            alpha = alpha(:,ind_rej);
                            alphamin = alphamin(:,ind_rej);
                            alphamax = alphamax(:,ind_rej);
                            ind_alpha = alpha>alpha_appro(:,ind_sam);
                            alphamin(~ind_alpha) = alpha(~ind_alpha);
                            alphamax(ind_alpha) = alpha(ind_alpha);
                        else
                            break
                        end
                    end
                    u_sed = u(:,:,k);
                    y_sed = y(:,:,k);
                end
            end
        end
    end
end