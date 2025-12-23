classdef SeMIS_Bay_Nataf
    % Sequential Multiple importance sampling with Nataf transfomration for Bayesian updating

    properties
        % Name       Description                                       Type              Size
        % -----------------------------------------------------------------------------------
        % LKF        Likelihood function                              /                [1,1]
        % Nfun       Length of function list                           double           [1,1]
        % Ndim       Number of dimension of parameters                 double           [1,1]
        % X          Samples in standard normal space                  double     [Ndim,Nsam]
        % Y          Y value of U                                      doubl         [4,Nsam]
        % -----------------------------------------------------------------------------------
        LKF
        Nfun = 4;
        Ndim
        Ncal
        X
        Y
    end

    properties
        % Properties to be updated:
        % Name       Description                                         Type            Size
        % -----------------------------------------------------------------------------------
        % G          parameters in likelihood function                   double         [1,2]
        %            --threshold of LKF
        %            --maximum value of LKF
        % -----------------------------------------------------------------------------------
        G = [-inf,-inf];
    end

    properties(Dependent)
        % Dependent properties:
        % Name        Description                                        Type            Size
        % -----------------------------------------------------------------------------------
        % FlgCvg      flag of convergence                                logical        [1,1]
        % -----------------------------------------------------------------------------------
        FlgCvg
    end

    methods
        function obj = SeMIS_Bay_Nataf(LKF)
            obj.LKF = LKF;
            obj.Ndim = LKF.Ndim;
        end

        function Li = EvlLKF(obj)
            % Evaluate the Intermediate likelihood Li
            % ----------------------------------------------------------------------
            % SYNTAX:
            % Li = EvlLi(obj,f)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed                                        [1,1]
            % y     : function list of samples                              [4,Nsam]
            %         --likelihood function L
            %         --prior PDF P
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % Li    : intermediate likelihood function Li                   [1,Nsam]
            % ----------------------------------------------------------------------
            y = obj.Y;
            f = y(1,:);
            g = obj.G;
            gth = g(1);
            fmax = g(2);
            if isinf(g(1))
                Li = zeros(size(f));
            else
                Li = min(f-fmax-gth,0);
            end
        end

        function pi = EvlPDF(obj)
            % Evaluate the Intermediate prior PDF pi
            % ----------------------------------------------------------------------
            % SYNTAX:
            % Li = EvlLi(obj,f)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed                                        [1,1]
            % y     : function list of samples                              [4,Nsam]
            %         --likelihood function L
            %         --prior PDF P
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % pi    : intermediate likelihood function pi                   [1,Nsam]
            % ----------------------------------------------------------------------
            y = obj.Y;
            p = y(2,:);
            pi = p;
        end

        function [g,obj] = UpdObj(obj,y,p)
            % Update obj with Y
            % ----------------------------------------------------------------------
            % SYNTAX:
            % [g,h,obj] = UpdObj(obj,y)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed                                        [1,1]
            % y     : output function values                                [4,Nsam]
            %         --intermediate likelihood function Li
            %         --intermediate prior PDF pi
            %         --likelihood function L
            %         --prior PDF p
            % p     : conditional probability in SuS
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % g     : updated parameters for intermediate likelihood function
            % h     : updated parameters for intermediate prior distribution
            % obj   : class updated
            % ----------------------------------------------------------------------

            % update g
            % --------------------------------
            f = y(3,:);
            g = obj.G;
            gth = g(1);
            fmax = g(2);
            fmax = max(max(f,[],"all"),fmax);

            dfmax=fmax-g(2);

            maxi = 100;
            err_min = inf;
            fun = @(x) abs(logmean(ResRat(obj,y,[x,fmax]),"all")-log(p));
            gmin = max(-1e12,gth-dfmax);

            gmax = 0;
            gbnd = (gmax-gmin)/maxi;
            for i = 1:maxi
                [g_tmp,err] = fminbnd(fun,gmin+(i-1)*gbnd,gmin+i*gbnd);
                if err<err_min
                    err_min = err;
                    gth = g_tmp;
                end
            end
            g = [gth,fmax];
            % -------------------------------

            % update obj
            % --------------------------------
            obj.G = g;
            % --------------------------------
        end

        function FlgCvg = get.FlgCvg(obj)
            FlgCvg = obj.G(1)>=-1e-4; %original
        end

        function obj = EvlY(obj,u)
            % evaluate the Y values
            % ----------------------------------------------------------------------
            % SYNTAX:
            % y = EvlLKF(obj,theta)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed
            % u     : Samples in standard normal space                   [Ndim,Nsam]
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % obj   : class with updated Y
            % y     : Y value of X                                     [Nfun+2,Nsam]
            %         --likelihood function L
            %         --prior PDF P
            % ----------------------------------------------------------------------
            x = U2X(obj,u);
            L = obj.LKF.EvlLKF(x);
            P = logGauss(u);
            obj.X = u;
            obj.Y = [L;P];
            obj.Ncal = size(u,2);
        end

        function y = UpdY(obj,y)
            % evaluate the likelihood function values
            % ----------------------------------------------------------------------
            % SYNTAX:
            % y = EvlLKF(obj,theta)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed                                        [1,1]
            % x     : variable samples                                   [Ndim,Nsam]
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % y     : output function values                           [Nfun+4,Nsam]
            %         --intermediate likelihood function Li
            %         --intermediate prior PDF pi
            %         --likelihood function L
            %         --prior PDF p
            % ----------------------------------------------------------------------
            % extraction the y except for Li and pi
            y = y(3:end,:);
            obj.Y = y;
            Pi = obj.EvlPDF;
            Li = obj.EvlLKF;
            y = [Li;Pi;y];
        end

        function x = U2X(obj,u)
            % Nataf situition
            x = obj.LKF.Pdis.U2X(u);
        end
    end
end