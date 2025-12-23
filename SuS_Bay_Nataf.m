classdef SuS_Bay_Nataf
    % Subset simulation with Nataf transfomration for Bayesian updating

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
        % H          parameters in prior dsitribution                    double         [1,1]
        %            --constant
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
        function obj = SuS_Bay_Nataf(LKF)
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
            L = y(1,:);
            g = obj.G;
            gth = g(1);
            Lmax = g(2);
            if isinf(g(1))
                Li = zeros(size(L));
            else
                Lth = Lmax+gth;
                Li = log(double(L>Lth));
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
            %         --intermediate likelihood function Li (normalized)
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
            L = y(3,:);
            Nsam = size(L,2);
            l = sort(L,"descend");
            Nsed = round(Nsam*p);
            Lmax = max(max(L,[],"all"),obj.G(2));
            if Nsed>0
                fth = (l(Nsed)+l(Nsed+1))/2;
            else
                fth = l(1);
            end
            gth = fth-Lmax;
            g = [gth,Lmax];
            % -------------------------------

            % update obj
            % --------------------------------
            obj.G = g;
            % --------------------------------
        end

        function [u_sed,y_sed] = SltSed(obj,u,y,Nsam)
            % Update y values with updated obj
            % ----------------------------------------------------------------------
            % SYNTAX:
            % Li = UpdObj(obj,y)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj   : class constructed                                        [1,1]
            % y     : output function values                                [4,Nsam]
            %         --intermediate likelihood function Li (normalized)
            %         --intermediate prior PDF pi
            %         --likelihood function L
            %         --prior PDF p
            %         --target function f
            %         --reference PDF q
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % obj   : class updated                                            [1,1]
            % ----------------------------------------------------------------------

            u = u(1:obj.Ndim,:);
            y = y(1:obj.Nfun,:);
            L = y(3,:);
            g = obj.G;
            gth = g(1);
            Lmax = g(2);
            Lth = Lmax+gth;
            Ns = round(Nsam./(1:Nsam));
            Nsed = round(Nsam./Ns);
            % filter the candidates
            ind_sed = L>Lth;
            x_sed_can = u(:,ind_sed);
            y_sed_can = y(:,ind_sed);
            Nsed_can = sum(ind_sed,"all");
            Nsed = Nsed(find(Nsed_can-Nsed>=0,1,"last"));
            % generate seeds
            if Nsed_can>=Nsed
                randind = randperm(Nsed_can,Nsed);
            elseif Nsed_can>0
                randind = randi(Nsed_can,1,Nsed);
            else
                randind = [];
            end
            u_sed = x_sed_can(:,randind);
            y_sed = y_sed_can(:,randind);
            if ~isempty(y_sed)
                y_sed = obj.UpdY(y_sed);
            end
        end

        function FlgCvg = get.FlgCvg(obj)
            FlgCvg = obj.G(1)>=-1e-1;
        end

        function out = BayInf(obj,in)
            x = in.x;
            y = in.y;
            g = in.g;
            Nite = in.Nite;

            % Normalize y
            % --------------------------------------------------------------------------
            w_res = cell(Nite,1);
            c = zeros(Nite,1);
            for ite = 1:Nite-1
                w_res{ite} = ResRat(obj,y{ite},g{ite+1});
                c(ite+1) = c(ite) + logmean(w_res{ite},"all");
                y{ite+1}(1,:) = y{ite+1}(1,:)-c(ite+1);
            end
            % --------------------------------------------------------------------------

            % Evaluate Evidence
            % --------------------------------------------------------------------------
            zi = zeros(Nite,1);
            for ite = 1:Nite
                if ite<Nite
                    Li = g{ite}(1)+g{ite}(2);
                    Li1 = g{ite+1}(1)+g{ite+1}(2);
                    w_z = min(logminus(y{ite}(3,:),Li),logminus(Li1,Li));

                else
                    Li = g{ite}(1)+g{ite}(2);
                    w_z = logminus(y{ite}(3,:),Li);
                end
                zi(ite) = logmean(w_z,"all")+c(ite);
            end
            z = logsum(zi,"all");
            % --------------------------------------------------------------------------

            % Posterior simulation
            % --------------------------------------------------------------------------
            w_pos = cell(Nite,1);
            for ite = 1:Nite
                Nsze = size(y{ite});
                Nsze = Nsze(2:end);
                if ite<Nite
                    w_pos_tep = y{ite}(3,:) + y{ite}(4,:) - y{ite}(1,:) - y{ite}(2,:) + log(double(y{ite}(3,:)<g{ite+1}(1)+g{ite+1}(2)));
                else
                    w_pos_tep = y{ite}(3,:) + y{ite}(4,:) - y{ite}(1,:) - y{ite}(2,:);
                end
                w_pos{ite} = zeros([Nsze,1]);
                w_pos{ite}(:) = w_pos_tep(:);
                w_pos{ite}(isnan(w_pos{ite})) = -inf;
            end
            % flattern the samples
            [x_all,y_all] = VabCmb(x,y);
            % resampling to get posterior distribution
            w_pos = VabCmb(w_pos);
            w_pos = exp(w_pos-max(w_pos,[],"all"));
            Npos = round(sum(w_pos)^2/sum(w_pos.^2));
            % random resampling
            Nall = size(y_all,2);
            ind_pos = randsample(1:Nall,Npos,true,w_pos);
            x_pos = x_all(:,ind_pos);
            y_pos = y_all(:,ind_pos);
            % --------------------------------------------------------------------------

            % output
            % --------------------------------------------------------------------------
            out.z = z;
            out.x_pos = x_pos;
            out.y_pos = y_pos;
            % --------------------------------------------------------------------------
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