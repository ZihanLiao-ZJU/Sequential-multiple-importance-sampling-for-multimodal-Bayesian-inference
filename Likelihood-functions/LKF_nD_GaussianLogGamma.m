classdef LKF_nD_GaussianLogGamma
    % Gaussian + LogGamma
    % Configuration
    % ---------------------------------------------------------------------
    % 1. log evidence:
    %    logz = -8.19; % (Ndim = 2)
    %    logz = -20.47; % (Ndim = 5)
    %    logz = -40.94; % (Ndim = 10)
    %    logz = -81.887; % (Ndim = 20)
    % ---------------------------------------------------------------------
    % Reference:
    % ---------------------------------------------------------------------
    % [1] Beaujean, F., & Caldwell, A. (2013). “Initializing adaptive importance sampling with Markov chains.” arXiv preprint arXiv:1304.7808.
    % ---------------------------------------------------------------------

    properties
        % properties:
        % Name       Description                                      Type               Size
        % -----------------------------------------------------------------------------------
        % Ndim       number of dimensions                             double      [Ncha,Ndat]
        % Pdis       Prior distribution                               logical           [1,1]
        % -----------------------------------------------------------------------------------
        Ndim
        Pdis
        logz_ref
        PDF_marg
        CDF_marg
        X_marg
    end

    methods
        function obj = LKF_nD_GaussianLogGamma(Ndim)
            obj.Ndim = Ndim;
            obj.Pdis = Nataf(repmat(makedist("Uniform","lower",-30,"upper",30),Ndim,1),eye(Ndim));
            obj.logz_ref = -obj.Ndim*log(60);

            % marginal distribution
            dx_ref = 1e-3;
            x_min_ref = -30;
            x_max_ref =  30;
            x_ref = x_min_ref:dx_ref:x_max_ref;
            x_marg = repmat(x_ref,Ndim,1);
            N_marg = size(x_marg,2);
            pdf_marg = zeros(Ndim,N_marg);
            cdf_marg = zeros(Ndim,N_marg);

            beta_a=1;alpha_a=1;c_a=10;
            beta_b=1;alpha_b=1;c_b=-10;
            mu_c=10;sd_c=1;
            mu_d=-10;sd_d=1;
            beta_i=1;alpha_i=1;c_i=10;
            mu_i=10;sd_i=1;
            for idim = 1:Ndim
                if idim == 1
                    Log_g_a = alpha_a*(x_ref-c_a) - exp(x_ref-c_a)/beta_a - log(gamma(alpha_a)) - alpha_a*log(beta_a);
                    Log_g_b = alpha_b*(x_ref-c_b) - exp(x_ref-c_b)/beta_b - log(gamma(alpha_b)) - alpha_b*log(beta_b);
                    Log_g_ab = max(Log_g_a,Log_g_b);
                    logp = log(1/2) + Log_g_ab + log(exp(Log_g_a-Log_g_ab) + exp(Log_g_b-Log_g_ab));
                elseif idim == 2
                    Log_n_c = -1/2 * ((x_ref-mu_c)/sd_c).^2 -log(sqrt(2*pi)) - log(sd_c);
                    Log_n_d = -1/2 * ((x_ref-mu_d)/sd_d).^2 -log(sqrt(2*pi)) - log(sd_d);
                    Log_n_cd = max(Log_n_c,Log_n_d);
                    logp = log(1/2) + Log_n_cd + log(exp(Log_n_c-Log_n_cd) + exp(Log_n_d-Log_n_cd));
                elseif idim>=3 && idim<=(obj.Ndim+2)/2
                    logp = alpha_a*(x_ref-c_i) - exp(x_ref-c_i)/beta_a - log(gamma(alpha_i)) - alpha_a*log(beta_i);
                elseif idim>(obj.Ndim+2)/2
                    logp = -1/2 * ((x_ref-mu_i)/sd_i).^2 -log(sqrt(2*pi)) - log(sd_i);
                end
                pdf_marg(idim,:) = exp(logp);
                cdf_marg(idim,:) = cumsum(pdf_marg(idim,:).*dx_ref,2);
            end
            obj.PDF_marg = pdf_marg;
            obj.CDF_marg = cdf_marg;
            obj.X_marg = x_marg;
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

            % numebr of dimensions and samples
            Nsam = size(theta,2);
            L=zeros(1,Nsam);
            beta_a=1;alpha_a=1;c_a=10;
            beta_b=1;alpha_b=1;c_b=-10;
            mu_c=10;sd_c=1;
            mu_d=-10;sd_d=1;
            if obj.Ndim>=3
                Log_g=zeros(obj.Ndim,1);
                for j=1:Nsam
                    Log_g_a = alpha_a*(theta(1,j)-c_a) - exp(theta(1,j)-c_a)/beta_a - log(gamma(alpha_a)) - alpha_a*log(beta_a);
                    Log_g_b = alpha_b*(theta(1,j)-c_b) - exp(theta(1,j)-c_b)/beta_b - log(gamma(alpha_b)) - alpha_b*log(beta_b);
                    Log_g_ab = max(Log_g_a,Log_g_b);
                    Log_n_c = -1/2 * ((theta(2,j)-mu_c)/sd_c).^2 -log(sqrt(2*pi)) - log(sd_c);
                    Log_n_d = -1/2 * ((theta(2,j)-mu_d)/sd_d).^2 -log(sqrt(2*pi)) - log(sd_d);
                    Log_n_cd = max(Log_n_c,Log_n_d);
                    Log_g(1) = log(1/2) + Log_g_ab + log(exp(Log_g_a-Log_g_ab) + exp(Log_g_b-Log_g_ab));
                    Log_g(2) = log(1/2) + Log_n_cd + log(exp(Log_n_c-Log_n_cd) + exp(Log_n_d-Log_n_cd));

                    beta_i=1;alpha_i=1;c_i=10;
                    mu_i=10;sd_i=1;
                    for i=3:obj.Ndim
                        if i>=3 && i<=(obj.Ndim+2)/2
                            Log_g(i) = alpha_a*(theta(i,j)-c_i) - exp(theta(i,j)-c_i)/beta_a - log(gamma(alpha_i)) - alpha_a*log(beta_i);
                        elseif i>(obj.Ndim+2)/2
                            Log_g(i) = -1/2 * ((theta(i,j)-mu_i)/sd_i).^2 -log(sqrt(2*pi)) - log(sd_i);
                        end
                    end
                    L(j)= L(j)+ sum(Log_g);
                end
            else
                for j=1:Nsam
                    Log_g_a = alpha_a*(theta(1,j)-c_a) - exp(theta(1,j)-c_a)/beta_a - log(gamma(alpha_a)) - alpha_a*log(beta_a);
                    Log_g_b = alpha_b*(theta(1,j)-c_b) - exp(theta(1,j)-c_b)/beta_b - log(gamma(alpha_b)) - alpha_b*log(beta_b);
                    Log_g_ab = max(Log_g_a,Log_g_b);
                    Log_n_c = -1/2 * ((theta(2,j)-mu_c)/sd_c).^2 -log(sqrt(2*pi)) - log(sd_c);
                    Log_n_d = -1/2 * ((theta(2,j)-mu_d)/sd_d).^2 -log(sqrt(2*pi)) - log(sd_d);
                    Log_n_cd = max(Log_n_c,Log_n_d);
                    L(j) =  2 * log(1/2) + Log_g_ab + log(exp(Log_g_a-Log_g_ab) + exp(Log_g_b-Log_g_ab)) + Log_n_cd + log(exp(Log_n_c-Log_n_cd) + exp(Log_n_d-Log_n_cd));
                end
            end
        end

        function PltFun(obj)
            % plot the function
            % ----------------------------------------------------------------------
            % SYNTAX:
            % PltFun (obj)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj  : constructed class
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % ----------------------------------------------------------------------

            x_ref = -30:0.1:30;
            y_ref = -30:0.1:30;
            xmin = min(x_ref);
            xmax = max(x_ref);
            Nx = length(x_ref);
            ymin = min(y_ref);
            ymax = max(y_ref);
            Ny = length(y_ref);
            [xx,yy] = meshgrid(x_ref,y_ref);
            zz = zeros(Ny,Nx);
            % evlauation
            for iy = 1:Ny
                zz(iy,:) = obj.EvlLKF([xx(iy,:);yy(iy,:)]);
            end
            zmax = max(zz,[],"all");
            zmin = zmax-15;

            % figure plot
            figure
            % plot the likelihood surf
            surf(xx,yy,zz,"EdgeColor","none","FaceAlpha",1);
            hold on
            plot3(xx(1:Ny,:)',yy(1:Ny,:)',zz(1:Ny,:)',"Color","black","LineStyle","none","LineWidth",0.2)
            hold on
            plot3(xx(:,1:Nx),yy(:,1:Nx),zz(:,1:Nx),"Color","black","LineStyle","none","LineWidth",0.2)

            % figure settings
            xlim([xmin,xmax]);
            ylim([ymin,ymax]);
            zlim([zmin+1,zmax]);
            clim([zmin zmax]);
            xlabel('$\theta_1$','Interpreter','latex')
            ylabel('$\theta_2$','Interpreter','latex')
            zlabel('$\mathrm{ln}(\it{L})$','Interpreter','latex')
            set(gca,"Fontsize",20)

            figure
            % plot the likelihood cloud and contour
            contourf(xx,yy,zz,zmin:(zmax-zmin)/5:zmax);
            % figure settings
            colorbar
            grid on
            clim([zmin zmax]);
            xlim([xmin,xmax]);
            ylim([ymin,ymax]);
            xlabel('$\theta_1$','Interpreter','latex')
            ylabel('$\theta_2$','Interpreter','latex')
            set(gca,"Fontsize",20)
        end

        function ks = KStest(obj,x_pos,ind_dim)
            ks = zeros(length(ind_dim),1);
            for idim = ind_dim
                ind_idim = idim==ind_dim;
                % sample distribution
                [cdf_emp, x_emp] = ecdf(x_pos(idim,:));
                [x_emp, idx1] = unique(x_emp, 'last');   cdf_emp = cdf_emp(idx1);
                x_emp = x_emp.'; cdf_emp = cdf_emp.';
                % x_min_emp = min(x_emp);
                % x_max_emp = max(x_emp);

                % reference distribution
                x_ref = obj.X_marg(idim,:);
                cdf_ref = obj.CDF_marg(idim,:);

                % ks ststistic
                x_all = unique([x_emp,x_ref]);
                cdf_emp = interp1(x_emp, cdf_emp, x_all, 'previous', 'extrap');
                cdf_ref = interp1(x_ref, cdf_ref, x_all, 'previous', 'extrap');
                cdf_emp(isnan(cdf_emp)) = 0;cdf_ref(isnan(cdf_ref)) = 0;
                ks(ind_idim) = max(abs(cdf_emp-cdf_ref));

                % plot
                % figure
                % histogram(x_pos(idim,:),"Normalization","pdf");
                % hold on
                % plot(x_any,pdf_any);
                % figure
                % plot(x_all,cdf_emp);
                % hold on
                % plot(x_all,cdf_any);
                % xlim([x_min_emp,x_max_emp])
                % ylim([0,1]);
            end
        end
    end
end