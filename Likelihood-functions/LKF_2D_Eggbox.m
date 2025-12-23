classdef LKF_2D_Eggbox
    % 2D eggbox shape likelihood
    % Configuration
    % ---------------------------------------------------------------------
    % 1. Dimension: 2
    % 2. prior:
    %    marg_x = [makedist("Uniform","lower",0,"upper",10*pi);makedist("Uniform","lower",0,"upper",10*pi)];
    %    rhoL_x = eye(2);
    % 3. logz = 235.856; % (d = 2);
    % ---------------------------------------------------------------------
    % Reference:
    % ---------------------------------------------------------------------
    % [1] Feroz, F., Hobson, M. P., & Bridges, M. (2009). “MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics.” Monthly Notices of the Royal Astronomical Society, 398(4), 1601-1614.
    % ---------------------------------------------------------------------

    properties(Constant)
        % Fixed properties:
        % Name       Description                                    Type                 Size
        % -----------------------------------------------------------------------------------
        % LogL       output with logrithm                           logical             [1,1]
        % Ndim       number of dimensions                           double        [Ncha,Ndat]
        % -----------------------------------------------------------------------------------
        Ndim = 2;
        logz_ref = 235.856;
        Pdis = Nataf(repmat(makedist("Uniform","lower",0,"upper",10*pi),2,1),eye(2));
    end

    properties
        PDF_marg
        CDF_marg
        X_marg
    end

    methods
        function obj = LKF_2D_Eggbox
            % marginal distribution
            dx_ref = 1e-3;
            x_min_ref = 0;
            x_max_ref = 10*pi;
            x_ref = x_min_ref:dx_ref:x_max_ref;
            x_marg = repmat(x_ref,2,1);
            N_marg = size(x_marg,2);
            pdf_marg = zeros(2,N_marg);

            dx_itg = 5e-4;
            for idim = 1:2
                x_itg = x_min_ref:dx_itg:x_max_ref;
                N_itg = length(x_itg);
                if idim == 1
                    for imarg = 1:N_marg
                        x_ing = [x_marg(idim,imarg)*ones(1,N_itg);x_itg];
                        L = obj.EvlLKF(x_ing);
                        pdf_marg(idim,imarg) = exp(logsum(L+log(dx_itg)-2*log(10*pi),"all")-obj.logz_ref);
                    end
                elseif idim == 2
                    for imarg = 1:N_marg
                        x_ing = [x_itg;x_marg(idim,imarg)*ones(1,N_itg)];
                        L = obj.EvlLKF(x_ing);
                        pdf_marg(idim,imarg) = exp(logsum(L+log(dx_itg)-2*log(10*pi),"all")-obj.logz_ref);
                    end
                end
            end
            cdf_marg = cumsum(pdf_marg.*dx_ref,2);
            obj.PDF_marg = pdf_marg;
            obj.CDF_marg = cdf_marg;
            obj.X_marg = x_marg;
        end

        function L = EvlLKF(~,theta)
            % evaluate the function value
            % ----------------------------------------------------------------------
            % SYNTAX:
            % L = EvlPdf(obj,theta)
            % ----------------------------------------------------------------------
            % INPUTS:
            % obj  : class constructed
            % theta: variable samples                                    [Ndim,Nsam]
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % L    : function value
            % ----------------------------------------------------------------------

            L=(2+cos(theta(1,:)/2).*cos(theta(2,:)/2)).^5;
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

            x_ref = 0:0.1:10*pi;
            y_ref = 0:0.1:10*pi;
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
            zmin = min(zz,[],"all");

            % figure plot
            figure
            % plot the likelihood surf
            surf(xx,yy,zz,"EdgeColor","none","FaceAlpha",1);
            hold on
            plot3(xx(1:Ny,:)',yy(1:Ny,:)',zz(1:Ny,:)',"Color","black","LineStyle","none","LineWidth",0.2)
            hold on
            plot3(xx(:,1:Nx),yy(:,1:Nx),zz(:,1:Nx),"Color","black","LineStyle","none","LineWidth",0.2)
            hold on
            % figure settings
            xlim([xmin,xmax]);
            ylim([ymin,ymax]);
            zlim([zmin,zmax]);
            clim([zmin zmax]);
            xlabel('$\theta_1$','Interpreter','latex')
            ylabel('$\theta_2$','Interpreter','latex')
            zlabel('$\mathrm{ln}(\it{L})$','Interpreter','latex')
            set(gca,"Fontsize",20)

            figure
            % plot the likelihood cloud and contour
            contourf(xx,yy,zz,zmin:(zmax-zmin)/5:zmax);
            hold on
            % figure settings
            colorbar
            grid on
            xlim([xmin,xmax]);
            ylim([ymin,ymax]);
            clim([zmin zmax]);
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

                % reference distribution
                x_ref = obj.X_marg(idim,:);
                cdf_ref = obj.CDF_marg(idim,:);

                % ks ststistic
                x_all = unique([x_emp,x_ref]);
                cdf_emp = interp1(x_emp, cdf_emp, x_all, 'previous', 'extrap');
                cdf_ref = interp1(x_ref, cdf_ref, x_all, 'previous', 'extrap');
                cdf_emp(isnan(cdf_emp)) = 0;cdf_ref(isnan(cdf_ref)) = 0;
                ks(ind_idim) = max(abs(cdf_emp-cdf_ref));
            end
        end
    end
end