classdef MIS
    % Posterior simulation with generated samples
    % ----------------------------------------------------------------------
    % SYNTAX:
    % out = SimPos(obj,in)
    % ----------------------------------------------------------------------
    % INPUTS:
    % obj   : class constructed
    % in    : output of SuS
    % ----------------------------------------------------------------------
    % OUTPUTS:
    % out   : outputs
    %         --w_mis   : weight of mutiple importance sampling
    %         --w_prior : weight of prior distribution
    %         --z_i     : evidence estimation in every iteration
    %         --z       : estimated evidence with all generated samples
    %         --x_pos   : posterior samples
    % ----------------------------------------------------------------------
    % Reference:
    % [1] Handley, Will, and Pablo Lemos. "Quantifying dimensionality: Bayesian cosmological model complexities." Physical Review D 100.2 (2019): 023512.
    % ----------------------------------------------------------------------
    properties
        X
        Y
        G
        H
        C
        NumIte
        NumSam
        SamSze
        IntBay
        Wmis
        Wpos
        Wres
        Z
        Zi
    end

    methods
        function obj = MIS(in)
            % Initialization
            % ----------------------------------------------------------------------
            % SYNTAX:
            % obj = GSuS(Sampler)
            % ----------------------------------------------------------------------
            % INPUTS:
            % Sampler   : sampler to generate samples
            %             --IntL : Intermediate likelihood function
            %             --IntP : Intermediate prior distribution
            % ----------------------------------------------------------------------
            % OUTPUTS:
            % obj       : constructed class
            % ----------------------------------------------------------------------

            obj.X = in.x;
            obj.Y = in.y;
            obj.G = in.g;
            obj.NumIte = in.Nite;
            obj.NumSam = in.Nsam;
            obj.SamSze = in.Nsze;
            obj.IntBay = in.intBay;
            Nite = obj.NumIte;
            y = obj.Y;
            g = obj.G;
            intBay = obj.IntBay;

            % Normalize y
            % --------------------------------------------------------------------------
            w_res = cell(Nite,1); % weight of resampling
            c = zeros(Nite,1); % conditional probability
            for ite = 1:Nite
                w_res{ite} = ResRat(intBay,y{ite},g{ite+1});
                c(ite+1) = c(ite) + logmean(w_res{ite},"all");
                if ite<Nite
                    y{ite+1}(1,:) = y{ite+1}(1,:)-c(ite+1);
                end
            end
            obj.C = c;
            % --------------------------------------------------------------------------

            % Weight of samples balance heuristic
            w_mis = WgtMis_Blc(intBay,g,c,y);

            % Evaluate the evidence
            % --------------------------------------------------------------------------
            % weight of posterior distribution
            w_pos = WgtPos(y);
            % calculate the sub evidence
            z_i = MisInt(w_mis,w_pos);
            % calculate the global evidence
            z = logsum(z_i,"all");
            % --------------------------------------------------------------------------

            % Normalization of posterior weight
            % --------------------------------------------------------------------------
            for ite = 1:Nite
                w_pos{ite} = w_pos{ite}-z;
            end
            % --------------------------------------------------------------------------
            % save as properties
            obj.Wmis = w_mis;
            obj.Wpos = w_pos;
            obj.Wres = w_res;
            obj.Z = z;
            obj.Zi = z_i;
        end

        function [u_pos,y_pos,u_all,y_all] = PosSim(obj)
            x = obj.X;
            y = obj.Y;
            w_pos = obj.Wpos;
            w_mis = obj.Wmis;

            % flattern the samples
            [u_all,y_all] = VabCmb(x,y);

            % resampling to get posterior distribution
            [w_mis_flt,w_pos_flt] = VabCmb(w_mis,w_pos);
            w_pos_mis = w_mis_flt+w_pos_flt;
            Npos = round(exp(2*logsum(w_pos_mis,"all")-logsum(2*w_pos_mis,"all")));
            % random resampling
            Nall = size(y_all,2);
            w_pos_mis_norm = exp(w_pos_mis-max(w_pos_mis,[],"all"));
            ind_pos = randsample(1:Nall,Npos,true,w_pos_mis_norm);

            u_pos = u_all(:,ind_pos);
            y_pos = y_all(:,ind_pos);
        end

        function [ccdf,dcdf] = CCDF(obj,L)
            w_mis = obj.Wmis;
            y = obj.Y;
            z = obj.Z;
            Ndot = length(L);
            ccdf = zeros(1,Ndot);
            dcdf = zeros(1,Ndot);

            for idot = 1:Ndot
                % calculate CCDF
                w_ipri = IndPriWgt(y,L(idot));
                ccdf(idot) = logsum(MisInt(w_mis,w_ipri),"all");
                % calculate dCDF
                if idot==1
                    w_ipos = IndPosWgt(y,z,-inf,L(idot));
                else
                    w_ipos = IndPosWgt(y,z,L(idot-1),L(idot));
                end
                dcdf(idot) = logsum(MisInt(w_mis,w_ipos),"all");
            end
        end

        function [Ip,BMD] = KLDEst(obj)
            % evidence
            y = obj.Y;
            w_mis = obj.Wmis;
            z = obj.Z;
            [w_epy,w_epy2] = EpyWgt(y,z);
            % KL divergence
            Ip = exp(logsum(MisInt(w_mis,w_epy),"all"));
            Ip2 = exp(logsum(MisInt(w_mis,w_epy2),"all"));
            % Bayesian model dimensionality
            BMD = (Ip2-Ip^2)*2;
        end

        function Nunq = UnqSam(obj)
            Nite = obj.NumIte;
            X_sam = obj.X;
            Nunq = zeros(Nite,1);
            for ite = 1:Nite
                X_tmp = X_sam{ite}(1:end,:);
                X_unq = unique(X_tmp',"rows","stable");
                Nunq(ite) = size(X_unq,1);
            end
        end
    end
end