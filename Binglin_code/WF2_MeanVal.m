function [CL_mvalold, CL_mu] = WF2_MeanVal(theta2, Cst, XMat, YMat, MuMat)
% This function computes the mean utility level delta implied by each
% theta2. It uses parallel processing and implements contraction mapping
% market by market. 
% Note: mvalold, mval, s_jt, mu are sliced by market for parfor
% Note: YMat.CL_mvalold stores mvalold from the previous iteration

% Note: function calls inside this function:
% In step 2, call WF3_Mu to calculate the nonlinear utility part (for
% pseudo hhs) given a set of theta2.
% In step 3, call WF4_IndShr 4 times (2 times each for parallel and
% non-parallel) to calculate the ind prob in order to do contraction
% mapping,

% ===================== %
% version 1.1
% Last update: May 1, 2018
% Changes:
% 1. call function WF3 with changed input argument.

% version 1.0
% Last update: Apr 29, 2018
% Changes:
% 1. Call WF3 function to calculate the individual market shares and change
%   the input values using Cst struct to summarize variables (line 75 & 120 bookmarked)
% 2. Call WF2 function to calculate the nonlinear part utility and
%   surpress return values for P_ij, price_coef (line 46 bookmarked)
%   also in the main return of WF1
% 3. inv? or /? matrix inversion
% ===================== %

%============
% code for debugging the function:
clearvars -except theta2 Cst XMat YMat MuMat
%============

% =====================
% Step 0: Use gradient-based inversion if Cst.gradInv==1
gradInv=Cst.gradInv;

% Step 0b: Use parfor if Cst.pfor==1
pfor=Cst.pfor;

% Step 1: Constants
% Obtain constants
nmk=Cst.nmk;
ndraws = Cst.ndraws;
maxiter=Cst.maxiter;
mvaltol=Cst.mvaltol;  % Use the same tolerance level throughout the mval inversion

index_begin=Cst.index_begin;
index_end=Cst.index_end;
nprod_vec=Cst.nprod_vec;

% CL_sjt (s_jt in a cell array - already sliced)
CL_sjt=YMat.CL_sjt;

% CL_mvalold: mvalold (in a cell array) from previous iteration
CL_mvalold=YMat.CL_mvalold;


% Step 2: mu_All: individual random utility; P_ij=alpha_i(Y_i)*p_j
mu_All = WF3_Mu(theta2, Cst, XMat, YMat, MuMat); 

% Slice mu market by market
CL_mu =cell(nmk,1);
for t = 1:nmk
    aa=index_begin(t);
    bb=index_end(t);
    CL_mu{t} = mu_All(aa:bb,:);
end;


% Step 3: mval inversion
if pfor==1
    parfor t=1:nmk
        avgnorm = 1;
        i = 0;
        nprod=nprod_vec(t); %nprod: number of product in mk t
        mvalold=CL_mvalold{t};
        s_jt=CL_sjt{t};
        mu=CL_mu{t};

        while i<20
            indshr = WF4_IndShr(mvalold, mu);
            mktshr = mean(indshr,2);
            mval = mvalold + (log(s_jt) - log(mktshr))*0.5;
            mvalold = mval;
            i = i + 1;
        end

        while i<maxiter && avgnorm > mvaltol
            indshr = WF4_IndShr(mvalold, mu);
            mktshr = mean(indshr,2);

            switch gradInv
                case 1  % Use gradient-based xi inversion to speed up convergence when niter is high
                    deriv=(diag(mktshr)-indshr*indshr'/ndraws)./(mktshr*ones(1,nprod));
                    mval=mvalold+0.5*inv(deriv)*(log(s_jt)-log(mktshr));
                case 0  % Use regular xi inversion
                    mval = mvalold + (log(s_jt) - log(mktshr));
                otherwise
                disp('gradInv needs to be defined and can only take value 0 or 1')
            end

            tt = abs(mval-mvalold);
            avgnorm = mean(tt);
            mvalold = mval;
            i = i + 1;
        end

        % Display a message when max num of iteration is exceeded
        if i>=maxiter
            disp(['contraction mapping over limit in Mk ' num2str(t) ' and avgnorm*10^6']);
            num2str(avgnorm*1000000)
        end
        CL_mvalold{t}=mval;
    end

elseif pfor==0
    for t=1:nmk
        avgnorm = 1;
        i = 0;
        nprod=nprod_vec(t); %nprod: number of product in mk t
        mvalold=CL_mvalold{t};
        s_jt=CL_sjt{t};
        mu=CL_mu{t};
 
        while i<20
            %indshr = WF3_IndShr_v1_0(Cst, mvalold, mu);
            indshr = WF4_IndShr(mvalold, mu);
            mktshr = mean(indshr,2);
            mval = mvalold + (log(s_jt) - log(mktshr))*0.5;
            mvalold = mval;
            i = i + 1;
        end

        while i<maxiter && avgnorm > mvaltol
            %indshr = WF3_IndShr_v1_0(Cst, mvalold, mu);
            indshr = WF4_IndShr(mvalold, mu);
            mktshr = mean(indshr,2);

            switch gradInv
                case 1  % Use gradient-based xi inversion to speed up convergence when niter is high
                    deriv=(diag(mktshr)-indshr*indshr'/ndraws)./(mktshr*ones(1,nprod));
                    mval=mvalold+0.5*inv(deriv)*(log(s_jt)-log(mktshr));
                    %mval=mvalold+0.5*eye(length(deriv))/deriv*(log(s_jt)-log(mktshr));
                    %mval=mvalold+0.5*deriv\(log(s_jt)-log(mktshr));
                case 0  % Use regular xi inversion
                    mval = mvalold + (log(s_jt) - log(mktshr));
                otherwise
                disp('gradInv needs to be defined and can only take value 0 or 1')
            end

            tt = abs(mval-mvalold);
            avgnorm = mean(tt);
            mvalold = mval;
            i = i + 1;
        end

        % Display a message when max num of iteration is exceeded
        if i>=maxiter
            disp(['contraction mapping over limit in Mk ' num2str(t) ' and avgnorm*10^6']);
            num2str(avgnorm*1000000)
        end
        CL_mvalold{t}=mval;
    end
end
