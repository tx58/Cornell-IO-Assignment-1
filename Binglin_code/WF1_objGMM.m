function [f] = WF1_objGMM(theta2, Cst, XMat, YMat, MuMat)
% This function calculates the GMM objective function by first inverting out xi;
% 

% Note: function calls inside this function:
% In step 0.0, call WF2_MeanVal to calculate the mean utility from delta
% given theta2.

%============
% code for debugging the function:
clearvars -except theta2 Cst XMat YMat MuMat HHMat InMat
%============

% Step 0: Preliminaries
% 0.0 First conditioning on one set of theta2, call WF2 to do contraction
% mapping in order to get CL_mvalold.
[CL_mvalold, CL_mu] = WF2_MeanVal(theta2, Cst, XMat, YMat, MuMat);

% 0.1 initialize the gmm obj
f = 0;

% 0.2 obtain coef
% not used here unlike MLE, only used in the inner loop

% 0.3 obtain constant
nmk=Cst.nmk;
nprods=Cst.nprods;
index_begin=Cst.index_begin;
index_end=Cst.index_end;
nprod_vec=Cst.nprod_vec;

% 0.4 obtain vehicle chars
price=YMat.price;
weight=XMat.weight;
hp=XMat.hp;
AC=XMat.AC;
ZMat=XMat.Z;

% 0.5 obtain HH chars
% no micro data

% 0.6 obtain ids - model id and HH choices of vehicles
% no micro data

% 0.7 slice product attributes and HHMat & initialize CL_indshr
% CL_YMat now stores sliced p
% the 1-4 cols of the cell array CL_XMat store the lgSize, safety, lgFC, CC respectively
% the 1-3 cols of the cell array CL_HHMat stores the fs, kids, hh_inc respectively
% CL_indshr is a nmk*1 cell vector with each cell storing a (nprod_m*nperson_m) matrix
% CL_hhmodel_id stores the choice model of each individual in each market:
% CL_model_id stores the model id for each product in each market
CL_YMat   = cell(nmk,1);
CL_XMat   = cell(nmk,3);
%CL_HHMat  = cell(nmk,1);
%CL_indshr = cell(nmk,1);
%CL_hhmodel_id = cell(nmk,1);
%CL_model_id   = cell(nmk,1);

for t = 1:nmk
    aa=index_begin(t);
    bb=index_end(t);
    CL_YMat{t}   = price(aa:bb);
    CL_XMat{t,1} = weight(aa:bb);
    CL_XMat{t,2} = hp(aa:bb);
    CL_XMat{t,3} = AC(aa:bb);
    CL_XMat{t,4} = ZMat(aa:bb,:);
    
end


% Step 1: Invert out residuals \xi
xi_vec=nan(nprods,1);    % initialize xi vector to store all stacked residuals

for m = 1:nmk
    
    delta = CL_mvalold{m};
    Z     = CL_XMat{m,4};
    X     = [CL_YMat{m} CL_XMat{m,1} CL_XMat{m,2} CL_XMat{m,3}];
    X     = [ones(size(X,1),1) X];
    
    beta_hat_IV_2sls = (X'*Z*(Z'*Z)^(-1)*Z'*X)\(X'*Z*(Z'*Z)^(-1)*Z'*delta);

    % Predicted values
    delta_hat_IV = Z*(Z'*Z)^(-1)*Z'*X*beta_hat_IV_2sls;

    % Residuals
    xi = delta_hat_IV - delta;
    
    % append to the residual vector
    aa=index_begin(m);
    bb=index_end(m);
    xi_vec(aa:bb)=xi;
    

end
% Step 2: Compute the moments and GMM obj
m = Z'*xi_vec;

f= m'*m;
