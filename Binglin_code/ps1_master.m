%%
%  IO PS#1
%

clc;
clear all;
cd '/Users/air/Dropbox/Study Materials Backup/Cornell Spring 2021/ECON7520/Assignments'

%% 0. Miscellaneous
data_raw = dataset('file','./PSet_BLP_data_no_header.txt', 'Delimiter','\t',...
    'ReadVarNames',false);
J = size(data_raw,1);   % # of models

data=data_raw;
data.id = (1:J)';      % raw index of models
data = sortrows(data,1);    % sort prices
price = data.Var1/1000;
quantity = data.Var2/1000;
weight = data.Var3/100;
hp = data.Var4/10;
AC = data.Var5;
firm = data.Var6;
id = data.id;
d_firm= dummyvar(firm);

M = 100*10^6/1000;   % market size
s = quantity/M; % market shares
s0 = 1-sum(s);  % outside share


%% 1. Vertical Model

% 1.1 inversion
lambda = 4*10^(-6);
delta = zeros(J,1);
delta(1) = -price(1)/lambda*log(s0);
for j=2:J
   delta(j) = delta(j-1) - (price(j)-price(j-1))/lambda*log(s0+sum(s(1:(j-1))));
end

% 1.2 estimation with OLS
XC = [ones(J,1) weight hp AC];
[result_ols.beta,result_ols.CI]= regress(delta, XC);
display('OLS estimates of beta coefficients')
result_ols.beta
display('OLS estimates of beta CI')
result_ols.CI


%% 3. BLP with only one nonlinear parameter (income)

% 3.0 preliminaries
% Set the seed for the random number generator
% 'rng' is Matlab's build-in function to generate random numbers
rng(12345,'twister');


% Constants
npv=1; % umber of provinces
nyr=1;    %
nmk=npv*nyr; % Total number of markets

% Draw 1000 individual per mk. Can use fewer draws when necessary
ndraws=1000; % number of pseudo households in each market

% 3.1 main data (macro BLP)
            %year = double(data.year);
nprods=J;   %nprods = length(year); % total number of model-by-market-year observations
            %prv_idx = double(data.mkt_idx); % Index 1-25 for each province. 1 is Beijing
mkt_idx=ones(J,1);    %mkt_idx = (year-2009)*npv+prv_idx;  %Index 1-175 for yr/province

data=data_raw;
data.id = (1:J)';      % raw index of models

% Car attributes
price = data.Var1/1000;
quantity = data.Var2;
weight = data.Var3;
hp = data.Var4;
AC = data.Var5;
firm = data.Var6;
id = data.id;
d_firm= dummyvar(firm);

% Market shares
M = 100*10^6;   % market size
s = quantity/M; % market shares
s0 = 1-sum(s);  % outside share

% construct instruments (code directly borrowed from Tianli): 
X = [weight hp AC];
% (1) average product characters produced by the same companies
Z1(J,3)=0;
Z2(J,3)=0;
for i=1:J
    Z1(i,:)= mean( X(firm==firm(i),:), 1);
end
% (2) average product characters produced by the rivals
for i=1:J
    Z2(i,:)= mean( X(firm~=firm(i),:), 1);
end
% (3) optimal instrument by JF Houde
for i=1:J
    Z3(i,:)= sum(abs( ones(J,1)*X(i,:)-X ));
end

Z=[Z1 Z2 Z3];

% First and last indices for each market
index_begin=zeros(nmk, 1);
index_end=zeros(nmk,1);
nprod_vec=zeros(nmk,1);

for m = 1:nmk
    ttt=find(mkt_idx == m);
    index_begin(m)=ttt(1);
    index_end(m)=ttt(end);
    nprod_vec(m)=length(ttt);
end

% 3.2 prepare pseudo draws
IncomeMat = nan(nmk,ndraws);
IncomeMat = lognrnd(35000/1000,45000/1000,[1,ndraws]);
% In each market, each household has the same draw for all products
nprods=J*nmk;
IncomeMat_all= zeros(nprods, ndraws);
IncomeMat_all= ones(nprods,1)*IncomeMat;

% 3.3 Initialize theta and collect variables
%  First Column: alpha1 (alpha bar) and alpha2 (power on income); beta1
%               beta2 (interaction btw fs==2/3 and logSize)
%  Second Column: gamma1 (interaction btw kids==1 and Safety); gamma2
%               gamma3 (interaction btw fs==2/3 and Safety)
%  Third Column: theta1 (interaction btw logInc and logFuelecon); theta2
%               (interaction btw logInc and Comf+Conv Features)

theta2w = [3.10];   % now only the coefficient on income and price interaction
       
% Vector form of theta2
[theti, thetj, theta2] = find(theta2w);

% collect constants, variables, and matrices relevant for estimation
% Create structure arrays Cst, YMat, XMat, MuMat, MicroMom
Cst.nmk = nmk; % Total number of markets
Cst.ndraws = ndraws; % number of pseudo households in each market
Cst.nprods = nprods; % total number of model-by-market-year observations
Cst.theti=theti;
Cst.thetj=thetj;

Cst.index_begin=index_begin;
Cst.index_end=index_end;
Cst.nprod_vec=nprod_vec;
Cst.idx=cumsum(nprod_vec);

Cst.mktsize = M;
Cst.mkt_idx=mkt_idx;

Cst.maxiter=500;
Cst.mvaltol=1e-14;  %tolerance for inversion of mean value 'delta'

Cst.gradInv=0; % Use gradient-based inversion if Cst.gradInv==1
Cst.pfor=0; % Use parfor if Cst.pfor==1


Cst.model_id=id;    %yr/model id, from 1 to 631
Cst.firm_id=firm;  %yr/firm id, from 1 to 135

% XMat includes regressors and IVs
XMat.weight=weight;
XMat.hp=hp;
XMat.AC=AC;
XMat.Z=Z;

% MuMat includes individual draws 
MuMat.inc_all = IncomeMat_all; % in thousand yuan

% Sales by market; sales for each segment by market
% YMat includes endogenous variables
YMat.price=price;
YMat.sales=s;

% Slice s_jt and mvalold market by market
% CL_sjt is a cell array
CL_sjt =cell(nmk,1);
CL_mvalold =cell(nmk,1);
for t = 1:nmk
    aa=index_begin(t);
    bb=index_end(t);
    CL_sjt{t} = s(aa:bb);
    nprod_m=nprod_vec(t);
    CL_mvalold{t} = zeros(nprod_m,1);
end;
YMat.CL_sjt=CL_sjt;
YMat.CL_mvalold=CL_mvalold;

% 3.4 Call optimization routine to estimate parameters
LowerBnd=[0];
UpperBnd=[10^10];

options = optimoptions('fmincon','Display','iter');

[theta2, fval, flag, output] = fmincon('WF1_objGMM', theta2w, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
 
display('alpha_2 estimate is')
theta2
 
% recover linear parameters
[CL_mval_final, CL_mu] = WF2_MeanVal(theta2, Cst, XMat, YMat, MuMat);

delta_vec=zeros(nprods,1);
for m = 1:nmk
    delta = CL_mval_final{m};
    % append to the residual vector
    aa=index_begin(m);
    bb=index_end(m);
    delta_vec(aa:bb,1)=delta;
end

X = [ones(nprods,1) price weight hp AC];
beta_hat_IV_2sls = (X'*Z*(Z'*Z)^(-1)*Z'*X)\(X'*Z*(Z'*Z)^(-1)*Z'*delta);
display('linear parameter estimates for constant, price, weight, hp, and AC are')
beta_hat_IV_2sls

% Code for debugging
[theta2_temp, fval, flag, output] = fmincon('WF1_objGMM', 2.5, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);


% 3.5 Sensitivity check using multi-start
theta2w_vec=20*rand(5,1);
theta2_vec=nan(length(theta2w_vec),1);
for k=1:length(theta2w_vec)
    [theta2_vec(k), fval, flag, output] = fmincon('WF1_objGMM', theta2w_vec(k), [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
end

% 3.6 One more exercise:
% We can actually compare with the case of weak instruments and alpha_2 is
% weakly identified. The GMM obj is pretty flat and does not search well
% over the parameter space
Z = [Z1];
XMat.Z=Z;

[theta2_temp, fval, flag, output] = fmincon('WF1_objGMM', 3.1, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
theta2_temp

[theta2_temp, fval, flag, output] = fmincon('WF1_objGMM', 2.5, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
theta2_temp

Z = [Z1 Z2];
XMat.Z=Z;

[theta2_temp, fval, flag, output] = fmincon('WF1_objGMM', 3.1, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
theta2_temp
[theta2_temp, fval, flag, output] = fmincon('WF1_objGMM', 2.5, [],[],[],[],...
        LowerBnd,UpperBnd,[], options, Cst, XMat, YMat, MuMat);
theta2_temp