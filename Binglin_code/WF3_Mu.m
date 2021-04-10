function mu = WF3_Mu(theta2, Cst, XMat, YMat, MuMat)
% This function computes the non-linear part of the utility mu_ij
% Input parameters:
% 1) theta2: non-linear paramters
% 2) p: price faced by consumers
% 3) chars: attributes with heteregeneous tastes
% 4) house_draws (hh taste for attributes and price), inc_draws (hh income)
%======================%
% version 1.1
% Last update: Apr 28, 2018
% Changes:
%   1. delete code that is commented out.
%   2. Change function name to WF2.
%   3. surpress return values for P_ij, price_coef]
%======================%

%============
% code for debugging the function:
clearvars -except theta2 Cst XMat YMat MuMat
%============

% ========================
% Step 0: no random coef for price if Cst.sigmaP=0
% in which case theta2w is less than 4 rows (incompatible with line 81: price_coef)
theta2w = full(sparse(Cst.theti,Cst.thetj,theta2));

% Step 1: read required data
% Obtain constants
nprods = Cst.nprods;
ndraws = Cst.ndraws;

% Obtain individual draws
%house_draws=MuMat.hh; %nprods by ndraws by nrands+1
inc  = MuMat.inc_all; % nprods by ndraws

% Obtain price and other car attributes
price=YMat.price;   % for now only price is relevant in Mu part utility
%weight=XMat.weight;
%hp=XMat.hp;
%AC=XMat.AC;

%%% Step 2: Calculate mu_ij

% initialize nonlinear utility
mu = zeros(nprods,ndraws);

% price coeffcient and mu_ij from price
price_coef = theta2w(1,1)./inc;
P_ij = -price_coef.*(price*ones(1,ndraws));
mu = mu + P_ij;

% other observed heterogeneities in mu_ij -> none


