%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IO HOMEWORK1 
% Tianli Xia
% Sep 22nd, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code: Submitted Version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; 
clc
tic

%% Initialization & Preprocessing
A= load("Yr18_PSet_BLP_data_no_header.txt");
A= sortrows(A, 1); % desceding according to price
j= size(A, 1);

price= A(:,1); % price in dollars
quantity= A(:,2); % quantities sold
household=100000000; % total market consumers
share= quantity/household; % calculate market share
share_outside= 1- sum(share);
X= A(:,3:5); % car weight, horse power, AC dummy
% Preprocess to make the scale similar
X(:,1)= A(:,3)/100;
X(:,2)= A(:,4)/10;
price= A(:,1)/1000;
quantity= A(:,2)/1000;
firm_index= A(:,6);
firm= dummyvar(firm_index);

% Set two sets of instruments: 
% (1) average product characters produced by the same companies
Z1(j,3)=0;
Z2(j,3)=0;
for i=1:j
    Z1(i,:)= mean( X(firm_index==firm_index(i),:), 1);
end
% (2) average product characters produced by the rivals
for i=1:j
    Z2(i,:)= mean( X(firm_index~=firm_index(i),:), 1);
end
% (3) optimal instrument by JF Houde
for i=1:j
    Z3(i,:)= sum(abs( X(i,:)-X ));
end

%% 2. Vertical model
% 2.1 Initialization
lamda= 1/(4*10^(-6));
alpha(j,1)=0;
delta(j,1)=0;
alpha(1)= -log(share_outside)/lamda;
delta(1)= alpha(1)*price(1); % horizontal normalization, outside option brings utility of 0.
for i=2:j
    alpha(i)=-log(exp(-lamda*alpha(i-1)) +share(i-1))/lamda;
    delta(i)= delta(i-1)- alpha(i)*price(i-1)+ alpha(i)*price(i);
end

XC= [ones(j,1) X];

% 2.2 regression: $\delta_{j}=x_{j}\beta+ \xi_{j}$
% Simple OLS
[result.ols,result.betaint]= regress(delta, XC);

% 2.3 Instrumental variable approach: take rival's products in the same markets as instruments.
  Z=[X Z1(:,1:2) Z2(:,1:2)];
result.IV = ivregression(delta, XC, Z);

% 2.4 Include supply side
S(j,j)=0; % DShare/Dprice matrix
for i=1:j
    if i==1
        S(i,i)= -lamda*exp(-lamda*alpha(i+1)) * (delta(i+1)-delta(i))/(price(i+1)-price(i))^2 ...
            -lamda*exp(-lamda*alpha(i)) * (delta(i))/(price(i))^2;
        S(i,i+1)= lamda*exp(-lamda*alpha(i+1)) * (delta(i+1)-delta(i))/(price(i+1)-price(i))^2;
    end
    if i~=1 && i~=j
        S(i,i)= -lamda*exp(-lamda*alpha(i+1)) * (delta(i+1)-delta(i))/(price(i+1)-price(i))^2 ...
            -lamda*exp(-lamda*alpha(i)) * (delta(i)-delta(i-1))/(price(i)-price(i-1))^2;
        S(i,i+1)= lamda*exp(-lamda*alpha(i+1)) * (delta(i+1)-delta(i))/(price(i+1)-price(i))^2;
        S(i,i-1)= lamda*exp(-lamda*alpha(i)) * (delta(i)-delta(i-1))/(price(i)-price(i-1))^2;
    end
    if i==j
        S(i,i)= -lamda*exp(-lamda*alpha(i)) * (delta(i)-delta(i-1))/(price(i)-price(i-1))^2;
        S(i,i-1)= lamda*exp(-lamda*alpha(i)) * (delta(i)-delta(i-1))/(price(i)-price(i-1))^2;
    end
end

% Case1: marginal cost pricing $mc_{j}=p_{j}=x_{j}\gamma+ \eta q_{j} + w_{j}$
Omega1= zeros(j,j); % Ownership matrix
mc1= price;

dep= [ mc1; delta ];
indep= [XC,quantity,zeros(size(XC,1),4); zeros(size(XC,1),5), XC ];
instrument=kron(eye(2),Z);
[result.MC, ~]= ivregression(dep, indep, instrument);

% Case2: single product firm
Omega2= eye(j);
mc2= price+ (Omega2.*S)\share;
dep= [ mc2; delta ];
[result.SP, ~]= ivregression(dep, indep, instrument);

% Case3: multiple product firm
Omega3= zeros(j);
for i=1:j
    for k=1:j
        if firm_index(i)==firm_index(k)
            Omega3(i,k)=1 ;
        end
    end
end
mc3= price+ (Omega3.*S)\share;
dep= [ mc3; delta ];
[result.MP, ~]= ivregression(dep, indep, instrument);

% Case4: collusion
Omega4= ones(j);
mc4= price+ (Omega4.*S)\share;
dep= [ mc4; delta ];
[result.CL, ~]= ivregression(dep, indep, instrument);


%% 3. Logit model

%3.1 Demand side logit
delta=log(share)-log(share_outside); % recover delta in logit
X_OLS=[ones(j,1) price X];
Y=delta;
beta_OLS= (X_OLS'*X_OLS)\X_OLS'*Y;
Z=[Z1(:,1:2) Z2(:,1:2) X]; % IV
P= Z*(Z'*Z)^(-1)*Z';
beta_IV= (X_OLS'*P*X_OLS)^(-1)*X_OLS'*P*Y;
result.logitdemand= beta_IV;

% 3.2 Supply side logit

global beta1 mc_logit S
options=optimset('Display', 'iter');    
[x0, fval]=fminsearch( 'LogitSupply', -beta_OLS(4), options); % starting value: demand-side results
result.logitsupply=[x0; beta1];
elasticity_logit=S./share.*price';
mc.logit= mc_logit;

%% 4. BLP model

delta=log(share)-log(share_outside); % recover delta in logit
%[beta,betaint]= regress(delta, [X price]);
Z=[Z3(:,1:2) X];       
save('data_blp.mat','X','price','Z1','Z2','Z3','Z','quantity','share','share_outside','firm_index')

% Use Halton Draw to get random sequence
global draw halton beta_2 W
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);

% Optimization: Two-step GMM
lb = 0;
ub = 1000000;
options=optimset('Display', 'iter'); 
[result.blpdemand, ~]=fmincon( 'BlpDemand4',20000,[],[],[],[],lb,ub, [],options);

p = haltonset(1,'Skip',1e3,'Leap',1e2); % Redraw before the second time
draw= 1000;
halton= net(p, draw);
[result.blpdemand, fval]=fmincon( 'BlpDemand5',result.blpdemand ,[],[],[],[],lb,ub, [],options);
result.blpdemand=[result.blpdemand beta_2'];

%% 5. BLP supply side
Z=[Z3(:,1:2) X];       
save('data_blp.mat','X','price','Z1','Z2','Z3','Z','quantity','share','share_outside','firm_index')

global price draw halton beta_3 S mc_blp W
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);
lb = [-10,0];
ub = [10,30000000];
options = optimoptions('fmincon','Display','iter');
[result.blpsupply, fval]=fmincon( 'BlpSupply2',[result.blpdemand(6), result.blpdemand(1)],[],[],[],[],lb,ub, [],options);

p = haltonset(1,'Skip',1e3,'Leap',1e2); % Redraw before the second time
draw= 1000;
halton= net(p, draw);
[result.blpsupply, fval]=fmincon( 'BlpSupply3',result.blpsupply,[],[],[],[],lb,ub, [],options);
result.blpsupply=[result.blpsupply beta_3'];
mc.blp=mc_blp;
elasticity_blp=S./share.*price';

