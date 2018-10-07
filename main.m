%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IO HOMEWORK1 
% Tianli Xia
% Sep 22nd, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Vertical model
% $u_{ij}=\delta_{j}-\alpha_{i}p_{j}$ ,$\delta_{j}=x_{j}\beta+ \xi_{j}$
% 2. Logit model 
% $u_{ij}=\delta_{j}-\alpha_{i}p_{j} + \epsilon_{j}$, $\delta_{j}=x_{j}\beta+ \xi_{j}$
% 3. BLP
% $u_{ij}=\delta_{j}-\alpha_{i}p_{j} + \epsilon_{j}$, $\delta_{j}=x_{j}\beta+ \xi_{j}$
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; 
clc
tic
%% Read and preprocess the data
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
% X(:,1)= A(:,3)./1000;
% X(:,2)= A(:,4)./100;
% price= A(:,1)./1e5;
% quantity= A(:,2)./1e5;
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
    Z3(i,:)= sum(abs( X(i)-X ));
end


%% Vertical model
% Initialization
lamda= 4*10^(-6);
alpha(j,1)=0;
delta(j,1)=0;
alpha(1)= -log(share_outside)/lamda;
delta(1)= alpha(1)*price(1); % horizontal normalization, outside option brings utility of 0.
for i=2:j
    alpha(i)=-log(exp(-lamda*alpha(i-1)) +share(i-1))/lamda;
    delta(i)= delta(i-1)- alpha(i)*price(i-1)+ alpha(i)*price(i);
end
 
% regression: $\delta_{j}=x_{j}\beta+ \xi_{j}$
% Simple OLS
[result.ols,result.betaint]= regress(delta, X);

% Instrumental variable approach: take rival's products in the same markets
% as instruments.
Z=[X Z1 Z2 Z3];
result.IV = regress(delta, X, Z);

% Include supply side
% S_ij matrix
S(j,j)=0;
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

% case1: marginal cost pricing $mc_{j}=p_{j}=x_{j}\gamma+ \eta q_{j} + w_{j}$
Omega1= zeros(j,j);
mc1= price;

dep= [ mc1; delta ];
indep= [X,quantity,zeros(size(X,1),size(X,2)); zeros(size([X,quantity],1),size([X,quantity],2)), X];
instrument=kron(eye(2),Z);
[result.MC, ~]= ivregression(dep, indep, instrument);

% case2: single product firm
Omega2= eye(j);
mc2= price+ (Omega2.*S)\share;
dep= [ mc2; delta ];
[result.SP, ~]= ivregression(dep, indep, instrument);

% case3: multiple product firm
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

% case4: collusion
Omega4= ones(j);
mc4= price+ (Omega4.*S)\share;
dep= [ mc4; delta ];
[result.CL, ~]= ivregression(dep, indep, instrument);

[mc1 mc2 mc3 mc4 price]
[result.MC result.SP result.MP result.CL]
%% Logit model

delta=log(share)-log(share_outside); % recover delta in logit
%[beta,betaint]= regress(delta, [X price]);
X_OLS=[ X price];
Y=delta;
beta_OLS= (X_OLS'*X_OLS)\X_OLS'*Y;
Z=[Z1 Z2 Z3 X]; % IV
P= Z*(Z'*Z)^(-1)*Z';
beta_IV= (X_OLS'*P*X_OLS)^(-1)*X_OLS'*P*Y;
%ivregression(Y,[X price],Z)
result.logitdemand= beta_IV;

save('data_logit.mat','X','price','Y','Z1','Z2','Z3','Z','quantity','share','firm_index')

% When we include supply side
% we need write the moment function and minimize it
% in the outer loop, we search for the price coefficient
% in the inner loop, it is a linear problem simply use GMM to evaluate the parameters
%     Omega= zeros(j);
%     S(j,j)=0;
%     for i=1:j
%         for k=1:j
%             if i==k
%                 S(i,k)= beta_OLS(4)*share(i)*(1-share(i));
%             else
%                 S(i,k)= -beta_OLS(4)*share(i)*share(k);
%             end
%             if firm_index(i)==firm_index(k)
%                 Omega(i,k)=1 ;
%             end
%         end
%     end
% 
%     mc_logit= price+ (Omega.*S)\share;
%     
global beta S mc_logit
    options=optimset('Display', 'iter');    
[x0, fval]=fminsearch( 'LogitSupply', -beta_OLS(4), options);
result.logitsupply=[beta; x0];

i=1;
for k=x0/10:x0/100:2*x0
    y(i)=LogitSupply(k);
    i=i+1;
end
x=1:i-1;
k=x0/10:x0/100:2*x0;
plot(x,y)
hold on
plot(x(90),y(90),'ro')
grid on
axis on
xlabel('Price coefficient \theta')
ylabel('GMM objective function value')
[mc_logit price]
[ [zeros(4,1); beta_OLS], [ zeros(4,1); result.logitdemand], result.logitsupply ]
%% BLP model
% The difference is that now we need to draw from the sample, the
% specification becomes:
% alpha_{i}=1/y_{i}, where y_{i} follows lognormal(mu, sigma)

delta=log(share)-log(share_outside); % recover delta in logit
%[beta,betaint]= regress(delta, [X price]);
X_OLS=[X price];
Y=delta;

save('data_blp.mat','X','price','Y','Z1','Z2','Z3','quantity','share','share_outside','firm_index')

global draw halton beta_2
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);

% tol=1e-4;
% mean=(35000);
% variance= 45000;
% mu=log(mean^2/sqrt(variance+mean^2));
% sigma=sqrt(log(variance/mean^2+1));
% %[mu,sigma]= lognstat(mean,variance)
% r = icdf('logn', halton, mu, sigma );
%BlpSupply([35000,45000])
lb = [0,1];
ub = [1000000,1000000];
     options=optimset('Display', 'iter'); 
     options = optimoptions('fmincon','Display','iter');
     [result.blpdemand, fval]=fmincon( 'BlpDemand',[35000,45000],[],[],[],[],lb,ub, [],options);
     [result.blpsupply, fval]=fmincon( 'BlpSupply',[35000,45000],[],[],[],[],lb,ub, [],options);

save('result.mat')

%% Micro BLP model



