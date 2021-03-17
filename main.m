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

x_summary= [X price quantity];
% Summary statistics:
% xlswrite('q0_st.xls', {'weight/100'; 'hp/10'; 'ac'; 'price/1000'; 'quantity/1000'} ,1, 'A2');
% xlswrite('q0_st.xls', {'Mean', 'Std', 'Max', 'Min'} ,1, 'B1');
% xlswrite('q0_st.xls', [mean(x_summary)', std(x_summary)', max(x_summary)', min(x_summary)'] , 1, 'B2');
% xlswrite('data.csv', [share price X Z1(:,1:2) Z2(:,1:2)]);
%% Vertical model
% Initialization
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

% regression: $\delta_{j}=x_{j}\beta+ \xi_{j}$
% Simple OLS
[result.ols,result.betaint]= regress(delta, XC);

% Instrumental variable approach: take rival's products in the same markets
% as instruments.
  Z=[X Z1(:,1:2) Z2(:,1:2)];
result.IV = ivregression(delta, XC, Z);

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
indep= [XC,quantity,zeros(size(XC,1),4); zeros(size(XC,1),5), XC ];
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

tableA=[mc1 mc2 mc3 mc4 price];
mean(tableA, 1)
vertical=[result.MC result.SP result.MP result.CL];
vertical(1:4,:)

xlswrite('q1.xls', {'D_constant', 'D_weight', 'D_hp', 'D_ac'}, 1, 'B1');
xlswrite('q1.xls', {'IV'; 'OLS'}, 1, 'A2');
xlswrite('q1.xls', [result.IV result.ols]', 1, 'B2');

xlswrite('q3.xls', {'C_constant'; 'C_weight'; 'C_hp'; 'C_ac'; 'quantity'; 'D_constant';'D_weight'; 'D_hp'; 'D_ac'}, 1, 'A2');
xlswrite('q3.xls', {'MC', 'SP', 'MP', 'CL'}, 1, 'B1');
xlswrite('q3.xls', vertical, 1, 'B2');

xlswrite('q4.xls', {'MC', 'SP', 'MP', 'CL', 'Price'}, 1, 'B1');
xlswrite('q4.xls', {'mean'; 'markup'}, 1, 'A2');
xlswrite('q4.xls', [mean(tableA, 1); mean(price)./mean(tableA, 1)-1] ,1, 'B2' );
%% Logit model

delta=log(share)-log(share_outside); % recover delta in logit
%[beta,betaint]= regress(delta, [X price]);
X_OLS=[ones(j,1) price X];
Y=delta;
beta_OLS= (X_OLS'*X_OLS)\X_OLS'*Y;
Z=[Z1(:,1:2) Z2(:,1:2) X]; % IV
P= Z*(Z'*Z)^(-1)*Z';
beta_IV= (X_OLS'*P*X_OLS)^(-1)*X_OLS'*P*Y;
%ivregression(Y,[X price],Z)
result.logitdemand= beta_IV;
save('data_logit.mat','X','price','Y','Z1','Z2','Z3','Z','quantity','share','firm_index')

% When we include supply side
% we need write the moment function and minimize it
% in the outer loop, we search for the price coefficient
% in the inner loop, it is a linear problem simply use GMM to evaluate the parameters

global beta1 mc_logit S
    options=optimset('Display', 'iter');    
[x0, fval]=fminsearch( 'LogitSupply', -beta_IV(2), options);
result.logitsupply=[x0; beta1];
elasticity_logit=S./share.*price';
mc.logit=mc_logit;

i=1;
for k=x0/10:x0/100:2*x0
    y(i)=LogitSupply(k);
    %y(i)=BlpSupply([1/k,0]);
    i=i+1;
end
x=x0/10:x0/100:2*x0;

plot(x,y)
hold on
plot(x(90),y(90),'ro')
grid on
axis on
xlabel('Price coefficient \theta')
ylabel('GMM objective function value')
print -djpeg -r600 q5.jpg

xlswrite('q5_1.xls', [mc.logit price])


logit= [ [zeros(5,1); beta_OLS], [ zeros(5,1); result.logitdemand], result.logitsupply];
xlswrite('q5_2.xls', {'C_cons'; 'C_weight'; 'C_hp'; 'C_ac'; 'quantity';'D_cons'; 'D_price'; 'D_weight'; 'D_hp'; 'D_ac'}, 1, 'A2');
xlswrite('q5_2.xls', {'logit_OLS', 'logit_IV', 'logit_Supply'}, 1, 'B1');
xlswrite('q5_2.xls', logit,1 ,'B2');

%% BLP model
% The difference is that now we need to draw from the sample, the
% specification becomes:
% alpha_{i}=1/y_{i}, where y_{i} follows lognormal(mu, sigma)

delta=log(share)-log(share_outside); % recover delta in logit
%[beta,betaint]= regress(delta, [X price]);
Z=[Z1(:,1:2) Z2(:,1:2)  X];       
save('data_blp.mat','X','price','Z1','Z2','Z3','Z','quantity','share','share_outside','firm_index')

global draw halton beta_2 W
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);
% BlpDemand([35000,45000]);
% BlpSupply([35000,45000]);
% xlswrite('q7.xls', [ [0;0;0;0;beta_2] beta_3]);
% xlswrite('q8.xls', S);
lb = [0,0];
ub = [100,3000000];
     options=optimset('Display', 'iter'); 
     [result.blpdemand, fval]=fmincon( 'BlpDemand2',[2.5, 100000],[],[],[],[],lb,ub, [],options);
     [result.blpdemand, fval]=fmincon( 'BlpDemand3',[2.5, 100000],[],[],[],[],lb,ub, [],options);
    result.blpdemand=[result.blpdemand beta_2'];
     
%      output=[];
% for input=result.blpdemand(1)/4: result.blpdemand(1)/20: 2*result.blpdemand(1)
%     output=[output (BlpDemand2([input,result.blpdemand(2)]))];
% end
% xaxis=result.blpdemand(1)/4: result.blpdemand(1)/20: 2*result.blpdemand(1);
% plot(xaxis, output)
% xlabel('Variance term of non-linear parameter \sigma')
% ylabel('objective function value')
% print -djpeg -r600 q9_1.jpg
%%
lb = 0;
ub = 10000000;
     options=optimset('Display', 'iter'); 
     [x0, fval]=fmincon( 'BlpDemand4',25000,[],[],[],[],lb,ub, [],options);
%% BLP supply side
Z=[Z3(:,1:2) X];       
save('data_blp.mat','X','price','Z1','Z2','Z3','Z','quantity','share','share_outside','firm_index')

global price draw halton beta_3 S mc_blp W
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);
lb = [-10,0];
ub = [10,30000000];
options = optimoptions('fmincon','Display','iter');
[result.blpsupply, fval]=fmincon( 'BlpSupply2',[1,100000],[],[],[],[],lb,ub, [],options);

p = haltonset(1,'Skip',1e3,'Leap',1e3);
draw= 1000;
halton= net(p, draw);

[result.blpsupply, fval]=fmincon( 'BlpSupply3',result.blpsupply,[],[],[],[],lb,ub, [],options);
result.blpsupply=[result.blpsupply beta_3'];
mc.blp=mc_blp;
elasticity_blp=S./share.*price';

%gs=GlobalSearch;
% ObjectiveFunction=@BlpSupply;
% problem= createOptimProblem('fmincon','x0',[0.5 0.5],...
%     'objective',ObjectiveFunction,'lb',[0 0],'ub',[10 10]);
%x=run(gs, problem)
output=[];
for input=result.blpsupply(1)-0.01:0.001:result.blpsupply(1)+0.01
    output=[output (BlpSupply2([ input, result.blpsupply(2)]))];
end
xaxis=result.blpsupply(1)-0.01:0.001:result.blpsupply(1)+0.01;
plot(xaxis, output)
xlabel('Variance term of non-linear parameter \alpha_{1}')
ylabel('objective function value (adding supply side)')
print -djpeg -r600 q9_2.jpg
hold off

output=[];
for input=log(result.blpsupply(2))-10:0.1:1+log(result.blpsupply(2))
    output=[output (BlpSupply2([ result.blpsupply(1), exp(input)]))];
end
xaxis=log(result.blpsupply(2))-1:0.1:10+log(result.blpsupply(2));
plot(xaxis, output)
xlabel('Variance term of non-linear parameter \alpha_{2}')
ylabel('objective function value (adding supply side)')
print -djpeg -r600 q9_3.jpg

%% Micro BLP model
% add an additional moments:

global price draw halton beta_4 S mc_blp
p = haltonset(1,'Skip',1e3,'Leap',1e2);
draw= 1000;
halton= net(p, draw);
lb = [0,0];
ub = [10,1000000];
options = optimoptions('fmincon','Display','iter');
[result.blpsupply, fval]=fmincon( 'BlpSupply3',[0.1,1000],[],[],[],[],lb,ub, [],options);
elasticity_microblp=S./share.*price';

%% Write out the results

label={'alpha_1','alpha_2','C_cons','C_weight','C_hp','C_ac','C_q','D_cons','D_weight','D_hp','D_ac'};
xlswrite('q9.xls', [result.blpsupply'], 1, 'B2');
xlswrite('q9.xls', label', 1, 'A2');
xlswrite('q9.xls', [result.blpdemand'], 1, 'C2');
xlswrite('q9.xls', [result.logitsupply(1);0;result.logitsupply(2:10) ], 1, 'D2');
xlswrite('q9.xls', [-result.logitdemand(1);0;0;0;0;0;0;result.logitdemand(2:5)], 1, 'E2');
xlswrite('q9.xls', {'blp_supply', 'blp_demand', 'logit_supply', 'logit_demand'}, 1, 'B1');


xlswrite('q9_mc.xls', [mc.logit(1), mc.blp(1), price(1)], 1 ,'A2');
xlswrite('q9_mc.xls', [mc.logit(131), mc.blp(131), price(131)], 1 ,'A3');
xlswrite('q9_mc.xls', mean([mc.logit, mc.blp, price]), 1 ,'A4');
xlswrite('q9_mc.xls', {'mc_logit', 'mc_blp', 'price'}, 1 ,'A1');




