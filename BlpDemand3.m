function f= BlpDemand3(alpha)
% This returns BLP value function with only demand side
    global halton r beta_1 beta_2 price W
    load data_blp.mat 
    % Step1. draw from distribution parameter theta
    j= size(X,1);
    meanvalue= 35000;
    std= 45000;
    mu=log(meanvalue/sqrt(1+std^2/meanvalue^2));
    sigma=sqrt(log(std^2/meanvalue^2+1));
    %[mu,sigma]= lognstat(mean,variance)
    if sigma>0
        r = icdf('logn', halton, mu, sigma );
    else
        r = meanvalue.*ones(size(halton));
    end
    tol= 1e-6; % Nevo's suggestion: to begin with a big tolerance, with 100 times more iteration, drop 1 digit
    diff=1;
    
    
    % Step2. recover delta
    delta=share./share_outside.*exp(alpha(1)* price./meanvalue);
    count=1;
    while diff>tol*ceil(count/10)  
        expshare= mktshare2(delta, alpha(1), alpha(2) );
        deltanew= delta.*share./expshare;
        diff =max(abs(share-expshare));
        delta= deltanew;
        count=count+1;
    end
    
    % Step3. use delta to calculate the linear parameter
    
    Z= [Z3(:,1:2) ones(j,1) X];
    dep=  log(delta) ;
    indep= [ones(j,1) X];
    instrument= Z;
    %[beta1, res1]= ivregression(Y-alpha*price, X, Z);
    %[beta2, res2]= ivregression(mc_logit, X, Z);
    [beta_2, resid]= ivregression(dep, indep, instrument, W);
    f= resid'*instrument*W*instrument'*resid;
end


function f= mktshare2(delta, alpha1, alpha2)
    % This function returns the calculated market share for each product:
    % s_j= 1/r*(exp(delta_j+mu_jr)/sum_{j}{exp(delta_j+mu_jr)})
    global price draw r
    nonlinear= exp(price* (alpha1+ alpha2./r)');
    numerator= ( delta* ones(1,draw) ./ nonlinear );
    denominator= sum(numerator, 1)+1;
    f= mean( numerator./denominator,2 );
end
