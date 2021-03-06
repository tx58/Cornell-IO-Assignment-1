function f= BlpDemand(theta)
% This returns BLP value function with only demand side
    global halton r beta_1 beta_2 price
    load data_blp.mat 
    % Step1. draw from distribution parameter theta
    meanvalue= theta(1);
    std= theta(2);

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
    delta=share./share_outside.*exp(price./meanvalue);
    count=1;
    while diff>tol*ceil(count/10)  
        expshare= mktshare(delta);
        deltanew= delta.*share./expshare;
        diff =max(abs(deltanew-delta));
        delta= deltanew;
        count=count+1;
    end
    
    % Step3. use delta to calculate the linear parameter
    
    %Z= [Z1 Z2 X];
    dep=  log(delta) ;
    indep= X;
    instrument= Z;
    %[beta1, res1]= ivregression(Y-alpha*price, X, Z);
    %[beta2, res2]= ivregression(mc_logit, X, Z);
    [beta_1, resid]= ivregression(dep, indep, instrument);
    temp=resid'*instrument;
    W= inv(instrument'*instrument);
    
    % If using two step GMM:
    gn=resid.*instrument;
    W= inv((gn-mean(gn))'*(gn-mean(gn)));
    
    [beta_2, resid]= ivregression(dep, indep, instrument, W);
    
    f= resid'*instrument*W*instrument'*resid;

    %save('result_logit','beta1','beta2')
end