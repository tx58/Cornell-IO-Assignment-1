function f= BlpDemand4(alpha)
% This returns BLP value function with only demand side
    global halton r beta_2 price W
    load data_blp.mat 
    % Step1. draw from distribution parameter theta
    j= size(X,1);
    meanvalue= 35000;
    std= 45000;
    mu=log(meanvalue/sqrt(1+std^2/meanvalue^2));
    sigma=sqrt(log(std^2/meanvalue^2+1));
    if sigma>0
        r = icdf('logn', halton, mu, sigma );
    else
        r = meanvalue.*ones(size(halton));
    end
    tol= 1e-6; % Nevo's suggestion: to begin with a big tolerance, with 100 times more iteration, drop 1 digit
    diff=1;
    
    
    % Step2. recover delta
    delta=share./share_outside.*exp(alpha* price./meanvalue);
    count=1;
    while diff>tol*ceil(count/10)  
        expshare= mktshare2(delta, alpha);
        deltanew= delta.*share./expshare;
        diff =max(abs(share-expshare));
        delta= deltanew;
        count=count+1;
    end
    
    % Step3. use delta to calculate the linear parameter
    
    Z= [Z1(:,1:2) Z2(:,1:2) ones(j,1) X];
    dep=  log(delta) ;
    indep= [ones(j,1) X price];
    instrument= Z;
    W= inv(instrument'*instrument);    
    [beta_2, resid]= ivregression(dep, indep, instrument);
    f= resid'*instrument*W*instrument'*resid;
    gn=resid.*instrument;
    W= inv((gn-mean(gn))'*(gn-mean(gn)));
end


function f= mktshare2(delta, alpha)
    % This function returns the calculated market share for each product:
    % s_j= 1/r*(exp(delta_j+mu_jr)/sum_{j}{exp(delta_j+mu_jr)})
    global price draw r
    nonlinear= exp(price.* (alpha./r)');
    numerator= ( delta* ones(1,draw) ./ nonlinear );
    denominator= sum(numerator, 1)+1;
    f= mean( numerator./denominator,2 );
end
