function f= BlpSupply(theta)
    global S halton r beta_1 beta_3
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
    tol= 1e-5; % Nevo's suggestion: to begin with a big tolerance, with 100 times more iteration, drop 1 digit
    diff=1;
    
    % Step2. recover delta
    delta=share./share_outside.*exp(price./meanvalue);
    %delta=ones(131,1);
    count=1;
    while diff>tol*ceil(count/100) 
        expshare= mktshare(delta);
        deltanew= delta.*share./expshare;
        diff =max(abs(deltanew-delta));
        delta= deltanew;
        count=count+1;
    end
    
    % Step3. find linear parameter
    
    j=size(X,1);
    Omega= zeros(j);
    S(j,j)=0;
    for i=1:j
        for k=1:j
            if i==k
                S(i,k)= -mean( 1./r.*share(i).*(1-share(k)));
            else
                S(i,k)= mean( 1./r.*share(i).*share(k));
            end
            if firm_index(i)==firm_index(k)
                Omega(i,k)=1 ;
            end
        end
    end
    
    Z= [Z3(:,1:2) X];
    mc_blp= price+ (Omega.*S)\share;
    dep= [ mc_blp; log(delta) ];
    indep= [X,quantity,zeros(size(X,1),size(X,2)); zeros(size([X,quantity],1),size([X,quantity],2)), X];
    instrument=kron(eye(2),Z);
    %[beta1, res1]= ivregression(Y-alpha*price, X, Z);
    %[beta2, res2]= ivregression(mc_logit, X, Z);
    [beta_1, resid]= ivregression(dep, indep, instrument);
    temp=resid'*instrument;
    W= inv(instrument'*instrument);
    
    % If using two step GMM:
    gn=resid.*instrument;
    W= inv((gn-mean(gn))'*(gn-mean(gn)));
    
    [beta_3, resid]= ivregression(dep, indep, instrument, W);
    
    f= resid'*instrument*W*instrument'*resid;

    %save('result_logit','beta1','beta2')
end