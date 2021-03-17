function f= BlpSupply2(alpha)
    global S halton r beta_1 beta_3 mc_blp W
    load data_blp.mat
    % Step1. draw from distribution parameter theta
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
    delta=share./share_outside.*exp(price./meanvalue);
    %delta=ones(131,1);
    count=1;
    while diff>tol*ceil(count/100) 
        expshare= mean( mktshare2(delta, alpha(1), alpha(2)), 2);
        deltanew= delta.*share./expshare;
        diff =max(abs(share-expshare));
        delta= deltanew;
        count=count+1;
    end
    
    sharer= mktshare2(delta, alpha(1), alpha(2));
    % Step3. find linear parameter
    
    j=size(X,1);
    Omega= zeros(j);
    S(j,j)=0;
    for i=1:j
        for k=1:j
            if i==k
                S(i,k)= -mean( (alpha(1)+ alpha(2)./r).* sharer(i,:)' .* (1- sharer(k,:))' );
            else
                S(i,k)= mean( (alpha(1)+ alpha(2)./r).*  sharer(i,:)' .* sharer(k,:)' );
            end
            if firm_index(i)==firm_index(k)
                Omega(i,k)=1 ;
            end
        end
    end
    Z= [Z1(:,1:2) Z2(:,1:2)  ones(j,1) X];
    mc_blp= price+ (Omega.*S)\share;
    dep= [ mc_blp; log(delta) ];
    X1=[ones(j,1),X,quantity];
    X2=[ones(j,1),X];
    indep= [X1, zeros(size(X2,1),size(X2,2)); zeros(size(X1,1),size(X1,2)), X2];   
    instrument=kron(eye(2),Z);
    %[beta1, res1]= ivregression(Y-alpha*price, X, Z);
    %[beta2, res2]= ivregression(mc_logit, X, Z);
    W= inv(instrument'*instrument);
    % If using two step GMM:
    
    [beta_3, resid]= ivregression(dep, indep, instrument);
    f= resid'*instrument*W*instrument'*resid;
    gn=resid.*instrument;
    W= inv((gn-mean(gn))'*(gn-mean(gn)));
end


function f= mktshare2(delta, alpha1, alpha2)
    % This function returns the calculated market share for each product:
    % s_j= 1/r*(exp(delta_j+mu_jr)/sum_{j}{exp(delta_j+mu_jr)})
    global price draw r
    nonlinear= exp(price* (alpha1+ alpha2./r)');
    numerator= ( delta* ones(1,draw) ./ nonlinear );
    denominator= sum(numerator, 1)+1;
    f= ( numerator./denominator);
end