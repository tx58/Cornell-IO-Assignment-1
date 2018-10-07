function f =LogitSupply(alpha)
    global beta S mc_logit
    load data_logit.mat
    j=size(X,1);
    Omega= zeros(j);
    S(j,j)=0;
    for i=1:j
        for k=1:j
            if i==k
                S(i,k)= -alpha*share(i)*(1-share(k));
            else
                S(i,k)= alpha*share(i)*share(k);
            end
            if firm_index(i)==firm_index(k)
                Omega(i,k)=1 ;
            end
        end
    end
    
    Z= [Z1 Z2 X];
    mc_logit= price+ (Omega.*S)\share;
    dep= [ mc_logit; Y-alpha*price ];
    indep= [X,quantity,zeros(size(X,1),size(X,2)); zeros(size([X,quantity],1),size([X,quantity],2)), X];
    instrument=kron(eye(2),Z);
    %[beta1, res1]= ivregression(Y-alpha*price, X, Z);
    %[beta2, res2]= ivregression(mc_logit, X, Z);
    [beta, resid]= ivregression(dep, indep, instrument);
    temp=resid'*instrument;
    W= inv(instrument'*instrument);
    
    % If using two step GMM:
    gn=resid.*instrument;
    W= inv((gn-mean(gn))'*(gn-mean(gn)));
    
    [beta, resid]= ivregression(dep, indep, instrument, W);
    
    f= resid'*instrument*W*instrument'*resid;

    %save('result_logit','beta1','beta2')
end