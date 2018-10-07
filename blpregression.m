function f =blpregression(mu, sigma)
    load data_logit.mat
    global r
    j=size(X,1);
    Omega= zeros(j);
    tol=1e-4;
    % contraction mapping
    % exp(deltanew)-exp(delta)=s-exps
    for norm< tol
        
end