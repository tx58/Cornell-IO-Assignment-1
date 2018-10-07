function fval=evalSingle(theta, delta, X, price)
    resid= delta- X*theta(1:3)- price*theta(4);
    Z=[X price];
    W=eye(size(Z,2));
    % this extracts the parameters for your specification
    fval=(resid'*Z)*W*(resid'*Z)';
end