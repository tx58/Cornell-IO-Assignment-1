function f = WF4_IndShr(mval, mu)
% This function computes the individual purchase probability in a specific market.
% nprod is the number of products in that market
% The function is made as lean as possible to reduce computation time.

% Numerator of is the exponent of utilities
% Denominator is the sum of the numerators of all products in the market

% ===================== %
% version 1.1
% Last update: May 1, 2018
% Changes:
% 1. remove the input argument Cst, use the size of mu instead (as we have
%    to obtain the same dimension anyway). This function becomes more
%    generic.
% ===================== %

ndraws=size(mu,2);
nprod=length(mval);
%display(nprod)

Numerator = exp(mu+mval*ones(1,ndraws));
Denominator=ones(nprod,1)*sum(Numerator);

f = Numerator./(1+Denominator);