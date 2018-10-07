function f= mktshare(delta, theta)
% This function returns the calculated market share for each product:
% s_j= 1/r*(exp(delta_j+mu_jr)/sum_{j}{exp(delta_j+mu_jr)})
global price draw r

numerator= mean( delta.*ones(1,draw)./exp(1./r' .*price), 2);
denominator= sum(numerator)+1;
f= numerator./denominator;
