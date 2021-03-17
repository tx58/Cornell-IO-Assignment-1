import excel "C:\Users\13695\Documents\IO\a1\data.xls", sheet("Sheet1") clear
rename A share
rename B price
rename C weight
rename D hp
rename E ac
rename F Z1
rename G Z2
rename H Z3
rename I Z4
gen mkt=1
gen cons=1
blp share weight hp ac, endog(price=Z1 Z2 Z3 Z4) stochastic(price) markets(mkt) 
