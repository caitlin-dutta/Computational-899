function ls = lsupply(aval,apval,eval,gammaval,thetaval,wval,rval) 
lstemp = (gammaval*(1-thetaval)*eval*wval - ...
            (1-gammaval)*((1+rval)*aval - apval))/((1-thetaval)*wval*eval);
ls = max(0,min(1,lstemp)) ;
end
    