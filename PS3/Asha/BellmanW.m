function [val_fun,pol_fun,pol_fun_ind,lsmat] = BellmanW(pol_fun,pol_fun_ind,val_fun,startage,endage,assetgrid,nasset,nprod,effvec,pival,rval,wval,sigmaval,betaval,gammaval,thetaval)
 
    val = zeros(1,nasset) ; 
    lsmat  = NaN(nasset,nprod,endage) ; 

    %Worker's problem
    for ij = endage:-1:startage
        for iz = 1:nprod
            for ia = 1:nasset 
                a = assetgrid(ia) ;
                for iap = 1:nasset
                    ap = assetgrid(iap) ;
                    ls = lsupply(a,ap,effvec(iz,ij),gammaval,thetaval,wval,rval) ;
                    cons = wval*(1-thetaval)*effvec(iz,ij)*ls + (1+rval)*a - ap ;            
                    if cons > 0 
                        util = (((cons^gammaval)*((1-ls)^(1-gammaval)))^(1-sigmaval)) / (1-sigmaval) ; 
                        val(iap) = util + betaval*val_fun(iap,:,ij+1)*pival(iz,:)' ;
                    else
                        val(iap) = -inf ;
                    end
                end
                [val_fun(ia,iz,ij), pol_fun_ind(ia,iz,ij) ] = max(val) ;
                pol_fun(ia,iz,ij) = assetgrid(pol_fun_ind(ia,iz,ij));
                lsmat(ia,iz,ij) = lsupply(a,pol_fun(ia,iz,ij),effvec(iz,ij),gammaval,thetaval,wval,rval) ;
            end 
        end
    end
end