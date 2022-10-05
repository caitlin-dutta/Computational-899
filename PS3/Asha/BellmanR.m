function [val_fun,pol_fun,pol_fun_ind] = BellmanR(startage,endage,assetgrid, nasset,nprod,bval,rval,sigmaval,betaval,gammaval)
    % Initialize matrices
    val_fun = NaN(nasset,nprod,endage) ; 
    pol_fun = NaN(nasset,nprod,endage) ; 
    pol_fun_ind = NaN(nasset,nprod,endage) ;
    

    % Value functions in the last period 
    cons = (1+rval)*assetgrid + bval ;
    val_fun(:,1,endage) = (cons.^((1-sigmaval)*gammaval))/(1-sigmaval);
    val_fun(:,2,endage) = (cons.^((1-sigmaval)*gammaval))/(1-sigmaval);
    
    % Policy functions in the last period 
    pol_fun(:,1,endage) = 0;
    pol_fun(:,2,endage) = 0;
    pol_fun_ind(:,1,endage) = 1;
    pol_fun_ind(:,2,endage) = 1;
    

    val = zeros(nasset,nprod) ; 

    % Backward induction 
    for ij = endage-1:-1:startage
        for ia = 1:nasset 
            a = assetgrid(ia) ;
            budgetR = (1+rval)*a + bval ;
            for iap = 1:nasset
                ap = assetgrid(iap) ;
                cons = budgetR - ap ;
                if cons > 0
                    util = (cons^((1-sigmaval)*gammaval))/(1-sigmaval) ;
                    val(iap,:) = util + betaval*val_fun(iap,:,ij+1);
                else
                    val(iap,:) = -inf ;
                end
            end     
            [val_fun(ia,:,ij), pol_fun_ind(ia,:,ij) ] = max(val) ;
            pol_fun(ia,:,ij) = assetgrid(pol_fun_ind(ia,:,ij));
                  
        end
    end
end