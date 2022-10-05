function F = Distribution(endage,zprobmat,pol_fun_ind_mat,nval,nasset,nprod,muval,pival)
 
    % Find initial distribution 
    F = zeros(nasset,nprod,endage) ;
    F(1,:,1) = zprobmat * muval(1) ; 
    
    for ij = 1:endage-1
        for ia = 1:nasset 
            for iz = 1:nprod
                iap = pol_fun_ind_mat(ia,iz,ij) ;
                for izp = 1:nprod
                    F(iap,izp,ij+1) =  F(iap,izp,ij+1) + ((F(ia,iz,ij)*pival(iz,izp)))/(1+nval) ;
                end
            end
        end
    end
end