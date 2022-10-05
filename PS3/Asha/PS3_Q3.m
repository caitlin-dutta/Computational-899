clear;
clc; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Parameters %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Life cycle characteristics
N   = 66    ;   %no of periods an agent lives for
n   = 0.011 ;   %population growth period
B   = 1     ;   %newly born agent age
R   = 46    ;   %retirement age
nj  = 66    ;   %length of life

% Tax
theta = 0.11 ;  %Labor income tax

% Utility function parameters
beta  = 0.97             ;          %discount rate
gamma = 0.42             ;          %consumption weight
sigma = 2                ;          %Coeff of RRA
cwt   = (1-sigma)*gamma  ;          %weight for retired agent

% Worker's productivity
eta     = readmatrix('ageef.txt',Delimiter="    ") ; %read age-specific productivities
eta     = eta(:,2)                ; %it's reading in the first col as NaN
zh      = 3                       ; %high productivity
zl      = 0.5                     ; %low productivity
z_grid  = [zh zl]                 ; %matrix with productivity realizations
nz      = 2                       ; %length of productivity states
zprob   = [0.2037 0.7963]         ; %matrix for P(Zhigh) and P(Zlow)
eff(1,:)= z_grid(1)*eta           ; %matrix for age-specific productivity 
eff(2,:)= z_grid(2)*eta           ; %matrix for age-specific productivity 
piHH    = 0.9261                  ; %Productivity transition matrix HH
piHL    = 1-0.9261                ; %Productivity transition matrix HL
piLL    = 0.9811                  ; %Productivity transition matrix LL
piLH    = 1-0.9811                ; %Productivity transition matrix LH
pi      = [piHH piHL;piLH piLL]   ; %put it all in a matrix

% Production
alpha = 0.36  ; %capital share
delta = 0.06  ; %depreciation


%Asset grid
na      = 500                          ; 
a_low   = 0.0001                        ;
a_max   = 100                           ;   
a_grid  = linspace(a_low,a_max,na)      ; 

%Prices
% r = 0.05   ; %interest rate
% w = 1.05   ; %wage
% b = 0.2    ; %social security benefit

% Find cohort size
mu = ones(1,N) ;
for ij = 1:N-1
    mu(ij+1) = mu(ij)./(1+n) ;
end

% Normalize mu 
denom = sum(mu,'all') ;
mu = mu./denom ;


%Guess initial K and L 
K0 = 3.4 ; 
L0 = 0.366 ; 

% Initialize errors and tolerance 
iter    = 1     ; 
err     = 10^3  ; 
maxiter = 10^3  ; 
maxerr  = 10^-3 ; 

%Iterate on K and L until convergence
while err >maxerr && iter<maxiter
    disp(iter)
    

    %%% compute new r, w and b from the new K and L %%%
    r = alpha*(K0^(alpha-1))*(L0^(1-alpha)) - delta ;
    w = (1-alpha)*(K0^(alpha))*(L0^(-alpha))    ;
    b = (theta*w*L0)/(sum(mu(R:N)))           ;


    %%% PART I: Retirees problem %%%
    [val_fun,pol_fun,pol_fun_ind] = BellmanR(R,N,a_grid,na,nz,b,r,sigma,beta,gamma); 
  
    %%% PART I: Workers problem %%%   
    [val_fun,pol_fun,pol_fun_ind,lsmat] = BellmanW(pol_fun,pol_fun_ind,val_fun,1,R-1,a_grid,na,nz,eff,pi,r,w,sigma,beta,gamma,theta) ;
    
    %%% PART II : COMPUTING STATIONARY DISTRIBUTION %%%
    F = Distribution(N,zprob,pol_fun_ind,n,na,nz,mu,pi) ;
 
    %%% PART III: AGGREGATE K AND L LOOP %%%

   
    %%% calculate k and l implied by the distribution above %%%
    
    %Initialize K and L
    Knew = 0 ;
    Lnew = 0 ;
    
    for ij = 1:N
        for iz = 1:nz
            for ia = 1:na
                Knew = Knew + F(ia,iz,ij)*a_grid(ia) ;
            end
        end
    end

    for ij = 1:R-1
        for ia = 1:na 
            for iz = 1:nz
                Lnew = Lnew + (F(ia,iz,ij)*eff(iz,ij)*lsmat(ia,iz,ij)) ;
            end
        end
    end
  
    % compute error and update guesses
    err = max(abs(Knew-K0), abs(Lnew-L0))  ; 
    
    K1 = (0.8*K0) + (0.2*Knew)             ;
    L1 = (0.8*L0) + (0.2*Lnew)             ;
    K0 = K1                                ; 
    L0 = L1                                ; 

    iter = iter+1                          ;
    disp(['K0=', num2str(K0)])             ; 
    disp(['L0=', num2str(L0)])             ;
    disp(['error=', num2str(err)])         ;
    disp(['r=', num2str(r), ' w=' num2str(w),' b=',num2str(b)])         ;

end

