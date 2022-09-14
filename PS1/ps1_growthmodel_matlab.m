% PROGRAM NAME: 387vfigrowth.M
% This program generates the value function and decision rules for
% a stochastic growth model.
% Date: 2/17/03
clear;
clc;
tic; 

% PARAMETERS
b=.99; %discount factor 
d=0.025; %depreciation rate
a=.36; %capital share

% ASSET VECTOR
klb=0.01; %lower bound of grid points
inc=0.025; %increments
kub=45;%upper bound of grid points
k=[klb:inc:kub];% asset (row) vector
N=size(k);
N=N(1,2);
c1=ones(N,1); % column vector of ones
K=c1*k; % rows are k, cols are k'

% TRANSITION MATRIX
pi = [0.977, 0.023; 0.07, 0.926];
zeta = [1.25; 0.2] ;

% CALCULATE CONS FOR EACH POSSIBLE STATE
cs=zeros(N,N,2); %rows are k, cols are k', 3rd dim is productivity shock
cs(:,:,1)=zeta(1)*K'.^a-(K-K'*(1-d));%cons
cs(:,:,2)=zeta(2)*K'.^a-(K-K'*(1-d));%cons
  
% % TABULATE CURRENT RETURN (UTILITY) FUNCTION
% cs=zeros(N,N); %rows are k, cols are k'
% cs=K'.^a-(K-K'*(1-d));%cons
is=find(cs<0);
cs(is)=0;
us=log(cs);
t=isinf(us);
j=find(t==1);
us(j)=-realmax;

% TABULATE INITIAL VALUE FUNCTION GUESS
visr=squeeze(us(:,1,:))'; %choose utility associated with k'=0 in the good state

pcntol=1; %tolerance for value function iteration
n=1; %if want to run vfi for a set number of iterations
while pcntol >.0001
   vis(:,:,1)=c1*visr(1,:); %generates future value function matrix from above row vector
   vis(:,:,2)=c1*visr(2,:); %generates future value function matrix from above row vector
   
   wis(:,:,1)=us(:,:,1) + b*vis(:,:,1).*pi(1,1) + b*vis(:,:,2).*pi(1,2) ; 
   wis(:,:,2)=us(:,:,2) + b*vis(:,:,1).*pi(2,1) + b*vis(:,:,2).*pi(2,2) ;
   
   %CHOOSE HIGHEST VALUE (ASSOCIATED WITH k' CHOICE)
   [vsr(1,:),I(1,:)]=max(wis(:,:,1)'); %since max gives highest element in each column of a matrix
   [vsr(2,:),I(2,:)]=max(wis(:,:,2)'); %since max gives highest element in each column of a matrix
   vsr = squeeze(vsr) ;
   n=n+1;
   tol=max(max(abs(vsr-visr))); 
   pcntol=tol/abs(vsr(1,N));
   visr=vsr; %update value functions
    
end
toc

save 387vdr vsr I k;
save 387parm b a d N inc klb kub;

plot1 = tiledlayout(3,1);
nexttile 
plot(k,vsr) % plot value function
title('Value Function')
legend('Z = 1.25','Z = 0.2', 'Location','southeast')

nexttile
plot(k,k([I])') % plot policy function
title('Policy Function')
legend('Z = 1.25','Z = 0.2', 'Location','northwest')

nexttile
plot(k,k([I])-d*k) %plot change in the decision rule
title("Savings (K' - {\delta}K)")
legend('Z = 1.25','Z = 0.2', 'Location','northwest')


saveas(plot1,'ps1_matlab_figures.png','png')
