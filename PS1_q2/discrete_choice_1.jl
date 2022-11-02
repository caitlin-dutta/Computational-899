using CSV, DataFrames, Distributions,Plots, Optim, LinearAlgebra, ForwardDiff
#Yi = 1 if β0 + Xiβ1 +εi1 > εi0


function get_data()
        cd("/Users/alexissmith/Desktop/compF22/q2/pset1")

        data = CSV.read("mortgage_performance_data.csv", DataFrame, header=true)

        Xi = hcat(data[:,"i_large_loan"], data[:,"i_medium_loan"], data[:,"rate_spread"], data[:,"i_refinance"]
        ,data[:,"age_r"], data[:,"cltv"], data[:,"dti"], data[:,"cu"], data[:,"first_mort_r"]
        ,data[:,"score_0"], data[:,"score_1"], data[:,"i_FHA"], data[:,"i_open_year2"], data[:,"i_open_year3"]
        ,data[:,"i_open_year4"], data[:,"i_open_year5"])

        Y = data[:,"i_close_first_year"]
        n = length(Y)
        X = hcat(ones(n),Xi)
        len = length(X[1,:])

        X, Y, n, len
end

#log-likelihood, score, and hessian conditional on β
function log_likelihood(n, β, X, Y,len)

        Λ = exp.((X*β))./(1 .+ exp.(X*β)) # probability
        l = Λ.^Y.*(1 .- Λ).^(1 .- Y) # to put in ML
        L= sum(log.(l))
        L

        # score
        s = zeros(n,len)
        S = zeros(len)
        for i=1:n, j=1:len
                s[i,j] = (Y[i]-Λ[i])*X[i,j]
        end
        for j = 1:len
                S[j] = sum(s[:,j])
        end
        S

        # hessian
        p = zeros(n,len,len)
        for i=1:n, j=1:len, k=1:len
                p[i,j,k] = Λ[i]*(1-Λ[i])*X[i,j]*X[i,k]
        end
        H=zeros(len,len)
        for j=1:len, k=1:len
                H[j,k] = -sum(p[:,j,k])
        end

        L, S, H
end



#solve maximum likelihood with Newton Algorithm
function newton_method(X,Y,n,len; err=10e-12)
        #second order taylor expansion
        #approximate β
        #iterate on β β_k = β_k-1 - inv(H(β_k-1))*g(β_k-1)
        β_0 = vcat(-1.0, zeros(16))
        β_1 = zeros(len)
        norm = 100
        while norm > err
                L,S,H = log_likelihood(n, β_0, X, Y,len)
                β_1 = β_0 .- inv(H)*S
                norm = sqrt(sum((β_1 .- β_0).^2))
                β_0 = β_1
        end
        β_0
end


function Solve()
        #get the data
        X, Y, n, len = get_data()
        β1 = zeros(16)
        β0 = -1.0
        β = vcat(β0, β1)


        #compute the likelihood, score, and hessian
        L,S,H = log_likelihood(n, β, X, Y,len)


        #use the optimization routines
        #just the maximum likelihood
        function LL(β)
                Λ = exp.((X*β))./(1 .+ exp.(X*β)) # probability
                l = Λ.^Y.*(1 .- Λ).^(1 .- Y) # to put in ML
                L= -sum(log.(l))
                L
        end


        #numerical derivatives
        function num_score(β)
                v = ForwardDiff.gradient(b -> LL(b), β)
                v
        end

        function num_hessian(β)
                v = ForwardDiff.hessian(b -> LL(b), β)
                v
        end

        S_num = -num_score(β) #need to flip the sign back around
        H_num = -num_hessian(β)

        #compare to BFGS and Simplex packages
        guess_init = β

        opt_bfgs = optimize(LL, guess_init, BFGS()) #this one does v well
        β_bfgs = opt_bfgs.minimizer

        opt_simplex = optimize(LL, guess_init) #this one is less good
        β_sim = opt_simplex.minimizer

        β_newton = newton_method(X,Y,n,len) #this is actually p fast


        #return results
        L, S, H, β_bfgs, β_sim, β_newton, S_num, H_num
end

L, S, H, β_bfgs, β_sim, β_newton, S_num, H_num = @elapsed Solve()
