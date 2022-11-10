#Comp 899 PS6 Caitlin Dutta
#keyword-enabled structure to hold model primitives
using Parameters, Plots, SharedArrays, LinearAlgebra, DataFrames, StatFiles
function load_data()
    data = DataFrame(load("C:/Users/caitl/OneDrive/Documents/Second_year/Computational 899/PS1b/Caitlin/Mortgage_performance_data.dta"))
    #keeping vars of interest i_close_first_year i_large_loan i_medium_loan rate_spread i_refinance age_r cltv dti
    #cu first_mort_r score_0 score_1 i_FHA i_open_year2-i_open_year5
    n = length(data[:,"i_large_loan"])
    Xi = hcat(data[:,"i_large_loan"], data[:,"i_medium_loan"], data[:,"rate_spread"], data[:,"i_refinance"]
        ,data[:,"age_r"], data[:,"cltv"], data[:,"dti"], data[:,"cu"], data[:,"first_mort_r"]
        ,data[:,"score_0"], data[:,"score_1"], data[:,"i_FHA"], data[:,"i_open_year2"], data[:,"i_open_year3"]
        ,data[:,"i_open_year4"], data[:,"i_open_year5"])
    X = hcat(ones(n), Xi)
    K = length(X[2,:]')
    Y = data[:, "i_close_first_year"]
end

function log_likelihood(β, X, Y)
    lam = exp.(X * β) ./ (1 .+ exp.(X * β)) #beta is just one number
    inner = lam.^Y .* (1 .- lam).^(1 .- Y)
    L = sum(log.(inner))

    s = zeros(1, K)
    for k = 1:K
    for i = 1:length(Y)
        s[1,k] += (Y[i] - lam[i,k])*X[i,k]
    end
    end
    s

    hessian = zeros(K, K)
    for i = 1:n
        hessian .-= lam[i,:] * (1 .- lam[i,:])*(X[i,:]*X[i,:]')
    end

    L, score, hessian
end


#Newton Method
function Newton(β_0, )
    #second order taylor expansion
    #L(β) ≈L(β0) + g (β0)(β −β0) + 1/2 (β −β0)′H (β0)(β −β0)
    tol = 1e-4
    error = 100
    β = β_0
    iter = 0
    while error > tol
        iter += 1
        println("Iteration ", iter, " error is ", error)
        L, S, H = log_likelihood(β, X, Y)
        β_new = zeros(1,14)
        β_new = β .- inv(H)*S'
        error = maximum(β_new - β)
    end
