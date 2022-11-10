#Comp 899 PS6 Caitlin Dutta
#keyword-enabled structure to hold model primitives
using Parameters,Plots, SharedArrays, LinearAlgebra, DataFrames, StatFiles, ForwardDiff, Optim
function load_data()
    df = DataFrame(
        load(
            "C:/Users/caitl/OneDrive/Documents/Second_year/Computational 899/PS1b/Caitlin/Mortgage_performance_data.dta",
        ),
    )
    #keeping vars of interest i_close_first_year i_large_loan i_medium_loan rate_spread i_refinance age_r cltv dti
    #cu first_mort_r score_0 score_1 i_FHA i_open_year2-i_open_year5
    n = length(df[:, "i_large_loan"])
    Xi = select(
        df,
        [
            :i_large_loan,
            :i_medium_loan,
            :rate_spread,
            :i_refinance,
            :age_r,
            :cltv,
            :dti,
            :cu,
            :first_mort_r,
            :score_0,
            :score_1,
            :i_FHA,
            :i_open_year2,
            :i_open_year3,
            :i_open_year4,
            :i_open_year5,
        ],
    )
    Xi = identity.(Array(Xi))
    X = hcat(ones(n), Xi)
    K = length(X[2, :]')
    Y = df[:, "i_close_first_year"]
    n, K, X, Y
end

function lambda(a)
    lam = exp(a) / (1 + exp(a))
    lam
end

function log_likelihood(β, X, Y)

    L = sum(log.(lambda.(X * β) .^ (Y) .* (1 .- lambda.(X * β)) .^ (1 .- Y)))
    L

    s = zeros(K)
    s = sum((Y .- lambda.(X * β)) .* X, dims = 1)
    s

    h = zeros(K, K)
    for i = 1:n
        h .-=
            lambda(X[i, :]' * β) *
            (1 .- lambda(X[i, :]' * β)) *
            (X[i, :] * X[i, :]') #iteratively subtract each observation's contribution to the Hessian
    end
    h

    L, s, h
end

#finding numerical gradient and hessian

function LL(β)
    L = sum(log.(lambda.(X * β) .^ (Y) .* (1 .- lambda.(X * β)) .^ (1 .- Y)))
    L
end

function num_condition(β)
    n_s = ForwardDiff.gradient(b -> LL(b), β)
    n_h = ForwardDiff.hessian(b -> LL(b), β)
    n_s, n_h
end

β = vcat(-1.0, zeros(16))
β_init = vcat(-1.0, zeros(16))

#Conmparison, they are the same.
#maximum(abs.(S' - n_s)) = 3.637978807091713e-12
#maximum(abs.(H - n_h)) = 6.810296326875687e-9

#Newton Method
function Newton(β_init, X, Y)
    L, S, H = log_likelihood(β_init, X, Y)
    β_old = vec(inv(H) * S')
    β_new = zeros(K)
    tol = 1e-10
    error = maximum(abs.(β_new - β_old))
    iter = 0

    while error > tol
        iter += 1
        L, S, H = log_likelihood(β_old, X, Y)
        β_new = β_old - vec(inv(H) * S')
        error = maximum(abs.(β_new - β_old))
        println("Iteration ", iter, " error is ", error," beta new ", β_new[1], " beta old is ", β_old[1])
        β_old = β_new
    end
    β_new
end

function solve()
    #get data
    n, K, X, Y = load_data()
    β_init = vcat(-1.0, zeros(16))

    L, S, H = log_likelihood(β_init, X, Y)

    n_s, n_h = num_condition(β_init)
    #error for score
    e_s = maximum(abs.(S' - n_s))
    #error for hessian
    e_h = maximum(abs.(H - n_h))

    #Newton
    β_newton = Newton(β_init, X, Y)

    #Try BFGS
    optim_BFGS = optimize(β -> -LL(β), β_init, BFGS(), Optim.Options(f_abstol=1e-12))
    β_BFGS = optim_BFGS.minimizer

    #Try Nelder-Mead
    optim_simplex = optimize(β -> -LL(β), β_init, Optim.Options(iterations=50000, f_abstol=1e-12))
    β_simplex = optim_simplex.minimizer
end
