using Distributed
addprocs(6)

@everywhere using Parameters, Plots, SharedArrays
@everywhere include("PS2_functions.jl")

#initialize the Primitives and Results structs, respectively
@everywhere prim, res = Initialize()
@everywhere res.q = 0.99427
@elapsed solve_vf(prim,res) #solve the model using functions in the included file


@unpack val_func, pol_func, pol_func_ind = res #unpack the value and policy functions
@unpack A = prim #unpack the capital grid for plotting

#part a
pfplot = plot(A, [pol_func, A], legend =:bottomright, labels=["g(a,e)" "g(a,u)" "45-degree line"], title="Policy function g(a,s) across a,s")
a_diff = pol_func .- A
pfdiffplot = plot(A, [a_diff, zeros(prim.na)], legend=:bottomright, labels=["g(a,e) - a" "g(a,u) - a" "g(a,s) - a = 0"], title="Difference between g(a, s) - a across a,s")
savefig(pfplot, "pfplot.png")
savefig(pfdiffplot, "pfdiffplot.png")
a_hat_i = findfirst(x -> x <0, a_diff)
a_hat = A[a_hat_i]

#part b
market_clearing(prim, res)
#eq price found above using market_clearing function is given by 0.99427
wealth_d = wealth_dist(prim, res)
wealth_plot = plot(A, wealth_d, legend=:topright, labels=["Employed" "Unemployed"], title="Wealth distribution by a,s")
savefig(wealth_plot, "wealthplot.png")

#part c
xg, lc = lorenz_curve(prim, res, wealth_d)
lorenz_plot = plot(xg, [lc xg], legend=:bottomright, labels=["Lorenz curve" "45-degree line"], title="Lorenz curve")
savefig(lorenz_plot, "lorenzplot.png")
gini_finder(xg, lc)


#part d
lambda= welfare_comparison(prim, res)
lambda_plot = plot(prim.A, lambda, labels=["Employed" "Unemployed"], title="Consumption Equivalents by employment status", legend=:topright)
savefig(lambda_plot, "lambdaplot.png")

W_INC = sum(res.μ .* res.val_func)
W_G = sum(lambda .* res.μ)

prop_favor = sum((lambda .>= 0) .* res.μ)