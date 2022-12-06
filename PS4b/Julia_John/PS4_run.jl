using Parameters, CSV, DataFrames, LinearAlgebra, Plots, Optim, LatexPrint

include("PS4_functions.jl")

#emotional support Euler constant
using .MathConstants: γ

prim = Primitives()
res = Initialize(prim.n_grid)

bellman_iteration(prim, res)

lap(hcat(prim.state_space, round.( res.v_bar, digits=3)))

ccp_est(prim)
ccp_computation(prim, res, -4.0)
res.v_bar_p
vf_plot = plot([res.v_bar, res.v_bar_p], legend=:bottomright,labels=["Expected value function" "CCP Implied value function (1 iteration)"])
savefig(vf_plot, "vfplot.png")
vbp_0 = res.v_bar_p
ccp_iteration(prim, res, -4.0)
plot([res.v_bar, res.v_bar_p], legend=:bottomright,labels=["Expected value function" "CCP Implied value function (iterated)"])

lap(hcat(prim.state_space, round.(res.v_bar, digits=3), round.(vbp_0, digits=3), round.( res.v_bar_p, digits=3)))

plot((res.v_bar - res.v_bar_p) ./ (res.v_bar))

log_likelihood(prim, res, -4.0)

function ll_plot(x)
    λ = x[1]
    -log_likelihood(prim, res, λ)
end

opt = optimize(x -> ll_plot(x) , [-4.0], LBFGS())
opt.minimizer
opt.minimum

lambda_range = collect(-10:0.1:0)
gr = zeros(length(lambda_range))
for (i,lambda) in enumerate(lambda_range)
    gr[i] = -ll_plot(lambda)
end
ll_opt = plot(lambda_range, gr, labels="Log likelihood function", legend=:bottomleft)
scatter!([round.(opt.minimizer, digits=2)], [-opt.minimum], labels="Optimal parameter")
savefig(ll_opt, "llopt.png")