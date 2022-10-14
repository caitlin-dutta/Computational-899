using Parameters, Plots, LinearAlgebra
include("PS6_func.jl")

prim, res = Init()
res.α = 2.0
res.cf = 15.0

using .MathConstants: γ #gotta include our emotional support euler constant :)

α_mat = [0.0, 1.0, 2.0]
cf_mat = [10.0, 15.0]
exit_probs = zeros(3,2, 5)
μ_mat = zeros(3,2,5)
for ai=1:3
    for ci = 1:2
        prim, res = Init()
        res.α = α_mat[ai]
        res.cf = cf_mat[ci]
        model_solve(prim, res)
        exit_probs[ai, ci, :] = res.exit_pol
        μ_mat[ai, ci, :] = res.μ
    end
end

exit_plot_10 = plot([exit_probs[1,1,:], exit_probs[2,1,:], exit_probs[3,1,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Probability of exit by firm size and alpha, c_f = 10", xlabel="Firm size", ylabel="Probability of exit")
savefig(exit_plot_10, "exitplot10.png")

exit_plot_15 = plot([exit_probs[1,2,:], exit_probs[2,2,:], exit_probs[3,2,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Probability of exit by firm size and alpha, c_f = 15", xlabel="Firm size", ylabel="Probability of exit")
savefig(exit_plot_15, "exitplot15.png")

mu_plot_10 = plot([μ_mat[1,1,:], μ_mat[2,1,:], μ_mat[3,1,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Mass of firms by size, c_f = 10", xlabel="Firm size", ylabel="Mass of firms")
savefig(mu_plot_10, "muplot10.png")

mu_plot_15 = plot([μ_mat[1,2,:], μ_mat[2,2,:], μ_mat[3,2,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Mass of firms by size, c_f = 15", xlabel="Firm size", ylabel="Mass of firms")
savefig(mu_plot_15, "muplot15.png")
#=
Bellman(prim, res, 0.739, 2.0)

free_entry(prim, res, 0.82531)

price_finder(prim, res)
invar_mu(prim, res.exit_pol, 3.5707)

market_clearing(prim, res, 0.739)


model_solve(prim, res)
=#


