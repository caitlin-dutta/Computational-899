using Parameters, Plots, LinearAlgebra
include("PS6_func.jl")

prim, res = Init()

using .MathConstants: γ #gotta include our emotional support euler constant :)

#loop over choices of alpha and fixed cost. Note that alpha = 0 indicates that the code should use the non-stochastic specification.
α_mat = [0.0, 1.0, 2.0]
cf_mat = [10.0, 15.0]
exit_probs = zeros(3,2, 5) #empty array for exit probabilities to plot
μ_mat = zeros(3,2,5) #empty invar dist array
for ai=1:3 #iterate over TIEV values (i.e. the non-stochastic case, alpha = 1, and alpha = 2)
    for ci = 1:2 #iterate over fixed costs
        prim, res = Init() 
        res.α = α_mat[ai] #set alpha parameter
        res.cf = cf_mat[ci] #set fc parameter
        model_solve(prim, res) #solve model with given specification
        exit_probs[ai, ci, :] = res.exit_pol #store exit decision
        μ_mat[ai, ci, :] = res.μ #store invariant distribution
    end
end
#plot exit probabilities for cf = 10
exit_plot_10 = plot([exit_probs[1,1,:], exit_probs[2,1,:], exit_probs[3,1,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Probability of exit by firm size and alpha, c_f = 10", xlabel="Firm size", ylabel="Probability of exit")
savefig(exit_plot_10, "exitplot10.png")

#plot exit probabilities for cf = 15
exit_plot_15 = plot([exit_probs[1,2,:], exit_probs[2,2,:], exit_probs[3,2,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Probability of exit by firm size and alpha, c_f = 15", xlabel="Firm size", ylabel="Probability of exit")
savefig(exit_plot_15, "exitplot15.png")

#plot invar dist for cf = 10
mu_plot_10 = plot([μ_mat[1,1,:], μ_mat[2,1,:], μ_mat[3,1,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Mass of firms by size, c_f = 10", xlabel="Firm size", ylabel="Mass of firms")
savefig(mu_plot_10, "muplot10.png")

#plot invar dist for cf=15
mu_plot_15 = plot([μ_mat[1,2,:], μ_mat[2,2,:], μ_mat[3,2,:]], labels=["Non-stochastic" "alpha = 1" "alpha = 2"], title="Mass of firms by size, c_f = 15", xlabel="Firm size", ylabel="Mass of firms")
savefig(mu_plot_15, "muplot15.png")


