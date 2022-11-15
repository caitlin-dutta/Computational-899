using StatFiles, CSV, Parameters, DataFrames, ForwardDiff, Optim, LatexPrint, BenchmarkTools, Plots, Random, Distributions, HaltonSequences, NLSolversBase

include("PS2b_func.jl")

data = Data()

@unpack T = data

#initial guess
θ_init = vcat([0,-1,-1], zeros(15), 0.3, 0.5)

#belapsed runs the function a bunch of times and determines the minimum amount of time the function took to execute
@belapsed ll_q, l_vec_q = log_like_quad_verbose(Data(), θ_init)
@belapsed ll_ghk, l_vec_ghk = log_like_GHK(Data(), θ_init,100)
@belapsed ll_ar, l_vec_ar = log_like_AR(Data(), θ_init,100)

#find the groups corresponding to each outcome
T1 = (T .== 1)
T2 = (T .== 2)
T3 = (T .== 3)
T4 = (T .== 4)

like_types = ["Quadrature", "GHK", "Accept_Reject"]
#compute log likelihoods for each type, create plots, and save them
likelihood_comparison(Data(), like_types, θ_init, 100)

#find optimal parameter vector; reaches convergence in around 150 iterations. Unfortunately it took around 36 minutes (even with the speed improvements of autodiff)
opt = optimize(θ -> -log_like_quad(Data(), θ), θ_init, LBFGS(), Optim.Options(show_trace = true, show_every = 1, iterations=500, g_tol=1e-3); autodiff=:forward)
#find optimal parameter vector
θ_opt = opt.minimizer
#make a nice little latex-friendly array
lap(round.(θ_opt, digits=3))

#find log likelihood at θ_opt 
ll_opt, ll_v_opt = log_like_quad(Data(), θ_opt)
opt_hist = histogram([ll_v_opt[T1], ll_v_opt[T2], ll_v_opt[T3], ll_v_opt[T4]], labels=["T = 1" "T = 2" "T = 3" "T = 4"], title="Histogram of likelihoods at optimal θ", legend=:topleft)
savefig(opt_hist, "opt_hist.png")

#scatter plot comparing likelihoods under θ_0 to θ_opt
scatter_comp = plot([test_0[T1], test_0[T2], test_0[T3], test_0[T4]], [ll_v_opt[T1], ll_v_opt[T2], ll_v_opt[T3], ll_v_opt[T4]], alpha = 0.1,seriestype=:scatter, labels=["T = 1" "T = 2" "T = 3" "T = 4"],legend=:topleft)
plot!([0.0, 0.8], [0.0, 0.8], l=2, labels="45-degree line", xlabel="Likelihood under θ_0", ylabel="Likelihood under θ_opt", title="Comparison of likelihoods under θ_opt vs θ_0")
savefig(scatter_comp, "scatter_comp.png")