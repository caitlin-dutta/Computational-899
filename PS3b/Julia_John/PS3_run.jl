using Parameters, CSV, StatFiles, DataFrames, Kronecker, LinearAlgebra, Plots, Optim

include("PS3_func.jl")
data = Data()

ind_85 = data.mkt_index[:,1]
price_85 = data.price[ind_85]
shares_85 = data.shares[ind_85]
δ_85 = data.shares[ind_85]
δ_contraction, error_contraction = BLP_contraction_mkt(data.income, price_85, shares_85, δ_85, 0.6, 0, verbose=true)
δ_newton, error_newton = BLP_contraction_mkt(data.income, price_85, shares_85, δ_85, 0.6, 1, verbose=true)

error_plot = plot([error_contraction, error_newton, ones(length(error_contraction))], labels=["Errors, Contraction" "Errors, Contraction + Newton" "Newton threshold"], title="Error by demand inversion iteration", xlabel="Iteration", ylabel="Error (sup norm)")
savefig(error_plot, "error_plot.png")

test = λ_grid_search(data, inv(data.Z' * data.Z))
λ_ind = argmin(test[:,1])
gmm_val = test[λ_ind, 1] #234.7611
λ_opt = test[λ_ind, 2] # 0.62
GMM_plot = plot(collect(0.0:0.01:1.0), test[:,1], title="GMM objective function", xlabel="λ_p", ylabel="GMM objective function", labels="GMM objective function value")
scatter!([0.62], [234.7611], labels="Minimum")
savefig(GMM_plot, "gmmplot.png")

opt = optimize(λ -> GMM_objective(data, λ, method="two-step", λ_hat = 0.62), [0.62], LBFGS(), Optim.Options(show_trace = true, show_every = 1))
opt.minimizer
opt.minimum
#
#minimizer: 0.5639
#minimum: 163.7396


test2 = λ_grid_search(data, method_ind= "two-step", λhat = 0.62 )

gmm_2step_plot = plot(collect(0.0:0.01:1.0), [test[:,1], test2[:,1]], labels=["GMM, 2SLS" "GMM, 2-step"], title="GMM objective function", xlabel="λ_p", ylabel="GMM objective function")
scatter!([[0.62], [0.5639]], [[234.7611], [163.7396]], labels=["2SLS minimizer" "2-step minimizer"])
savefig(gmm_2step_plot, "gmm2step.png")