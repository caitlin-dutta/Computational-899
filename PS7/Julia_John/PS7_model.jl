using Parameters, Plots, Random, Distributions, LinearAlgebra, Optim
include("PS7_functions.jl")

prim = Primitives(seed =64)
savefig( plot(prim.x_seq, labels="", title="True data"), "x_plot.png")
#find all requested objects for each moment specification
b1, b2, W, dg, vc, se, J = res_finder(64,1,1)
J
#print requested objects
println("b_1: ", round.(b1, digits=4))
println("b_2: ",round.(b2, digits=4))
println("W: ", map(x -> round.(x, digits=4),W))
println("∇_g: ", map(x -> round.(x, digits=4),dg))
println("Variance-covariance matrix: ", map(x -> round.(x, digits=4),vc))
println("SE: ", map(x -> round.(x, digits=4),se))
println("J: ", map(x -> round.(x, digits=4),J))

#find p values of J test:
pvals = 1 .-ccdf.(Chisq(1), J)
println("p-values: ", map(x -> round.(x, digits=4),pvals))

#find 1000 bootstrap samples and report results alongside histograms for each parameter
rho_1, sigma_1, rho_2, sigma_2 = bootstrapper(1000)
println("Mean ρ_1: ", mean(rho_1)) 
println("Mean ρ_2: ", mean(rho_2))
println("Mean σ_1: ", mean(sigma_1))
println("Mean σ_2: ", mean(sigma_2))
savefig(histogram([rho_1, rho_2],fillalpha=0.7, labels=["rho_1" "rho_2"], title="Bootstrap distribution of rho estimators" ), "rho_bootstrap.png")
savefig(histogram([sigma_1, sigma_2],fillalpha=0.7, labels=["sigma_1^2" "sigma_2^2"], title="Bootstrap distribution of sigma^2 estimators"), "sigma_bootstrap.png")

