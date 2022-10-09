using Random, LinearAlgebra, Distributions, Parameters, Plots, Interpolations, Optim, Statistics

include("PS5_func_2.jl")

P, G, S = Initialize()
R = Init_R()
coeff_finder(P, G, S, R)

#this creates the scatter plots
iterate(P,G, S,R)
K_test, V_test = create_panel(P,G,S,R)
coeff, r2, xb, yb, xg, yg, pyg, pyb = coeff_estimates(P, G, S, R, K_test)
b_plot = plot(xb[:,2], yb, seriestype=:scatter, ma=0.05, title="log(K_t+1) vs log(K_t), z_t = z_b", labels="Actual",xlabel="log(K_t)", ylabel="log(K_t+1)", markersize=3)
b_plot = plot!(xb[:,2], pyb, labels="Predicted", linewidth=2)
display(b_plot)
savefig(b_plot, "bplot.png")
g_plot = plot(xg[:,2], yg, seriestype=:scatter, ma=0.05, title="log(K_t+1) vs log(K_t), z_t = z_g", labels="Actual",xlabel="log(K_t)", ylabel="log(K_t+1)", markersize=3)
g_plot = plot!(xg[:,2], pyg, labels="Predicted", linewidth=2)
display(g_plot)
savefig(g_plot, "gplot.png")

#this creates the aggregate and individual capital holding plots
ak_plot = plot(K_test[1:11000], title="Evolution of aggregate capital", labels="", xlabel="Time", ylabel="Capital")
indiv_plot = plot([V_test[86,1:11000], K_test[1:11000]], title="Selected individual's capital holding (w/ K for comp)", labels=["Agent 86" "Aggregate capital"], xlabel="Time", ylabel="Capital")
savefig(ak_plot, "akplot.png")
savefig(indiv_plot, "indiv_plot.png")
