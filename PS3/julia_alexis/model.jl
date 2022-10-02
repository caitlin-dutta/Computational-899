using Parameters, Plots, LinearAlgebra, SharedArrays #import the libraries we want
include("ps3.jl") #import the functions that solve our growth model

@elapsed prim, res = SolveModel()
@unpack val_func, pol_func, l_opt, F, mu, w, r, b = res
@unpack a_grid = prim


#plots
Plots.plot(a_grid, val_func[:,1,50], title="value function for a 50 y.o. retiree")
Plots.plot(a_grid, pol_func[:,1,50], title="policy function for a 50 y.o. retiree")

Plots.plot(a_grid, val_func[:,:,20], label = ["high productivity" "low productivity"], title="value function for a 20 y.o. worker")
Plots.plot(a_grid, pol_func[:,:,20], label = ["high productivity" "low productivity"], title="policy function for a 20 y.o. worker")
Plots.plot(a_grid, pol_func[:,:,20].-a_grid, label = ["high productivity" "low productivity"], title="policy function for a 20 y.o. worker")

Plots.plot(a_grid, l_opt[:,:,20], label = ["high productivity" "low productivity"], title="labor supply choice for a 20 y.o. worker")
