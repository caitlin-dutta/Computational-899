using Parameters, Plots, LinearAlgebra, Distributions, Random, Interpolations, Optim #import the libraries we want
include("PS5_Functions.jl")

@elapsed R, P, S, G = SolveModel()


@unpack k_grid = G
@unpack pf_k, pf_v, a0, a1, b0, b1, R2= R



Plots.plot(k_grid, pf_k[:,:,8,2])
