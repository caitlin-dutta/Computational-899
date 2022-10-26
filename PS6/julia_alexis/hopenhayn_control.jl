using Parameters, Plots, LinearAlgebra, Distributions, Random, Interpolations, Optim
include("hopenhayn_functions.jl")

@elapsed prim, res = SolveModel()
