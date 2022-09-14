using Parameters, Plots #import the libraries we want
include("02Growth_model_uncertainty.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res)
#solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
Plots.plot(k_grid, val_func, label = ["Good State" "Bad State"] ,title="Value Function")

#policy functions
Plots.plot(k_grid, pol_func, label = ["Good State" "Bad State"], title="Policy Functions")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes", label = ["Good State" "Bad State"])

println("All done!")
################################
