using Distributed
addprocs(4) #import the libraries we want
@everywhere include("02Growth_model_parallel.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

##############Make plots
#value function
Plots.plot(k_grid, val_func, title="Value Function", label = ["Good state" "Bad State"])
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS1/02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions", label = ["Good state" "Bad State"])
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS1/02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Policy Functions Changes", label = ["Good state" "Bad State"])
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS1/02_Policy_Functions_Changes.png")

println("All done!")
################################
