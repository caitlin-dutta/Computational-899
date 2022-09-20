using Distributed
addprocs(6)
include("Ps2_huggett_model_par.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack a_grid, q = prim

##############Make plots
#value function
Plots.plot(a_grid, val_func, title="Value Function, q = $q", label = ["Employed state" "Unemployed State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/Value_Functions.png")

#policy functions
Plots.plot(a_grid, pol_func, title="Policy Functions, q = $q", label = ["Employed state" "Unemployed State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/Policy_Functions.png")


println("All done!")
################################
