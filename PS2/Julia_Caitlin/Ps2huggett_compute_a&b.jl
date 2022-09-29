using Distributed
addprocs(6)
include("Ps2_huggett_model_par.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
res.q = 0.9942768177032469 #mkt clearer
@time Solve_model(prim, res) #solve the model!
@time Stationary_Dist(prim, res)


#market clearing
@time findq(prim, res)


@unpack val_func, pol_func, pol_func_ind, μ, q = res
mu_e = μ[:,1]
mu_u = μ[:,2]
@unpack a_grid = prim

##############Make plots
#value function
using Plots
Plots.plot(a_grid, val_func, title="Value Function, q = $q", label = ["Employed state" "Unemployed State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/Value_Functions.png")

#policy functions
Plots.plot(a_grid, pol_func, title="Policy Functions, q = $q", label = ["Employed state" "Unemployed State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/Policy_Functions.png")
#max a = 4.47
#mu

Plots.plot(a_grid, μ, title="μ , q = $q", label = ["Employed state" "Unemployed State"], legend=:topright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu.png")

Plots.plot(a_grid, mu_u, title="μ unemployed, q = $q")
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu_u.png")



println("All done!")
################################
