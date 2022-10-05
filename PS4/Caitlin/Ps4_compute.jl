

include("Ps3_conesakrueger_cv_model.jl") #import the functions that solve our growth model
@time prim, res, welfare = Solve_model() #solve the model!

#=using Distributed
addprocs(6)
include("Ps3_conesakrueger_model.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim,res)
@time create_F(prim, res)
=#

xax = zeros(66)
for i =1:66
    xax[i] = i
end

@unpack val_func, pol_func, pol_func_ind, labor, budg, F, w, r, b =res
@unpack a_grid = prim

#results:
#w ss: K = 3.36430, L =0.34373, w = 1.455, r = 0.0236, b = 0.226, welf = -35.76
#wo ss: K = 4.6205887, L = 0.365495, w = 1.5950, r = 0.01100, b = 0.0, welf = -37.3


##############Make plots
#value function
using Plots
Plots.plot(a_grid, val_func[50,:,:], title="Value Function 50yo", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/Value_Function_50yo.png")

Plots.plot(a_grid, labor[20,:,:], title="Labor 20yo", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/Labor_Function_20yo.png")

#policy functions
Plots.plot(a_grid, pol_func[20,:,:], title="Policy Function 20yo", label = ["High Prod state" "Low Prod State"], legend=:topright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/Policy_Function_20yo.png")
#max a = 4.47
#mu


Plots.plot(a_grid, F[20,:,:], title="Distribution", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.plot(xax, labor[:,500,:], title="Labor", label = ["High Prod state" "Low Prod State"], legend=:bottomright)


Plots.plot(a_grid, μ, title="μ , q = $q", label = ["Employed state" "Unemployed State"], legend=:topright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu.png")

Plots.plot(a_grid, mu_u, title="μ unemployed, q = $q")
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu_u.png")



println("All done!")
################################
