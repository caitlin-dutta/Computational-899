using Distributed
addprocs(6)
include("Ps3_conesakrueger_model.jl") #import the functions that solve our growth model

prim, res = Initialize() #initialize primitive and results structs
#res.q = 0.9942768177032469 #mkt clearer
@time Solve_model(prim, res) #solve the model!
@time Stationary_Dist(prim, res)


#market clearing
@time findq(prim, res)

xax = zeros(66)
for i =1:66
    xax[i] = i
end


@unpack val_func, pol_func, pol_func_ind, labor = res
@unpack a_grid = prim

xax = zeros(66)
for i = 1:66
    xax[i] = i
end

##############Make plots
#value function
using Plots
Plots.plot(a_grid, val_func[50,:,:], title="Value Function", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/Value_Function_50yo.png")

Plots.plot(a_grid, labor[20,:,:], title="Labor", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.plot(xax, labor[:,500,:], title="Labor", label = ["High Prod state" "Low Prod State"], legend=:bottomright)


#policy functions
Plots.plot(a_grid, pol_func[20,:,:], title="Policy Function", label = ["High Prod state" "Low Prod State"], legend=:bottomright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/Policy_Function_20yo.png")
#max a = 4.47
#mu

Plots.plot(a_grid, μ, title="μ , q = $q", label = ["Employed state" "Unemployed State"], legend=:topright)
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu.png")

Plots.plot(a_grid, mu_u, title="μ unemployed, q = $q")
Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS2/Julia_Caitlin/mu_u.png")



println("All done!")
################################
