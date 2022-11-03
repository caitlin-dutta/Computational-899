

include("Ps6_hopenhayn_model.jl") #import the functions that solve our growth model
prim, res = Initialize()

for α = (0.0, 1.0, 2.0)
    for cf = (10.0, 15.0)
        mkt_clearing(α, cf)
        Plots.plot(prim.s_grid, res.pol_func, title="Exit Policy Function, Alpha =  $α CF = $cf", legend=:none)
        Plots.savefig("C:/Users/caitl/OneDrive/Documents/Second_year/Computational 899/PS6/Caitlin/Exit_$α $cf.png")
    end
end


##############Make plots
#value function
using Plots
Plots.plot(prim.s_grid, res.pol_func, title="Exit Policy Function, Alpha =  $α CF = $cf", legend=:none)
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
