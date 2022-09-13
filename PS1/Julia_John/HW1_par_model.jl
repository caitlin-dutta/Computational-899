using Distributed

addprocs(6)

@everywhere using Parameters, Plots, SharedArrays
@everywhere include("HW1_par_func.jl")

#initialize the Primitives and Results structs, respectively
@everywhere prim, res = Initialize()
@elapsed solve_model(prim,res) #solve the model using functions in the included file

@unpack val_func, pol_func = res #unpack the value and policy functions
@unpack k_grid = prim #unpack the capital grid for plotting

#println(res.val_func)

#plot the value and policy functions for each value of productivity
vfplot = Plots.plot(prim.k_grid, res.val_func, title="Value Functions (Julia)", legend=:bottomright, labels=["High Productivity" "Low Productivity"]) #plot value function
pfplot = Plots.plot(prim.k_grid, res.pol_func, title="Policy Functions (Julia)", legend=:bottomright, labels=["High Productivity" "Low Productivity"]) #plot value function

#find the difference between the high productivity value function and the low productivity value function, then plot the result
netvf = res.val_func[:,1] - res.val_func[:,2]
netvfplot = Plots.plot(prim.k_grid, netvf, title="Difference between high and low Z value functions (Julia)", legend=false)

#find the difference between the next period level of capital and the current level of capital for each productivity type and plot
net_k_pf = res.pol_func .- prim.k_grid
net_pfplot = Plots.plot(prim.k_grid, net_k_pf, title="Change in decision rule, by K (Julia)", legend=:bottomleft, labels=["High Productivity" "Low Productivity"])

#display and save the aforementioned plots
display(vfplot)
display(pfplot)
display(netvfplot)
display(net_pfplot)
savefig(vfplot, "vfplot.png")
savefig(pfplot, "pfplot.png")
savefig(netvfplot, "netvfplot.png")
savefig(net_pfplot, "netpfplot.png")