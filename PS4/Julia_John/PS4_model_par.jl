using Distributed
addprocs(6)

@everywhere using Parameters, Plots, Interpolations, Optim, SharedArrays
@everywhere include("PS4_func_par.jl")

#initialize the Primitives and Results structs, respectively
@everywhere prim, res = Initialize()


@everywhere tg = 80 #number of time periods
k_path, l_path = path_finder(prim, res, tg) #find the transition paths for capital and labor
r_path, w_path = price_paths(k_path, l_path) #find the resulting paths of interest rates and wages

#plot and save each series
rplot = plot([r_path, fill(r_path[tg+1], tg+1)], title="Evolution of interest rate", legend=:topright, labels=["Transition path" "New steady state r"], xlabel = "t", ylabel="Interest rate")
wplot = plot([w_path, fill(w_path[tg+1], tg+1)], title="Evolution of wage", legend=:bottomright, labels=["Transition path" "New steady state w"],  xlabel = "t", ylabel="Wage")
kplot = plot([k_path, fill(k_path[tg+1], tg+1)], title="Evolution of capital", legend=:bottomright, labels=["Transition path" "New steady state K"],  xlabel = "t", ylabel="Capital")
lplot = plot([l_path, fill(l_path[tg+1], tg+1)], title="Evolution of effective labor supply", legend=:topright, labels=["Transition path" "New steady state L"],  xlabel = "t", ylabel="Effective labor supply")
savefig(kplot, "kplot.png")
savefig(rplot, "rplot.png")
savefig(wplot, "wplot.png")
savefig(lplot, "lplot.png")

#make a csv too :) 
CSV.write("output.csv", (k = k_path, l = l_path, r = r_path, w = w_path))

#find consumption equivalent variation and the time zero stationary distribution 
lambd, gamm = equivalent_variation_bench(vcat([0.11],  fill(0.0, tg)), k_path, l_path)
#find the measure of CE for each age
ce_meas_age = ce_meas_func(prim, lambd, gamm)
#plot measure of CE by age
ce_meas_age_pl = plot(ce_meas_age, title="Measure of CE by model age", legend=false)
#find the average CE for each age
ce_avg_age = ce_avg_a(prim, lambd, gamm)
#plot average CE
ce_avg_age_pl = plot(ce_avg_age, title="Average CE by model age", legend=false)
#save them!!!
savefig(ce_meas_age_pl, "cemeasage.png")
savefig(ce_avg_age_pl, "ceavgage.png")

prop_favor(prim, lambd, gamm) #find proportion of agents who prefer elimination
#
#11.45 percent of the population votes in favor w transition path :)

##### Part 2 #####

@everywhere tg2 = 110 
#find transition paths for capital and labor given abolition at t = 21
k_path_2, l_path_2 = path_finder(prim, res, tg2, vcat(fill(0.11, 21) , fill(0.0, tg2-20)))
#find implied transition paths of r and w
r_path_2, w_path_2 = price_paths(k_path_2, l_path_2)

#plot and the relevant series
rplot2 = plot([r_path_2, fill(r_path_2[tg2+1], tg2+1)], title="Evolution of interest rate (t = 21 phaseout)", legend=:topright, labels=["Transition path" "New steady state"])
wplot2 = plot([w_path_2, fill(w_path_2[tg2+1], tg2+1)], title="Evolution of wage (t = 21 phaseout)", legend=:bottomright, labels=["Transition path" "New steady state"])
kplot2 = plot([k_path_2, fill(k_path_2[tg2+1], tg2+1)], title="Evolution of capital (t = 21 phaseout)", legend=:bottomright, labels=["Transition path" "New steady state"])
lplot2 = plot([l_path_2, fill(l_path_2[tg2+1], tg2+1)], title="Evolution of effective labor supply (t = 21 phaseout)", legend=:bottomright, labels=["Transition path" "New steady state"])
savefig(kplot2, "kplot_21.png")
savefig(rplot2, "rplot_21.png")
savefig(wplot2, "wplot_21.png")
savefig(lplot2, "lplot_21.png")

#make a CSV!
using CSV
CSV.write("output_transition.csv", (k = k_path_2, l = l_path_2, r = r_path_2, w = w_path_2))

#pt 2: T = 50 didn't work, increased to 110 lol

#find equivalent variation and time zero stationary distribution
lambd, gamm = equivalent_variation_bench(vcat(fill(0.11, 21) , fill(0.0, tg2-20)), k_path_2, l_path_2)
#find and plot measure of CV across age
ce_meas_age = ce_meas_func(prim, lambd, gamm)
ce_meas_age_pl = plot(ce_meas_age, title="Measure of CE by model age", legend=false)
#find average CV across age
ce_avg_age = ce_avg_a(prim, lambd, gamm)
ce_avg_age_pl = plot(ce_avg_age, title="Average CE by model age", legend=false)
savefig(ce_meas_age_pl, "cemeasage_21.png")
savefig(ce_avg_age_pl, "ceavgage_21.png")
#find proportion which favor elimination at t = 21
prop_favor(prim, lambd, gamm)

