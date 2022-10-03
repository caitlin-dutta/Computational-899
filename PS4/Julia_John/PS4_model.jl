using Parameters, Plots, Interpolations, Optim, CSV
include("PS4_functions.jl")

#initialize the Primitives and Results structs, respectively
prim, res = Initialize()
#θ_seq_SS = fill(0.11, 31)
#param = init_param(θ_seq = θ_seq_SS, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 1.0, t =0, T = 30)
#k_0, l_0, w_0, r_0, b_0 = kl_search(prim, res, param) #search for equilibrium quantities/parameters
#welf_0 = welfare(prim, res, param) #find total welfare in equilibrium
#cv_0 = coeff_of_var(prim, res, param) #compute coefficient of variation
#Γ_0 = res.Γ
#println("Social security benchmark: K = $(k_0), L = $(l_0), w = $(w_0), r = $(r_0), b = $(b_0), welfare = $(welf_0), cv = $(cv_0)")

#prim, res = Initialize()
#θ_seq_T = vcat([0.11],  fill(0.0, 30))
#param = init_param(θ_seq = θ_seq_T, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 1.0, t =30, T = 30)
#k_T, l_T, w_T, r_T, b_T = kl_search(prim, res, param) #search for equilibrium quantities/parameters
#welf_T = welfare(prim, res, param) #find total welfare in equilibrium
#cv_T = coeff_of_var(prim, res, param) #compute coefficient of variation
#Γ_T = res.Γ
#println("Social security benchmark: K = $(k_T), L = $(l_T), w = $(w_T), r = $(r_T), b = $(b_T), welfare = $(welf_T), cv = $(cv_T)")

#k_path_30 =path_finder(prim, res, param, k_0, 30)

#k_path_50 =path_finder(prim, res, 30)
#plot(k_path_50[1])


#NOTE: clean up variable names

#k_path_50_test, l_path_50_test = path_finder(prim, res, 80)



tg = 80
k_path, l_path = path_finder(prim, res, tg)


r_path, w_path = price_paths(k_path, l_path)
rplot = plot([r_path, fill(r_path[tg+1], tg+1)], title="Evolution of interest rate", legend=:topright, labels=["Transition path" "New steady state r"], xlabel = "t", ylabel="Interest rate")
wplot = plot([w_path, fill(w_path[tg+1], tg+1)], title="Evolution of wage", legend=:bottomright, labels=["Transition path" "New steady state w"],  xlabel = "t", ylabel="Wage")
kplot = plot([k_path, fill(k_path[tg+1], tg+1)], title="Evolution of capital", legend=:bottomright, labels=["Transition path" "New steady state K"],  xlabel = "t", ylabel="Capital")
lplot = plot([l_path, fill(l_path[tg+1], tg+1)], title="Evolution of effective labor supply", legend=:topright, labels=["Transition path" "New steady state L"],  xlabel = "t", ylabel="Effective labor supply")
savefig(kplot, "kplot.png")
savefig(rplot, "rplot.png")
savefig(wplot, "wplot.png")
savefig(lplot, "lplot.png")
CSV.write("output.csv", (k = k_path, l = l_path, r = r_path, w = w_path))

#T = 30: 4.552 vs target of 4.62597
#T = 50: 4.622 target of 4.62597



lambd, gamm = equivalent_variation_bench(vcat([0.11],  fill(0.0, tg)), k_path, l_path)
ce_age = ce_avg(prim, lambd, gamm)
ceavg = plot(ce_age, title="Average CE by model age", legend=false)
savefig(ceavg, "ceavg2.png")
prop_favor(prim, lambd, gamm)
#
#11.45 percent of the population votes in favor w transition path :)

pk_path_2, l_path_2 = path_finder(prim, res, 110, vcat(fill(0.11, 21) , fill(0.0, 90)))

r_path_2, w_path_2 = price_paths(k_path_2, l_path_2)
rplot2 = plot(r_path_2, title="Evolution of interest rate (t = 21 phaseout)", legend=false)
wplot2 = plot(w_path_2, title="Evolution of wage (t = 21 phaseout)", legend=false)
kplot2 = plot(k_path_2, title="Evolution of capital (t = 21 phaseout)", legend=false)
savefig(kplot2, "kplot2.png")
savefig(rplot2, "rplot2.png")
savefig(wplot2, "wplot2.png")
#pt 2: T = 50: 10.35369 vs goal of 10.49403
#T = 80: 10.49378 vs 10.49403
#T = 110: 10.49403 :)

