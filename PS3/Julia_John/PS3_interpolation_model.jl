using Parameters, Plots, Interpolations, Optim
include("PS3_interpolation_func.jl")

#initialize the Primitives and Results structs, respectively
prim, res = Initialize()
param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 0.42)

model_solver(prim, res, param)
@unpack val_func, cap_pf, labor_pf = res #unpack the value and policy functions
#@elapsed vf_test = cons_opt(prim, res, param)

#part 1
vfplot50 = plot(prim.A, val_func[50, :, 1], title="Value function of retired agent at model-age 50", legend=false, xlabel="Capital", ylabel="Value")
savefig(vfplot50, "vfplot50.png")
savplot20 = plot(prim.A, [cap_pf[20, :, :] .- prim.A], title="Savings function of worker at model-age 20", legend=false, xlabel="Capital", ylabel="Savings")
savefig(savplot20, "savplot20.png")
labplot20 = plot(prim.A, labor_pf[20, :, :], title="Labor supply at model-age 20 by productivity type", labels=["High productivity" "Low productivity"], legend=:topright, xlabel="Capital", ylabel="Labor supply")
savefig(labplot20, "labplot20.png")
#part 2
F_finder(prim, res, param)

#part 3

#social security
prim, res = Initialize()
param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 0.42) #initialize benchmark model
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for equilibrium quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #compute coefficient of variation
println("Social security benchmark: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

#eliminate social security
prim, res = Initialize()
param = init_param(θ=0.0, w=1.05, r = 0.05, b = 0.0, Z = [3.0, 0.5], γ = 0.42) #model without social security has θ = 0
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for eq quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #find coefficient of variation
println("No social security: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

#no idiosyncratic risk, with social security
prim, res = Initialize()
param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [0.5, 0.5], γ = 0.42)
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param)#search for equilibrium quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #find coefficient of variation
println("No idiosyncratic risk, with SS: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

#no idiosyncratic risk, without social security
prim, res = Initialize()
param = init_param(θ=0.0, w=1.05, r = 0.05, b = 0.0, Z = [0.5, 0.5], γ = 0.42)
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param)#search for equilibrium quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #find coefficient of variation
println("No idiosyncratic risk, without SS: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

#exogenous labor supply, with social security
prim, res = Initialize()
param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 1.0)
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for equilibrium quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #find coefficient of variation
println("Exogenous labor supply, with SS: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

#exogenous labor supply, without social security
prim, res = Initialize()
param = init_param(θ=0.0, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 1.0)
k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for equilibrium quantities/parameters
welf_e = welfare(prim, res, param) #find total welfare in equilibrium
cv_e = coeff_of_var(prim, res, param) #find coefficient of variation
println("Exogenous labor supply, without SS: K = $(k_e), L = $(l_e), w = $(w_e), r = $(r_e), b = $(b_e), welfare = $(welf_e), cv = $(cv_e)")

lambda = equivalent_variation()

prop_favor = sum((lambda .>= 0) .* res.F)

prop_favor_by_age = prop_favor_age(prim, res, lambda)
age_plot = plot(prop_favor_by_age, title="Prop. of each age voting to eliminate", legend=false)
savefig(age_plot, "ageplot.png")

lambda_inelast = equivalent_variation_inelastic()
prop_favor_inelast = sum((lambda_inelast .>= 0) .* res.F)
prop_favor_by_age_in = prop_favor_age(prim, res, lambda_inelast)


age_plot_inelast = plot(1:66,[prop_favor_by_age, prop_favor_by_age_in], title="Prop. of each age voting to eliminate", legend=:topright, labels=["Benchmark" "Inelastic labor"])
savefig(age_plot_inelast, "ageplot_in.png")
