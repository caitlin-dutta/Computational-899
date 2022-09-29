@with_kw struct Primitives
    β::Float64 = 0.97 #discount factor
    Π::Array{Float64,2} = [0.9261 0.0739; 0.0189 0.9811 ] #transition matrix
    Z_freq::Vector{Float64} = [0.2037, 0.7963] #ergodic distribution of initial productivity
    η_arr ::Vector{Float64} = [0.59923239,0.63885106 ,0.67846973 ,0.71808840 ,0.75699959 ,0.79591079 ,0.83482198 ,0.87373318 ,0.91264437 ,0.95155556 ,0.99046676 ,0.99872065 ,1.0069745 ,1.0152284 ,1.0234823 ,1.0317362 ,1.0399901 ,1.0482440 ,1.0564979 ,1.0647518 , 1.0730057 ,1.0787834 ,1.0845611 ,1.0903388 , 1.0961165 , 1.1018943 , 1.1076720 , 1.1134497 , 1.1192274 , 1.1250052 , 1.1307829 ,1.1233544 ,1.1159259 ,1.1084974 ,1.1010689 ,1.0936404 ,1.0862119 ,1.0787834 ,1.0713549 ,1.0639264,1.0519200,1.0430000,1.0363000,1.0200000,1.0110000] #age efficiency profile (sorry!)
    n::Float64 = 0.011 #population growth rate
    N::Int64 = 66 #age of death :(
    jr::Int64 = 46 #retirement age
    σ :: Float64 = 2.0 #CRRA parameter
    α::Float64  = 0.36  #capital share in production function
    δ::Float64 = 0.06 #depreciation rate of capital
    na ::Int64 = 1000
    a_max = 100.0
    a_min = 0.0001
    a_range = range(a_min, length=na, stop=a_max)
    A :: Array{Float64} = collect(a_range) #capital grid
end

mutable struct Param
    θ :: Float64 #social security tax rate
    w :: Float64 #wage rate
    r ::Float64 #interest rate
    b :: Float64 #retirement benefits
    Z :: Vector{Float64}#idiosyncratic productivity shock vector
    γ  :: Float64  #weight on consumption
end

mutable struct Results
    val_func :: Array{Float64,3} #value function struct
    cap_pf :: Array{Float64,3} #capital policy function struct
    labor_pf :: Array{Float64,3} #labor supply policy function struct
    μ :: Vector{Float64} #size of cohorts by age
    F :: Array{Float64,3} #ss dist of agents over age, prod, and assets
    K :: Float64 #aggregate capital supply
    L :: Float64 #aggregate labor supply
    wealth_d :: Array{Float64, 3} #wealth distribution by age
end

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.N, prim.na, 2) #initial value function guess; depends on age, asset, and productivity 
    cap_pf = zeros(prim.N, prim.na, 2) #initial policy function guess
    labor_pf = ones(prim.N, prim.na,2) #initial policy function guess index
    F = ones(prim.N, prim.na, 2) #create initial guess for F, will change later
    μ = mu_finder(prim, 1.0) #find size of cohorts, will sum to one
    K = 3.36#3.6239 #aggregate capital supply guess informed by model output with starting parameters
    L = 0.343#0.3249 #aggregate labor supply guess informed by model output with starting parameters 
    wealth_d = ones(prim.N, prim.na, 2) #initial wealth distribution, will be changed later
    res = Results(val_func, cap_pf, labor_pf, μ, F, K, L, wealth_d) #initialize results struct
    prim, res #return structs
end

#this function allows us to change the relevant model parameters in a compact, systematic manner
function init_param(;θ::Float64, w::Float64, r::Float64, b::Float64, Z::Vector{Float64}, γ::Float64)
    param = Param(θ, w, r, b, Z, γ)
    param
end

#find the mass of agents in each age based on population growth
function mu_finder(prim::Primitives,mu_1::Float64)
    @unpack n, N = prim
    μ = zeros(N) #empty array
    μ[1] = mu_1
    for i=2:N
        μ[i] = μ[i-1]/(1+n)
    end
    μ = μ/sum(μ)
    μ
end

#based on agent's productivity and chosen level of a_prime, find the optimal labor supply
function opt_l(ap::Float64,a::Float64, prod::Float64, param::Param)
    @unpack γ, θ, w, r = param
    val = max(0,min(1,(γ*(1-θ)*prod*w - (1-γ)*((1+r)*a - ap))/((1-θ)*w*prod))) #the agent's labor supply must be in [0,1]. If it is inside, it will be given by the following function
    val 
end

#based on a consumer's chosen level of a_prime, current assets, and productivity, find their income
function work_budget(ap::Float64, a::Float64, prod::Float64, param::Param )
    @unpack w, θ, r = param
    #find income of agent
    budg = w*(1-θ)*prod*opt_l(ap, a, prod, param) + (1+r)*a
    budg
end

function utility_w(ap::Float64, a::Float64, prod::Float64, prim::Primitives, param::Param )
    @unpack σ = prim
    @unpack γ = param
    u = 0.0
    if work_budget(ap, a, prod, param) > ap && (opt_l(ap, a, prod, param) < 1 || γ == 1.0)
            u = (((work_budget(ap, a, prod, param)-ap)^(γ) *(1-opt_l(ap, a, prod, param))^(1-γ))^(1-σ))/(1-σ)
    else
        u = -Inf
    end
    u
end


#solve consumer's problem given w, r, and b fixed.
function cons_opt(prim::Primitives, res::Results, param::Param)
    @unpack na, A, N, jr, σ, δ, β, Π, η_arr, a_range, a_min, a_max = prim
    @unpack θ, w, r, b, Z, γ = param

    #solve problem for age 66 agents first
    for a_i=1:na #eat everything! 
        a = A[a_i]
        c = (1+r)*a + b
        res.val_func[N, a_i, :] .= (c)^((1-σ)*γ)/(1-σ)
        res.cap_pf[N,a_i, :] .= 0.0
    end
    for age in reverse(jr:N-1)#loop backwards over retirement ages
        #interpolate value functions
        vf_interp_h = scale(interpolate(res.val_func[age+1, :, 1],  BSpline(Linear())), a_range)
        vf_interp_l = scale(interpolate(res.val_func[age+1, :, 2],  BSpline(Linear())), a_range)
        vf_interp = [vf_interp_h, vf_interp_l] #combine into one nice object
        #print(age)
        for a_i=1:na
            #the budget for a retired agent is their previous asset holdings, interest, and retirement benefits
            budget = (1+r)*A[a_i] + b
            #find optimal level of a_prime using a numerical optimization algorithm
            val_opt = optimize(ap ->  -(budget - ap)^((1-σ)*γ)/(1-σ) - β*vf_interp[1](ap) , a_min, min(budget, a_max))
            val = -val_opt.minimum #finds the value at optimal choice
            #find the corresponding optimal choice of a_prime
            opt_ap = val_opt.minimizer
            #of course, we searched over an interval - this means the optimal a_prime may not be on our grid. Here, we find the closest level of capital which lies on the grid
            closest_ap_i = findmin(abs.(A .- opt_ap ))[2]
            #find the corresponding level of a in the grid
            closest_ap = A[closest_ap_i]
            #update the value function evaluated at the closest point in the grid to the optimal a_prime
            res.val_func[age, a_i, :] .= (budget - closest_ap)^((1-σ)*γ)/(1-σ) + β*vf_interp[1](closest_ap)
            #update policy function for capital
            res.cap_pf[age, a_i, :] .= closest_ap
        end
    end

    for age=reverse(1:jr-1)#loop backwards over working ages
        #interpolate value functions of high and low productivity agents
        vf_interp_h = scale(interpolate(res.val_func[age+1, :, 1], BSpline(Linear())), a_range) 
        vf_interp_l = scale(interpolate(res.val_func[age+1, :, 2], BSpline(Linear())), a_range)
        vf_interp = [vf_interp_h, vf_interp_l] #combine into one object for ease of use
        η = η_arr[age] #find efficiency level of agent
        for a_i=1:na #iterate over asset holdings
            a = A[a_i]
            for z_i = 1:2 #iterate over productivity levels
                prod = η*Z[z_i] #find productivity given by efficiency times factor z
                budget_a_i = findfirst([(work_budget(ap, a, prod, param) < ap) for ap in A]) #find the index of the highest choice of a_prime such that a_prime will exceed the income of the agent. The income of the agent is computed using the work_budget function (note: the income is a function of their labor supply, which is in turn a function of a_prime, which is why this is a bit more complicated!). If the agent chooses an a_prime which exceeds this level, their utility goes to negative infinity. So, we know the optimal choice must lie below this. 
                #practically, this is needed because the optimization algorithm needs a finite set to maximize over and will kinda freak out if you accidentally allow negative consumption. The only tricky part is that the agent's budget constraint depends on a_prime (since their optimal labor supply depends on a_prime).
                if isnothing(budget_a_i) #if the agent's choice of a_prime never exceeds their budget, set the upper bound of search space to the max capital level
                    budget_a = a_max
                else #otherwise, set the search domain to be [a_min, budget_a]
                    budget_a = A[budget_a_i]
                end

                #find the optimal choice of a_prime. The optimization algorithm can only find minima, so we minimize the negative of the value function. Note we use the interpolated value functions at each productivity level 
                val_opt = optimize(ap ->  -utility_w(ap, a, prod, prim, param) - β*(Π[z_i, 1]*vf_interp[1](ap) + Π[z_i, 2]*vf_interp[2](ap))  , a_min, budget_a )
                val = -val_opt.minimum #the max value
                #find the corresponding optimal choice of a_prime
                opt_ap = val_opt.minimizer
                #of course, we searched over an interval - this means the optimal a_prime may not be on our grid. Here, we find the closest level of capital which lies on the grid
                closest_ap_i = findmin(abs.(A .- opt_ap ))[2]
                #find the corresponding level of a in the grid
                closest_ap = A[closest_ap_i]
                #update the value function evaluated at the closest point in the grid to the optimal a_prime
                res.val_func[age, a_i, z_i] = utility_w(closest_ap, a, prod, prim, param) + β*(Π[z_i, 1]*vf_interp[1](closest_ap) + Π[z_i, 2]*vf_interp[2](closest_ap))
                #update value function
                res.cap_pf[age, a_i, z_i] = closest_ap
                #update labor policy function 
                res.labor_pf[age, a_i, z_i] = opt_l(closest_ap, a, prod, param)
                
            end
        end
    end
    res.val_func
end

#this function finds the steady-state distribution over age, assets, and productivity levels
function F_finder(prim::Primitives, res::Results, param::Param)
    @unpack N, n,na, A, Z_freq, Π = prim
    μ = res.μ
    F = zeros(N, na, 2)
    F[1, 1, 1] = Z_freq[1]*μ[1] #mass of agents of age one with zero assets and shock z_h 
    F[1, 1, 2] = Z_freq[2]*μ[1] #above, but with shock z_l
    for age=2:N
        for a_i=1:na
            a = A[a_i] #the selected capital level corresponding to a_i
            a_i_sel = findall(az -> az == a, res.cap_pf[age-1, :, :]) #find all (a,z) pairs in the previous age which select a
            for az in a_i_sel #iterate over all pairs in previous age which select a_i
                a_pre_i = az[1] #extract the previous a index
                z_pre_i = az[2] #extract previous z index
                #find mass of agents who chose a_i for each productivity type
                F[age, a_i, 1] += F[age-1, a_pre_i, z_pre_i]*Π[z_pre_i, 1]/(1+n) #all agents who chose a_i and got z_h
                F[age, a_i, 2] += F[age-1, a_pre_i, z_pre_i]*Π[z_pre_i, 2]/(1+n) #all agents who chose a_i and got z_l
            end
        end
    end
    res.F = F
end

#this function computes the aggregate supply of capital 
function cap_supply(prim::Primitives, res::Results, param::Param)
    @unpack N, A, na, η_arr = prim
    K_agg = 0.0
    for age = 1:N #iterate over all states, add the asset holdings weighted by mass of agents with that asset level
        for z_i = 1:2
            K_agg += sum(res.F[age, :, z_i] .* A)
        end
    end
    K_agg
end

#this function computes the aggregate labor supply 
function lab_supply(prim::Primitives, res::Results, param::Param)
    @unpack A, na, jr, η_arr= prim
    Z = param.Z
    L_agg = 0.0
    for age = 1:jr-1 #iterate over all working ages and productivity levels
        for z_i = 1:2
            prod = η_arr[age]*Z[z_i] #find productivity of worker of given age with given productivity level
            lab = res.labor_pf[age, :, z_i] #find optimal labor supply of agent
            L_agg += sum(res.F[age, :, z_i] .* (prod*lab)) #add mass of agents times their labor supply
        end
    end
    L_agg
end

function model_solver(prim::Primitives, res::Results, param::Param)
    cons_opt(prim, res, param) #find value functions and policy functions for given parameters
    res.F = F_finder(prim,  res, param) #find stationary distribution
    cap = cap_supply(prim,  res, param) #find aggregate asset supply
    lab = lab_supply(prim, res, param) #find aggregate labor supply
    cap, lab
end

function eq_finder(prim::Primitives, res::Results, param::Param)
    @unpack α, jr, N, δ = prim
    @unpack θ, w = param
    L_0 = res.L
    K_0 = res.K
    param.r = α*(L_0/K_0)^(1-α)-δ #market clearing interest rate based on K_0 and L_0
    param.w = (1-α)*(K_0/L_0)^α #market clearing wage based on K_0 and L_0
    param.b = (θ * w * L_0)/(sum(res.μ[jr:N])) #social security benefits implied by govt budget constraint
    #then, solve model with parameters and find K_new and L_new
    K_new, L_new = model_solver(prim, res,param)
    K_new, L_new
end

#this funcion iteratively finds equilibrium aggregate quantities K and L
function kl_search(prim::Primitives, res::Results, param::Param)
    K_new, L_new = eq_finder(prim, res, param) #using initial guess for K_0 and L_0 in the results struct, find the levels of capital and labor supplied in steady state
    tol = 0.005
    error = max(abs(K_new - res.K), abs(L_new - res.L)) #compute max difference between guesses and observed outputs
    n = 0
    println("Finding market clearing K and L") 
    while error > tol #iterate until guesses become sufficiently close together
        n +=1
        print("Iteration: $(n), error: $(error). ") #little progress message :)
        res.K = 0.6*res.K + 0.4*K_new #update guess of K as weighted average of previous guess and model output
        res.L = 0.6*res.L + 0.4*L_new #ditto for L
        K_new, L_new = eq_finder(prim, res, param) #find aggregate K and L given parameters implied by these new guesses
        error = max(abs(K_new - res.K), abs(L_new - res.L)) #find error
    end
    println("Convergence in $(n) iterations! K = $(res.K), L = $(res.L)")
    res.K, res.L, param.w, param.r, param.b
end

#this function computes the aggregate welfare given the value function and the steady-state asset distribution
function welfare(prim::Primitives, res::Results, param::Param)
    @unpack N, A, na = prim
    @unpack Z = param
    @unpack F, val_func = res
    #Z = param.Z
    #F = param.F
    welfare = 0.0
    for age=1:N #loop over states and add the mass of agents in each state times the value function of those agents
        for z_i =1:2
            welfare += sum(F[age, :, z_i] .* val_func[age, :, z_i])
        end
    end
    welfare
end

#this function computes the steady-state wealth distribution based on the steady-state asset distribution 
function wealth_dist(prim::Primitives, res::Results, param::Param)
    @unpack η_arr, na, N , A, jr= prim
    @unpack w, r, b, θ, Z = param 
    wd = zeros(N, na, 2) #initialize empty array to store wealth distribution
    for age = 1:N #iterate over all ages, productivity levels, and asset holdings
        for z_i = 1:2
            for a_i=1:na
                if age < jr #if the agent is working, their income will depend on their productivity, asset, and labor choice
                    prod = η_arr[age]*Z[z_i]
                    wealth = w*(1-θ)*prod*res.labor_pf[age, a_i, z_i] + (1+r)*A[a_i] #find wealth level of agent of given age in state z_i with asset holding a
                else #if the agent is retired, their income will depend only on b and their level of assets
                    wealth = b + (1+r)*A[a_i]
                end
                #find the point in the capital grid which corresponds to the agent's level of wealth
                w_i = findmin(abs.(A .- wealth ))[2]
                wd[age, w_i, z_i] += res.F[age, a_i, z_i] #add the mass of agents with given wealth level
            end
        end
    end
    res.wealth_d = wd
end

#this function finds the coefficient of variation for the wealth distribution. The coefficient of variation is defined as the ratio of the population standard deviation over the population mean 
function coeff_of_var(prim::Primitives, res::Results, param::Param)
    @unpack N, A, na = prim
    @unpack Z = param
    wealth_var = 0.0 #initial value for variance of the wealth distribution 
    mean_wealth = 0.0 #initial value for mean of the wealth distribution
    wealth_dist(prim, res, param) #find stationary wealth distribution
    @unpack wealth_d = res #load the wealth dist
    for age=1:N #iterate over states to find the mean level of wealth (weighted by wealth distribution)
        for z_i =1:2
            mean_wealth += sum(wealth_d[age, :, z_i] .* A)
        end
    end
    for age = 1:N #iterate over states to find squared deviation of wealth from the mean, weighted by wealth dist
        for z_i = 1:2
            for a_i = 1:na
                wealth_var += wealth_d[age, a_i, z_i] * (A[a_i] - mean_wealth)^2
            end
        end
    end
    wealth_sd = sqrt(wealth_var)
    print("Mean wealth: ", mean_wealth,", Wealth SD: ", wealth_sd) #the standard deviation is the square root of the variance
    wealth_sd/mean_wealth #return coefficient of variation defined as the standard deviation over the mean
end

function equivalent_variation_bench()
    #first, solve model with social security and find value function
    prim, res = Initialize()
    param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 0.42) #initialize benchmark model
    k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for equilibrium quantities/parameters
    val_ss = res.val_func

    #now, solve model without social security and find value function
    prim, res = Initialize()
    param = init_param(θ=0.0, w=1.05, r = 0.05, b = 0.0, Z = [3.0, 0.5], γ = 0.42) #model without social security has θ = 0 (and thus b = 0)
    k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for eq quantities/parameters
    val_no_ss = res.val_func
    @unpack γ = param
    @unpack σ = prim
    λ = (val_no_ss ./ val_ss).^(1/(γ*(1-σ))) .- 1
    λ
end

function equivalent_variation_inelastic()
    #first, solve model with inelastic labor and social security and find value function
    prim, res = Initialize()
    param = init_param(θ=0.11, w=1.05, r = 0.05, b = 0.2, Z = [3.0, 0.5], γ = 1.0) #initialize benchmark model
    k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for equilibrium quantities/parameters
    val_ss = res.val_func

    #now, solve model without social security and find value function
    prim, res = Initialize()
    param = init_param(θ=0.0, w=1.05, r = 0.05, b = 0.0, Z = [3.0, 0.5], γ = 1.0) #model without social security has θ = 0 (and thus b = 0)
    k_e, l_e, w_e, r_e, b_e = kl_search(prim, res, param) #search for eq quantities/parameters
    val_no_ss = res.val_func
    @unpack γ = param
    @unpack σ = prim
    λ = (val_no_ss ./ val_ss).^(1/(γ*(1-σ))) .- 1
    λ
end

function prop_favor_age(prim,res, λ)
    @unpack N = prim
    @unpack F, μ = res
    prop_age = zeros(N)
    for age = 1:N
        prop_age[age] = sum((λ[age, :, :] .>= 0) .* res.F[age, :, :])/μ[age]
    end
    prop_age
end