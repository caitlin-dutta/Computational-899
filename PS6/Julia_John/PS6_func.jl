@with_kw struct Primitives
    β::Float64 = 0.8 #discount factor
    θ::Float64 = 0.64 #production parameter
    ns::Int64 = 5 #number of grid pts
    s_grid ::Vector{Float64} = [3.98e-4, 3.58, 6.82, 12.18, 18.79]#productivity grid
    F::Array{Float64,2} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                           0.1997 0.7201 0.0420 0.0326 0.0056;
                           0.2000 0.2000 0.5555 0.0344 0.0101;
                           0.2000 0.2000 0.2502 0.3397 0.0101; 
                           0.2000 0.2000 0.2500 0.3400 0.0100] #productivity transition matrix
    ent_dist::Vector{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063] #entrant distribution
    A::Float64 = 0.005 #consumer labor disutility parameter
    ce::Float64 = 5.0 #entry cost
end

mutable struct Results
    W::Vector{Float64} #value function 
    exit_pol::Vector{Float64} #exit decision 
    labor_pol::Vector{Float64} #labor demand 
    p::Float64 #price level
    μ::Vector{Float64} #invariant distribution
    M::Float64 #mass of entrants
    M_exit::Float64 #mass of exits
    L_D::Float64 #labor demand
    L_I::Float64 #labor of incumbents
    L_E::Float64 #labor of entrants
    α::Float64 #TIEV shock 
    cf::Float64 #per-period fixed cost
end

function Init()
    prim = Primitives()
    W = zeros(prim.ns)
    exit_pol = zeros(prim.ns)
    labor_pol = zeros(prim.ns)
    p = 1.0
    μ = zeros(prim.ns)
    M = 1.0
    M_exit = 0.2
    L_D = 100.0
    L_I = 100.0
    L_E = 10.0
    α = 0.0
    cf = 10.0
    res = Results(W, exit_pol, labor_pol, p, μ, M, M_exit, L_D, L_I, L_E, α, cf)
    prim, res
end

#finds the optimal n based on the price level and productivity
function opt_n(prim::Primitives, p::Float64,s::Float64)
    @unpack θ = prim
    n_opt = max(0, (θ*p*s)^(1/(1-θ))) #find optimal n using firm's first order condition
    n_opt
end

#this function computes the bellman equation for the non-stochastic specification
function W_comp(prim::Primitives, res::Results, p::Float64)
    @unpack s_grid,ns,β, θ, F, A, ce = prim
    @unpack cf = res
    vf_new = zeros(length(s_grid))
    for s_i = 1:ns #iterate over firm size index
        s = s_grid[s_i] #find corresponding firm size
        n = opt_n(prim, p, s) #find optimal labor demand for firm
        res.labor_pol[s_i] = n #update labor policy function
        val = p*s*n^θ - n - p*cf #find firm's static profits
        cont_val = 0.0 #initial continuation value of 0
        for sp_i = 1:ns #iterate over future states
            cont_val += β*F[s_i, sp_i]*res.W[sp_i] #add the discounted value at state s_prime multiplied by probability of transitioning to s_prime
        end
        if cont_val < 0  #if the firm has negative expected value of staying in, they should exit
            res.exit_pol[s_i] = 1 #set the firm's exit policy function equal to 1 to indicate they exit
            vf_new[s_i] = val #update value function to be only static profits
        else #otherwise, they choose to stay
            res.exit_pol[s_i] = 0 #set exit pf equal to 1 to indicate they stay
            vf_new[s_i] = val + cont_val #set their value function equal to static profit + continuation value
        end
    end
    vf_new
end

#this function computes the bellman equation for the stochastic specification 
function W_comp_stoch(prim::Primitives, res::Results, p::Float64, α::Float64)
    @unpack s_grid,ns, β, θ, F, A, ce = prim
    @unpack cf = res
    U_new = zeros(length(s_grid))
    for s_i = 1:ns #iterate over firm size index
        s = s_grid[s_i] #find corresponding firm size
        n = opt_n(prim, p, s) #find optimal labor demand for firm
        res.labor_pol[s_i] = n #update labor policy function
        val = p*s*n^θ - n - p*cf #find firm's static profits
        cont_val = 0.0 #initial continuation value
        for sp_i = 1:ns #iterate over future states
            cont_val += β*F[s_i, sp_i]*res.W[sp_i] #add the discounted value at state s_prime multiplied by probability of transitioning to s_prime
        end
        v_x0 = val + cont_val #the value of staying in is the static profit plus continuation value
        v_x1 = val #value of exiting is simply the static profit
        res.exit_pol[s_i] = exp(α*v_x1)/(exp(α*v_x0) + exp(α*v_x1)) #use the given closed-form formula for exit probability given TIEV shocks
        U_new[s_i] = γ/α + log(exp(α*v_x0 ) + exp(α*v_x1))/α #find new value function given closed-form formula
    end
    U_new
end

#this function iterates value function until convergence; does either stochastic or non-stochastic depending on passed alpha parameter
function iterate(prim::Primitives, res::Results, p::Float64, α::Float64)
    res.W = zeros(prim.ns) #empty initial value function guess
    if α != 1.0 && α != 2.0 #if we are in the non-stochastic case
        error = 100 
        tol = 1e-9
        n = 0
        while error > tol && n < 1000 #while we have higher error than tolerance or under maxiter
            n += 1
            w_next = W_comp(prim, res, p) #find new value function
            error = maximum(abs.(w_next - res.W)) #find error between old and new value functions
            res.W = w_next #update vf
        end
    else
        error = 100
        tol = 1e-9 
        n = 0
        while error > tol && n < 1000 #while we have higher error than tolerance or under maxiter
            n += 1
            w_next = W_comp_stoch(prim, res, p, α) #find new value function
            error = maximum(abs.(w_next - res.W)) #find error between old and new value functions
            res.W = w_next #update vf
        end
    end

    #println("Convergence!")
end

#determine the expected value of entering minus the cost of entering
function free_entry(prim::Primitives, res::Results, p::Float64)
    @unpack ns, ent_dist, ce = prim
    iterate(prim, res, p, res.α) #iterate bellman equation until convergence
    ec = 0.0
    for s_i = 1:ns #iterate over states
        ec += ent_dist[s_i]*res.W[s_i]/p #find the value of entering and getting state s, weighted by the probability of getting state s upon entrance
    end
    ec -= ce #subtract cost of entering
    ec
end

#find the price which makes firms indifferent between entering and exiting
function price_finder(prim::Primitives, res::Results)
    p_low = 0.0
    p_high = 1.0
    p_guess = 0.5 #initial guess of price
    tol = 1e-6
    EC = free_entry(prim, res, p_guess) #find expected value of entering
    n = 0
    while abs(EC) > tol && n < 1000 #while value of entering is not zero (or max iter is reachet)
        n += 1
        #println("Guessed p: $(p_guess), error: $(EC)")
        if EC >= 0 #if value of entering is positive, lower price guess
            p_high = p_guess 
        else #otherwise, raise price guess
            p_low = p_guess
        end
        p_guess = (p_low + p_high)/2 #find new guess price as midpoint of the high and low values
        EC = free_entry(prim, res, p_guess) #compute value of entry again
    end
    res.p = p_guess #store price which satisfies entry condition
    p_guess
end


#find invariant firm size distribution
function invar_mu(prim::Primitives, exit_pf::Vector{Float64}, M::Float64)
    @unpack F, ent_dist = prim
    F_hat = F' .* (1 .- exit_pf) #multiply the transition matrix element-wise by the probability that a firm exits given state s.
    μ = M*inv(I - F_hat)*ent_dist #solve for invariant distribution by setting mu_t = mu_t+1 and rearranging
    μ
end

#find aggregate labor demand
function labor_demand(prim::Primitives, res::Results, M::Float64)
    @unpack ns, ent_dist, ce = prim
    μ = invar_mu(prim,  res.exit_pol, M) #find invariant dist of firms
    ld = 0.0
    res.L_E = sum(res.labor_pol .* ent_dist * M) #find labor demand of entrants
    res.L_I = sum(res.labor_pol .* μ) #find labor demand of incumbents
    ld = res.L_I + res.L_E #total labor supply
    ld
end

#find aggregate firm profit
function profits(prim::Primitives, res::Results, μ::Vector{Float64}, p::Float64, M::Float64)
    @unpack ns, s_grid, θ, ent_dist =prim
    @unpack cf = res
    Π = sum((p*s_grid .*res.labor_pol.^θ - res.labor_pol .- p*cf) .*(μ + M*ent_dist)) #find profits of incumbent + entering firms
    Π
end
    
#find aggregate labor supply
function cons_opt(prim::Primitives, Π::Float64, p::Float64)
    @unpack A = prim
    n_supp = 1/A - Π #this comes from consumers' first order condition for labor supply
    n_supp
end

#find market clearing level of entrants by equating labor demand and labor supply
function market_clearing(prim::Primitives, res::Results, p::Float64)
    M_low = 0.0
    M_high = 100.0
    M_guess = 50.0
    tol = 1e-4
    μ = invar_mu(prim, res.exit_pol, M_guess) #find invar dist
    Π = profits(prim, res, μ, p, M_guess) #find aggregate profits
    L_supp = cons_opt(prim, Π, p) #determine aggregate labor supply
    L_d = labor_demand(prim, res, M_guess) #determine aggregate labor demand
    error = L_d - L_supp #find difference between labor supply and demand
    n = 0
    while abs(error) > tol && n < 1000
        n += 1
        #println("Iteration $(n), error: $(error)")
        if error >= 0 #if too much labor is demanded relative to supply, lower the guessed mass of entrants
            M_high = M_guess
        else #if more labor is supplied than is demanded, raise the guessed mass of entrants
            M_low = M_guess
        end
        M_guess = (M_high + M_low)/2 #update guessed entrant mass
        μ = invar_mu(prim, res.exit_pol, M_guess) #compute new invar dist
        Π = profits(prim, res, μ, p, M_guess) #compute aggregate profits
        L_supp = cons_opt(prim, Π, p) #compute labor supply
        L_d = labor_demand(prim, res, M_guess) #compute labor demand
        error = L_d - L_supp #find excess demand of labor
    end
    println("Convergence!")
    res.L_D = L_d
    res.M = M_guess
    M_guess
end

#solve the model and print output!
function model_solve(prim::Primitives, res::Results)
    res.p = price_finder(prim, res)
    market_clearing(prim, res, res.p)
    res.μ = invar_mu(prim, res.exit_pol, res.M)
    println("**********************************")
    println("α = $(res.α), c_f = $(res.cf):")
    println("p = $(res.p)") 
    println("Mass of incumbents: $(sum(res.μ))")
    println("M = $(res.M)")
    println("Mass of exits: $(sum(res.μ .* res.exit_pol))")
    println("μ = $(res.μ)")
    println("Aggregate labor = $(res.L_D)")
    println("Labor of incumbents = $(res.L_I)")
    println("Labor of entrants = $(res.L_E)")
    println("Fraction of labor of entrants = $(res.L_E/(res.L_E + res.L_I))")
end
