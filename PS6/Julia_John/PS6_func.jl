@with_kw struct Primitives
    β::Float64 = 0.8
    θ::Float64 = 0.64
    ns::Int64 = 5 #number of grid pts
    s_grid ::Vector{Float64} = [3.98e-4, 3.58, 6.82, 12.18, 18.79]
    n_grid::Vector{Float64} = [1.3e-9, 10.0, 60.0, 300.0, 1000.0]
    F::Array{Float64,2} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                           0.1997 0.7201 0.0420 0.0326 0.0056;
                           0.2000 0.2000 0.5555 0.0344 0.0101;
                           0.2000 0.2000 0.2502 0.3397 0.0101; 
                           0.2000 0.2000 0.2500 0.3400 0.0100]
    ent_dist::Vector{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    A::Float64 = 0.005
    ce::Float64 = 5.0
end

mutable struct Results
    W::Vector{Float64}
    exit_pol::Vector{Float64}
    labor_pol::Vector{Float64}
    p::Float64
    μ::Vector{Float64}
    M::Float64
    M_exit::Float64
    L_D::Float64
    L_I::Float64
    L_E::Float64
    α::Float64
    cf::Float64
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
    res = Results(W, exit_pol, labor_pol, p, μ, M, M_exit, L_D, L_I, L_E, α, 10.0)
    prim, res
end

function opt_n(prim::Primitives, p::Float64,s::Float64)
    @unpack θ = prim
    n_opt = max(0, (θ*p*s)^(1/(1-θ)))
    n_opt
end

function W_comp(prim::Primitives, res::Results, p::Float64)
    @unpack s_grid,ns, n_grid,β, θ, F, A, ce = prim
    @unpack cf = res
    vf_new = zeros(length(s_grid))
    for s_i = 1:ns #iterate over firm size index
        s = s_grid[s_i] #find corresponding firm size
        n = opt_n(prim, p, s) #find optimal labor demand for firm
        res.labor_pol[s_i] = n
        val = p*s*n^θ - n - p*cf
        cont_val = 0.0
        for sp_i = 1:ns #iterate over future states
            cont_val += β*F[s_i, sp_i]*res.W[sp_i]
        end
        if cont_val < 0
            res.exit_pol[s_i] = 1
            vf_new[s_i] = val
        else
            res.exit_pol[s_i] = 0
            vf_new[s_i] = val + cont_val
        end
    end
    vf_new
end

function W_comp_stoch(prim::Primitives, res::Results, p::Float64, α::Float64)
    @unpack s_grid,ns, n_grid,β, θ, F, A, ce = prim
    @unpack cf = res
    U_new = zeros(length(s_grid))
    for s_i = 1:ns #iterate over firm size index
        s = s_grid[s_i] #find corresponding firm size
        n = opt_n(prim, p, s) #find optimal labor demand for firm
        res.labor_pol[s_i] = n
        val = p*s*n^θ - n - p*cf
        cont_val = 0.0
        for sp_i = 1:ns #iterate over future states
            cont_val += β*F[s_i, sp_i]*res.W[sp_i]
        end
        v_x0 = val + cont_val
        v_x1 = val
        res.exit_pol[s_i] = exp(α*v_x1)/(exp(α*v_x0) + exp(α*v_x1))
        U_new[s_i] = γ/α + log(exp(α*v_x0 ) + exp(α*v_x1))/α
    end
    U_new
end

function Bellman(prim::Primitives, res::Results, p::Float64, α::Float64)
    res.W = zeros(prim.ns)
    if α != 1.0 && α != 2.0
        error = 100
        tol = 1e-9
        n = 0
        while error > tol && n < 1000
            n += 1
            w_next = W_comp(prim, res, p)
            error = maximum(abs.(w_next - res.W))
            res.W = w_next
        end
    else
        error = 100
        tol = 1e-9 
        n = 0
        while error > tol && n < 1000
            n += 1
            w_next = W_comp_stoch(prim, res, p, α)
            error = maximum(abs.(w_next - res.W))
            res.W = w_next
        end
    end

    #println("Convergence!")
end

function free_entry(prim::Primitives, res::Results, p::Float64)
    @unpack ns, ent_dist, ce = prim
    Bellman(prim, res, p, res.α)
    ec = 0.0
    for s_i = 1:ns
        ec += ent_dist[s_i]*res.W[s_i]/p
    end
    ec -= ce
    ec
end

function price_finder(prim::Primitives, res::Results)
    p_low = 0.0
    p_high = 10.0
    p_guess = 5.0
    tol = 1e-6
    EC = free_entry(prim, res, p_guess)
    n = 0
    while abs(EC) > tol && n < 1000
        n += 1
        #println("Guessed p: $(p_guess), error: $(EC)")
        if EC >= 0
            p_high = p_guess
        else
            p_low = p_guess
        end
        p_guess = (p_low + p_high)/2
        EC = free_entry(prim, res, p_guess)
    end
    res.p = p_guess
    p_guess
end

function invar_mu(prim::Primitives, exit_pf::Vector{Float64}, M::Float64)
    @unpack F, ent_dist = prim
    F_hat = F' .* (1 .- exit_pf) #multiply the transition matrix by an indicator for whether the firm stays in the market or not. Only keep transition probabilities for firms which remain
    μ = M*inv(I - F_hat)*ent_dist
    μ
end

function labor_demand(prim::Primitives, res::Results, M::Float64)
    @unpack ns, ent_dist, ce = prim
    μ = invar_mu(prim,  res.exit_pol, M)
    ld = 0.0
    res.L_E = sum(res.labor_pol .* ent_dist * M)
    res.L_I = sum(res.labor_pol .* μ)
    ld = res.L_I + res.L_E
    ld
end

function profits(prim::Primitives, res::Results, μ::Vector{Float64}, p::Float64, M::Float64)
    @unpack ns, s_grid, θ, ent_dist =prim
    @unpack cf = res
    Π = 0.0
    for s_i=1:ns
        Π += (p*s_grid[s_i]*res.labor_pol[s_i]^θ - res.labor_pol[s_i] - p*cf)*(μ[s_i] + M*ent_dist[s_i])
    end
    Π
end
    

function cons_opt(prim::Primitives, Π::Float64, p::Float64)
    @unpack A = prim
    n_supp = 1/A - Π
    n_supp
end

function market_clearing(prim::Primitives, res::Results, p::Float64)
    M_low = 0.0
    M_high = 100.0
    M_guess = 50.0
    tol = 1e-4
    μ = invar_mu(prim, res.exit_pol, M_guess)
    Π = profits(prim, res, μ, p, M_guess)
    L_supp = cons_opt(prim, Π, p)
    L_d = labor_demand(prim, res, M_guess)
    error = L_d - L_supp
    n = 0
    while abs(error) > tol && n < 1000
        n += 1
        #println("Iteration $(n), error: $(error)")
        if error >= 0
            M_high = M_guess
        else
            M_low = M_guess
        end
        M_guess = (M_high + M_low)/2
        μ = invar_mu(prim, res.exit_pol, M_guess)
        Π = profits(prim, res, μ, p, M_guess)
        L_supp = cons_opt(prim, Π, p)
        L_d = labor_demand(prim, res, M_guess)
        error = L_d - L_supp
    end
    println("Convergence!")
    res.L_D = L_d
    res.M = M_guess
    M_guess
end

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
