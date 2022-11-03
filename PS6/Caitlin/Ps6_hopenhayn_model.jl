#Comp 899 PS6 Caitlin Dutta
#keyword-enabled structure to hold model primitives
using Parameters, Plots, SharedArrays, LinearAlgebra
@with_kw struct Primitives
    β::Float64 = 0.8 #discount rate
    θ::Float64 = 0.64 #labor share
    A::Float64 = 1/200 #productivity of labor
    ce::Float64 = 5 #entry cost
    markov::Matrix{Float64} = [0.6598 0.2600 0.0416 0.0331 0.0055 ; 0.1997 0.7201 0.0420 0.0326 0.0056 ;0.2000 0.2000 0.5555 0.0344 0.0101 ;0.2000 0.2000 0.2502 0.3397 0.0101 ; 0.2000 0.2000 0.2500 0.3400 0.0100]
    v::Vector{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063] #entrant dist

    s_grid::Vector{Float64} = [3.98e-4, 3.58, 6.82, 12.18, 18.79]
    ns::Int64 = length(s_grid)
end

using .MathConstants: γ

#structure that holds model results
 mutable struct Results
    val_func::Array{Float64,1} #value function
    pol_func::Array{Float64,1} #policy function
    labor_func::Array{Float64, 1} #budg choices
    F::Array{Float64, 1} #F dist
    p::Float64
    M_Incumbents::Float64
    M_Entrants::Float64
    M_Exits::Float64
    Agg_Labor::Float64
    L_Incumbents::Float64
    L_Entrants::Float64
    LF_Entrants::Float64
    cf::Float64


end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives()
    val_func = Array{Float64}(zeros(prim.ns)) #initial value function guess
    pol_func = Array{Float64}(zeros(prim.ns)) #initial policy function guess
    labor_func = Array{Float64}(zeros(prim.ns)) #labor policy
    F = Array{Float64}(zeros(prim.ns)) #F dist
    p::Float64 = 0.0
    M_Incumbents::Float64 = 0.0
    M_Entrants::Float64 = 0.0
    M_Exits::Float64 = 0.0
    Agg_Labor::Float64 = 0.0
    L_Incumbents::Float64 = 0.0
    L_Entrants::Float64 = 0.0
    LF_Entrants::Float64 = 0.0
    cf::Float64 = 0.0
    res = Results(val_func, pol_func, labor_func, F, p, M_Incumbents, M_Entrants, M_Exits, Agg_Labor, L_Incumbents, L_Entrants, LF_Entrants, cf)
    prim, res
end

function Util(prim::Primitives, c::Float64, n::Float64)
    @unpack A = prim #unpack model primitives
    util = ln(c) - A*n
    util
end

function Profit(prim::Primitives, p::Float64, s::Float64, ldemand::Float64)
    @unpack θ = prim #unpack model primitives
    @unpack cf = res
    profit = p*s*ldemand^θ - ldemand - p*cf
    profit
end

function labor_demand(prim::Primitives, p::Float64, s::Float64)
    #function of price and prims.
    @unpack θ = prim
    ldemand =max(0, (θ*p*s)^(1/(1-θ)))
    ldemand
end

#Bellman Operator

function W_func(prim::Primitives, res::Results, p::Float64)
    @unpack  s_grid, β, ns, markov = prim #unpack model primitives
    @unpack cf = res
    #only choice is over nx
    val_new = zeros(ns)
    for si = 1:ns # current state
        s = s_grid[si]
        ldemand = labor_demand(prim, p, s)
        prof = Profit(prim, p, s, ldemand)
        res.labor_func[si] = ldemand
        val = 0
        for spi = 1:ns
            val += β*res.val_func[spi]*markov[si,spi] #compute value
        end
        if val>=0 #if continuation value is pos, firm should stay in
            res.pol_func[si] = 0 #update policy function
            val_new[si] = prof + val
        elseif val < 0
            res.pol_func[si] = 1
            val_new[si] = prof #no continuation value if they exit
        end #if val
    end #si
    val_new
end #func

function W_func_ev(prim::Primitives, res::Results, p::Float64, α::Float64)
    @unpack  s_grid, β, ns, markov, θ = prim #unpack model primitives
    @unpack cf = res
    #only choice is over nx
    U_new = zeros(ns)
    for si = 1:ns # current state
        s = s_grid[si]
        ldemand = labor_demand(prim, p, s)
        prof = p*s*ldemand^θ - ldemand - p*cf
        res.labor_func[si] = ldemand
        val = 0
        for spi = 1:ns
            val += β*markov[si,spi]*res.val_func[spi] #compute continuation value
        end
        v_x0 = prof + val
        v_x1 = prof #no continuation value if they exit
        res.pol_func[si] =  exp(α*v_x1)/(exp(α*v_x0) + exp(α*v_x1))#given formula for policy fn
        U_new[si] = γ/α + log(exp(α*v_x0 ) + exp(α*v_x1))/α
    end #si
    U_new
end #func



function convergence(prim::Primitives, res::Results, p::Float64, α::Float64)
    @unpack ns = prim

    if α == 0
        tol = 1e-6
        error = 100
        iter = 0
        while error > tol
            iter += 1
            val_new = W_func(prim, res, p)
            error = maximum(abs.(res.val_func - val_new))
        #    println("iteration ", iter, " error is ", error)
            res.val_func = val_new
        end
    elseif α != 0
        tol = 1e-6
        error = 100
        iter = 0
        while error > tol
            iter += 1
            U_new = W_func_ev(prim, res, p, α)
            error = maximum(abs.(res.val_func - U_new))
        #    println("iteration ", iter, " error is ", error)
            res.val_func = U_new
        end
    end
end

function EC_p(prim::Primitives, res::Results, p::Float64)
    @unpack v, ns, ce = prim
    EC = 0
    int = 0
    for si = 1:ns
        int += v[si] * res.val_func[si]/p
    end
    EC = int-ce
    EC
end

function solve_w(prim::Primitives, res::Results, α::Float64)
    #we need ec = 0
    p = 0.738 # starting p guess
    convergence(prim, res, p, α)
    EC = EC_p(prim, res, p)
    tol = 0.1
    iter = 0

    while abs(EC) > tol && iter < 1000
        iter += 1
        if EC > 0
            p_new = 0.9*p
        elseif EC <= 0
            p_new = 1.1*p
        end
        p = p_new
        convergence(prim, res, p, α)
        EC = EC_p(prim, res, p)
    #    println("iteration", iter, " ec = ", EC, " p = ", p, " lab 2 is ", res.labor_func[2])
    end
    #println("p is ", p)
    p
end

## T*

#find invariant distribution of firm sizes

function create_mu(prim::Primitives,res::Results, M::Float64)
    @unpack markov, v = prim
    @unpack pol_func = res
    first = markov' .* (1 .- pol_func)
    mu = M * inv(I - first)*v
    mu
end

function agg_labor_d(prim::Primitives, res::Results, mu::Vector{Float64}, M::Float64)
    @unpack v = prim
    #lagg = entrants demand + incumbents demand (same func just diff dist)
    ld = 0
    ld_e = sum(res.labor_func .* v * M)
    ld_i = sum(res.labor_func .* mu)
    res.L_Incumbents = ld_i
    res.L_Entrants = ld_e
    res.LF_Entrants = ld_e/(ld_i + ld_e)
    ld = ld_e + ld_i
    ld
end

function agg_labor_s(prim::Primitives, res::Results, p::Float64, pagg::Float64)
    @unpack A = prim
    ls = p/A - pagg
    #ls = 1/A - pagg
    ls
end

function agg_prof(prim::Primitives, res::Results, mu::Vector{Float64}, p::Float64, M::Float64)
    @unpack θ, v, s_grid = prim
    @unpack labor_func, cf = res
    pagg = sum((p*s_grid .* labor_func.^θ - labor_func .- p*cf) .* (mu + M*v))
    pagg
end

#labor mkt clearing to find M
function mkt_clearing(α::Float64, cf::Float64)
    res.cf = cf
    M = 50.0 #starting guess
    M_high = 100.0
    M_low = 0.0
    tol = 1e-4
    iter = 0
    #prim, res = Initialize()
    p = solve_w(prim, res, α)
    res.p = p
    mu = create_mu(prim, res, M)
    pagg = agg_prof(prim, res, mu, p, M)
    ls = agg_labor_s(prim, res, p, pagg)
    ld = agg_labor_d(prim, res, mu, M)
    lmc = ld - ls

    while abs(lmc) > tol
        iter += 1
    #    println("this is iteration ", iter, " lmc is ", lmc, " M is ", M, " ld is ", ld)
        if lmc >= 0
            M_high = M
        else
            M_low = M
        end
        if iter > 1000
            break
        end
        M = (M_high + M_low)/2
        mu = create_mu(prim, res, M)
        pagg = agg_prof(prim, res, mu, p, M)
        ls = agg_labor_s(prim, res, p, pagg)
        ld = agg_labor_d(prim, res, mu, M)
        lmc = ld - ls
    end
    M
    res.M_Incumbents = sum(mu)
    res.M_Entrants = M
    res.M_Exits = sum(mu.*res.pol_func)
    res.Agg_Labor = ld

    println("(Alpha, CF) are ", α, " , ", cf, " Price is ", res.p, " Mass of Incumbents is ", res.M_Incumbents, " Mass of Entrants ",res.M_Entrants,
    " Mass of Exits ", res.M_Exits, " Aggregate Labor ", res.Agg_Labor, " Labor of Incumbents ", res.L_Incumbents,
    " Labor of Entrants ", res.L_Entrants, " Fraction of Labor in Entrants ", res.LF_Entrants)

end
