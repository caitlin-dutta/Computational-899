@with_kw struct Primitives 
    sim_data_df :: DataFrame = DataFrame(CSV.File("PS4_simdata.csv")) 
    sim_data :: Array{Int64} = identity.(Array(sim_data_df[!, Not([:Column1])]))
    state_space_df :: DataFrame = DataFrame(CSV.File("PS4_state_space.csv")) 
    state_space :: Array{Int64} = identity.(Array(state_space_df[!, Not([:Column1, :id])]))
    trans_a0_df :: DataFrame = DataFrame(CSV.File("PS4_transition_a0.csv")) 
    trans_a0 :: Array{Float64} = identity.(Array(trans_a0_df[!, Not([:Column1, :id])]))
    trans_a1_df :: DataFrame = DataFrame(CSV.File("PS4_transition_a1.csv")) 
    trans_a1 :: Array{Float64} = identity.(Array(trans_a1_df[!, Not([:Column1, :id])]))
    #vector of consumption shocks
    c_grid :: Vector{Int64} = [0, 1]
    #probability of each consumption shock
    c_prob :: Vector{Float64} = [0.5, 0.5]
    #normal price
    p_r :: Float64 = 4
    #sale price 
    p_s :: Float64 = 1
    #price vector
    p_grid :: Vector{Float64} = [p_r, p_s] 
    #transition matrix for prices
    Π :: Array{Float64,2} = [0.75 0.25; 0.9 0.1]
    #stockout penalty
    λ :: Float64 = -4
    α :: Float64 = 2
    #capacity constraint
    ibar :: Float64 = 8
    #discount factor
    β :: Float64 = 0.99 
    #total number of grid points
    n_grid :: Int64 = length(p_grid)*length(c_grid)*(ibar + 1)
    #matrix to keep track of which variables correspond to which state
    S :: Array{Int64, 2} = S_maker(length(p_grid), length(c_grid), ibar + 1)
end

mutable struct Results
    #expected value function
    v_bar :: Vector{Float64}
    #implied expected value function using CCP
    v_bar_p :: Vector{Float64}
    #conditional choice probability 
    p_hat :: Vector{Float64}
end

function Initialize(n_grid)
    v_bar = zeros(n_grid)
    v_bar_p = zeros(n_grid)
    p_hat = zeros(n_grid)
    res = Results(v_bar, v_bar_p, p_hat)
    res
end

function S_maker(np, nc, ni)
    S = zeros(Int64, 36, 3)
    n=0
    for p =1:np, c=1:nc, i=1:ni
        n +=1
        S[n,:] =Int64.([i, c, p])
    end
    S
end

function u_icp(prim::Primitives, a::Int64, i::Int64, c::Int64, p::Int64, λ)
    @unpack α = prim
    u = 0
    if a == 1
        u = α*c - p
    else
        if i > 0
            u = α*c
        else
            u = λ *(c > 0)
        end
    end
    u
end


function bellman(prim::Primitives, res::Results)
    @unpack c_grid, c_prob, p_grid, λ, α, trans_a0, trans_a1, state_space, ibar, n_grid, β = prim
    V = zeros(n_grid)
    for k=1:n_grid
        i, c, p = state_space[k,:]
        #i = i_ind -1
        #c = c_grid[c_i]
        #p = p_grid[p_i]
        u0 = u_icp(prim, 0, i, c, p, λ)
        u1 = u_icp(prim, 1, i, c, p, λ)
        v_prime_0 = β*dot(res.v_bar,trans_a0[k, :])
        v_prime_1 =β*dot(res.v_bar,trans_a1[k, :])
        v0 = u0 + v_prime_0 
        v1 = u1 + v_prime_1
        V[k] = log(exp(v0) + exp(v1)) + γ
    end
    return V
end

function bellman_iteration(prim::Primitives, res::Results)
    @unpack p_grid, c_grid, ibar, n_grid = prim
    tol = 1e-10
    error = 100
    res.v_bar = zeros(n_grid)
    V1 = bellman(prim, res)
    error = maximum(abs.(V1 - res.v_bar))
    n = 0
    while error > tol
        n += 1
        res.v_bar = V1
        V1 = bellman(prim, res)
        error = maximum(abs.(V1 - res.v_bar))
    end
    println("Convergence!! $n")
end


function ccp_est(prim::Primitives)
    @unpack sim_data, S, n_grid =  prim
    ccp = zeros(n_grid)
    for k=1:n_grid 
        sim_sk = sim_data[(sim_data[:,2] .+ 1 .== k), :]
        prob_k = sum(sim_sk[:,1])/size(sim_sk, 1)
        ccp[k] = min(0.999, max(0.001, prob_k))
    end
    res.p_hat = ccp
    ccp
end

function ccp_computation(prim, res, λ)
    @unpack n_grid, c_grid, p_grid, state_space, trans_a0, trans_a1, β = prim
    e1 = γ .- log.(res.p_hat)
    e0 = γ .- log.(1 .- res.p_hat)
    v0 = zeros(n_grid)
    v1 = zeros(n_grid)
    for k=1:n_grid
        i, c, p = state_space[k, :]
        v0[k] = u_icp(prim, 0, i, c, p, λ)
        v1[k] = u_icp(prim, 1, i, c, p, λ)
    end
    F_P = (1 .- res.p_hat) .* trans_a0 .+ res.p_hat .* trans_a1
    res.v_bar_p = inv(diagm(ones(n_grid)) - β*F_P)*((1 .- res.p_hat) .*(v0 .+ e0) + res.p_hat .*(v1 .+ e1))
    v_tilde = (v1 + β*trans_a1*res.v_bar_p) - (v0 + β*trans_a0*res.v_bar_p)
    p_hat = 1 ./(1 .+ exp.(-v_tilde))
    p_hat
end

function ccp_iteration(prim, res, λ)
    ccp_est(prim)
    p_hat = ccp_computation(prim,res, λ)
    tol = 1e-12
    error = maximum(abs.(p_hat - res.p_hat))
    n = 0
    while error > tol
        n += 1
        res.p_hat = p_hat
        p_hat = ccp_computation(prim,res, λ)
        error = maximum(abs.(p_hat - res.p_hat))
    end
    t = p_hat
    print("Convergence!! $n")
    t
end

function log_likelihood(prim::Primitives, res::Results, λ::Float64)
    @unpack sim_data, n_grid = prim
    ccp_iteration(prim, res, λ)
    if sum(res.v_bar_p .== NaN) > 0
        val = -1e10
        print("Not allowed!")
    else
        val = sum((sim_data[:, 1]) .* log.(res.p_hat[sim_data[:, 2] .+ 1 ]) + (1 .- sim_data[:,1]) .* log.((1 .- res.p_hat[sim_data[:, 2] .+ 1])))
    end
    val
end

t1 = ccp_iteration(prim, res, -10)
t2 = ccp_iteration(prim, res, -4)