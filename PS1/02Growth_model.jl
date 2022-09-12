#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 75.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    markov::Array{Float64,2} = [0.977 0.023 ; 0.074 0.926] #transition matrix
    z_grid::Array{Float64,1} = [1.25, 0.2] #good/bad zeta state space, the zetas will affect the budget and kapital choices so we need to add to bellman
    nz::Int64 = length(z_grid)
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nk, prim.nz) #initial value function guess
    pol_func = zeros(prim.nk, prim.nz) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack k_grid, z_grid, β, δ, α, nk, nz, markov = prim #unpack model primitives
    v_next = zeros(nk, nz) #next guess of value function to fill

    #choice_lower = 1 #for exploiting monotonicity of policy function
    for k_index = 1:nk, z_index = 1:nz #loop over k/z states
        k, z = k_grid[k_index], z_grid[z_index] #value of k and z
        candidate_max = -Inf #bad candidate max
        budget = z*k^α + (1-δ)*k #budget

        for kp_index in 1:nk #loop over possible selections of k',we dont choose z so we dont loop over choices of it
            kp = k_grid[kp_index]
            c = budget - kp #consumption given k' selection
            if c>0 #check for positivity
                val = log(c) + β * sum(res.val_func[kp_index,:].*markov[z_index,:]) #compute value, expectation over z'
                if val>candidate_max #check for new max value
                    candidate_max = val #update max value
                    res.pol_func[k_index, z_index] = kp #update policy function
                    #choice_lower = kp_index #update lowest possible choice
                end
            end
        end
        v_next[k_index, z_index] = candidate_max #update value function
    end
    v_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter
    err = 100

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = abs.(maximum(v_next.-res.val_func))/abs(v_next[prim.nk, 1]) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
