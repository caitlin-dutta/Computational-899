#Comp 899 PS2 Caitlin Dutta
#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #coefficient of rr a
    a_min::Float64 = -2 #asset lower bound
    a_max::Float64 = 5 #asset upper bound
    na::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) #capital grid
    markov::Array{Float64,2} = [0.97 0.03 ; 0.5 0.5] #transition matrix
    s_grid::Array{Float64,1} = [1, 0.5] #unemployed/employed earnings shocks
    ns::Int64 = length(s_grid)
    q::Float64 = 0.9 #starting price
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64,2} #value function
    pol_func::Array{Float64,2} #policy function
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.na, prim.ns) #initial value function guess
    pol_func = zeros(prim.na, prim.ns) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack a_grid, s_grid, β, α, na, ns, markov, q = prim #unpack model primitives
    v_next = zeros(na, ns) #next guess of value function to fill

    #choice_lower = 1 #for exploiting monotonicity of policy function
    for a_index = 1:na, s_index = 1:ns #loop over a/s states
        a, s = a_grid[a_index], s_grid[s_index] #value of a and s
        candidate_max = -Inf #bad candidate max
        budget = s + a #budget
        choice_lower = 1
        for ap_index in choice_lower:na #loop over possible selections of a',we dont choose s so we dont loop over choices of it
            ap = a_grid[ap_index]
            c = budget -  q*ap #consumption given a' selection
            if c>0 #check for positivity
                val = ((c^(1-α) -1)/(1-α)) + β * sum(res.val_func[ap_index,:].*markov[s_index,:]) #compute value, expectation over s'
                if val>candidate_max #check for new max value
                    candidate_max = val #update max value
                    res.pol_func[a_index, s_index] = ap #update policy function
                    choice_lower = ap_index #update lowest possible choice
                end
            end
        end
        v_next[a_index, s_index] = candidate_max #update value function
    end
    v_next #return next guess of value function
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 #counter
    err = 100

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = (maximum(abs.(v_next.-res.val_func))/abs(v_next[prim.na, 1])) #reset error level
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
