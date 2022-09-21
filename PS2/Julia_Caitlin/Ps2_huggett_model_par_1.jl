#Comp 899 PS2 Caitlin Dutta
#keyword-enabled structure to hold model primitives
@everywhere using Parameters, Plots, SharedArrays
@everywhere @with_kw struct Primitives
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #coefficient of rr a
    a_min::Float64 = -2 #asset lower bound
    a_max::Float64 = 5 #asset upper bound
    na::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) #capital grid
    markov::Array{Float64,2} = [0.97 0.03 ; 0.5 0.5] #transition matrix
    s_grid::Array{Float64,1} = [1, 0.5] #unemployed/employed earnings shocks
    ns::Int64 = length(s_grid)
    q::Float64 = 0.99 #starting price
end

#structure that holds model results
@everywhere mutable struct Results
    val_func::SharedArray{Float64,2} #value function
    pol_func::SharedArray{Float64,2} #policy function
    pol_func_ind::SharedArray{Int64,2} #policy function
    μ::SharedArray{Float64} #invar_dist
    Q::SharedArray{Float64} #endog transition matrix
end

#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = SharedArray{Float64}(zeros(prim.na, prim.ns)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.na, prim.ns)) #initial policy function guess
    pol_func_ind = SharedArray{Int64}(zeros(prim.na, prim.ns))
    Q = Q_finder(prim, res) #empty transition matrix
    μ = create_mu0(prim) #guessed initial distribution
    res = Results(val_func, pol_func, pol_func_ind, μ, Q) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack a_grid, s_grid, β, α, na, ns, markov, q = prim #unpack model primitives
    v_next = SharedArray{Float64}(zeros(na, ns)) #next guess of value function to fill

    #choice_lower = 1 #for exploiting monotonicity of policy function
    @sync @distributed for i = 1:na*ns #loop over a/s states
        s_index = mod(i,ns) + 1
        a_index = mod(ceil(Int64, i/ns), na) + 1
        a, s = a_grid[a_index], s_grid[s_index] #value of a and s
        candidate_max = -Inf #bad candidate max
        budget = s + a #budget
        choice_lower = 1
        for ap_index in choice_lower:na #loop over possible selections of a',we dont choose s so we dont loop over choices of it
            ap = a_grid[ap_index]
            c = budget -  q*ap #consumption given a' selection
            if c>0 #check for positivity
                val = ((1/sqrt(c) -1)/(1-α)) + β * sum(res.val_func[ap_index,:].*markov[s_index,:]) #compute value, expectation over s'
                if val>candidate_max #check for new max value
                    candidate_max = val #update max value
                    res.pol_func[a_index, s_index] = ap #update policy function
                    res.pol_func_ind[a_index, s_index] = ap_index #update policy function
                    choice_lower = ap_index #update lowest possible choice
                end
            end
        end
        v_next[a_index, s_index] = candidate_max #update value function
    end
    v_next #return next guess of value function
end

#Value function iteration
@everywhere function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
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
@everywhere function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end

## T*
    @everywhere function create_mu0(prim::Primitives)
        @unpack na, ns = prim #unpack model primitives
        μ_0 = SharedArray{Float64}(zeros(na, ns))
            for j=1:na
                for si = 1:2
                    μ_0[j, si] = (1)/(2*na) #* Π_st[si]
                end
            end
        μ_0
    end

    @everywhere function Q_finder(prim::Primitives, res::Results) #find Q
        @unpack  na, markov = prim
        pf_ind = res.pol_func_ind #policy index matrix
        Q = zeros(na, 2, na, 2)
        for sp_i = 1:2
            for ap_i = 1:na
                ap_choosers = findall(==(ap_i), pf_ind) #find all indices a_i, s_i which lead to choice of ap
                for x in ap_choosers #iterate over each element
                    ai = x[1]
                    si = x[2]
                    Q[ai, si, ap_i, sp_i] = markov[si, sp_i]
                end
            end
        end
        Q
    end


    @everywhere function create_mu1(prim::Primitives,res::Results)
        @unpack na, ns = prim #unpack model primitives
        @unpack pol_func, pol_func_ind = res #unpack model primitives
        μ_0 = res.μ
        Q = res.Q
        μ_1 = SharedArray{Float64}(zeros(na, ns))
        for ap = 1:na
            val_e = 0.0
            val_u = 0.0
            for ai = 1:na
                for si = 1:ns
                    val_e = μ_0[ai, si] * Q[ai, si, ap, 1] #emp
                    val_u = μ_0[ai, si] * Q[ai, si, ap, 2] #unemp
                end
            end
            μ_1[ap, 1] = val_e
            μ_1[ap, 2] = val_u
        end
        μ_1
    end




    @everywhere function Stationary_Dist(prim::Primitives,res::Results)
        @unpack na, ns = prim #unpack model primitives
        res.Q = Q_finder(prim::Primitevs, res::Results)
        supnorm = 1
        n = 0
        while supnorm > 1e-3
            n += 1
            μ_1 = create_mu1(prim, res)
            supnorm = maximum(abs.(μ_1 - res.μ))/maximum(abs.(res.μ))
            res.μ = copy(μ_1)
            println("supnorm =", supnorm)
            if mod(n, 100) == 0
                 println(n, ": ", supnorm)
             end
        end
        res.μ
        println(" dist converged in ", n, " iterations.")
    end

#=
     for i = 1:na*ns #loop over a/s states
                si = mod(i,ns) + 1
                ai = mod(ceil(Int64, i/ns), na) + 1
                    for sp = 1:ns
                        μ_1[res.pol_func_ind[ai,si], sp] += μ_0[ai, si] * Q[ai, si,  sp]
                    end
                μ_0 = μ_1
=#


    ##############################################################################
