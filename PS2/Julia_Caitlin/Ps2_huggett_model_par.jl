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
end

#structure that holds model results
@everywhere mutable struct Results
    val_func::SharedArray{Float64,2} #value function
    pol_func::SharedArray{Float64,2} #policy function
    pol_func_ind::SharedArray{Int64,2} #policy function
    μ::SharedArray{Float64} #invar_dist
    Q::SharedArray{Float64} #endog transition matrix
    q::Float64
    #xd::Float64
end

#guess initial dist
@everywhere function create_mu0(prim::Primitives)
    @unpack na, ns = prim #unpack model primitives
    μ_0 = zeros(na, ns)
        for j=1:na
            for si = 1:2
                μ_0[j, si] = (1)/(2*na) #* Π_st[si]
            end
        end
    μ_0
end


#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = SharedArray{Float64}(zeros(prim.na, prim.ns)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.na, prim.ns)) #initial policy function guess
    pol_func_ind = SharedArray{Int64}(ones(prim.na, prim.ns))
    Q = zeros(prim.na, 2, prim.na, 2) #empty transition matrix
    μ = create_mu0(prim) #guessed initial distribution
    q = 0.99 #starting q guess
#    xd = 0 #starting xd
    res = Results(val_func, pol_func, pol_func_ind, μ, Q, q) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
@everywhere function Bellman(prim::Primitives,res::Results)
    @unpack val_func, q = res #unpack value function
    @unpack a_grid, s_grid, β, α, na, ns, markov= prim #unpack model primitives
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

@everywhere function Q_finder(prim::Primitives, res::Results) #find Q
    @unpack na, markov= prim
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
        @unpack μ, Q = res
        #@unpack pol_func, pol_func_ind = res #unpack model primitives
        μ_0 = μ
        #Q = Q
        μ_1 = SharedArray{Float64}(zeros(na, ns))
        @sync @distributed for ap_i =1:na
             val_h = 0.0
             val_l = 0.0
             for a_i = 1:na
                 for s_i = 1:2
                     val_h += Q[a_i, s_i, ap_i, 1]*μ_0[a_i, s_i] #employed
                     val_l += Q[a_i, s_i, ap_i, 2]*μ_0[a_i, s_i] #unemployed
                 end
             end
             μ_1[ap_i, 1] = val_h
             μ_1[ap_i, 2] = val_l
         end
         μ_1
    end

    @everywhere function Stationary_Dist(prim::Primitives,res::Results)
        @unpack a_grid, markov, na, ns = prim #unpack model primitives
        @unpack Q,  μ = res
        println("finding ms. Q")
        res.Q = Q_finder(prim::Primitives, res::Results)
        println("found ms. Q")
        supnorm = 100
        tol = 1e-4
        n = 0
        #res.μ = create_mu0(prim::Primitives)
        while supnorm > tol
            n += 1
            μ_1 = create_mu1(prim, res)
            supnorm = maximum(abs.(μ_1 - res.μ))/maximum(abs.(res.μ))
            res.μ = μ_1
            #println("iteration ", n, " supnorm = ", supnorm)
                if mod(n, 100) == 0
                     println(n, ": ", supnorm)
                 end
        end
        println(" dist converged in ", n, " iterations.")
        #res.μ = μ
        #=check = 0
        for ai = 1:na
            for si = 1:ns
                check += res.μ[ai, si]
            end
        end=#
    end

##
#Asset market clearing
    @everywhere function excess_demand(prim::Primitives, res::Results)
        @unpack na, ns = prim
        @unpack pol_func, μ, q = res
        xd = 0
        for ai = 1:na
            for si = 1:ns
                xd += (pol_func[ai, si] * μ[ai, si])
            end
        end
        println("excess demand is", xd)
        xd
    end

    @everywhere function findq(prim::Primitives, res::Results)
        @unpack q = res
        @unpack β = prim
        xd = excess_demand(prim, res) #starting xd
        n = 0
        tol = 1e-2
        q_low = β
        q_high = 1
        while abs(xd) > tol
            n += 1
        #    adjustment_step = 0.1*q
            if xd>0
                q_low = res.q
            #    res.q = q+adjustment_step
            elseif xd<0
                #res.q = q-adjustment_step
                q_high = res.q
            end
            res.q = (q_high + q_low)/2
            println("new q is", res.q)
            Solve_model(prim, res)
            Stationary_Dist(prim, res)
            xd = excess_demand(prim,res)
            println("iteration #", n)
        end
        println("xd < 1e-4, market clears at q = ", res.q)
    end
#q = 0.9942768177032469

##############################################################################
