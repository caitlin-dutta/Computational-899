@everywhere using Parameters, Plots

@everywhere @with_kw struct Primitives
    β :: Float64 = 0.9932 #discount rate
    α :: Float64 = 1.5 #crra parameter
    S :: Array{Float64,1} = [1, 0.5] #possible earning levels
    Π :: Array{Float64} = [0.97 0.03; 0.5 0.5] #transition matrix
    Π_st :: Array{Float64} = [0.9434, 0.0566] #change back
    A :: Array{Float64, 1} = collect(range(-2.0, length = 1000, stop= 5.0)) #asset grid
    na :: Int64 = length(A) #number of grid points
end

@everywhere mutable struct Results
    val_func :: SharedArray{Float64,2} #value function struct
    pol_func :: SharedArray{Float64,2} #policy function struct
    pol_func_ind :: SharedArray{Int64,2} #policy function index struct
    q :: Float64 #bond price 
    μ :: SharedArray{Float64} #invar_dist
    Q :: SharedArray{Float64} #endog transition matrix
end

@everywhere function mu_guess(prim::Primitives)
    @unpack A,na, Π_st = prim
    μ_0 = zeros(na, 2)
    for j=1:na
        for si = 1:2
            μ_0[j, si] = (1)/(2*na) #* Π_st[si]
        end
    end
    μ_0
end

@everywhere function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.na, 2) #initial value function guess
    pol_func = zeros(prim.na, 2) #initial policy function guess
    pol_func_ind = ones(Int64, prim.na,2) #initial policy function guess index
    q = (1 + prim.β)/2 #initial price guess
    μ = mu_guess(prim) #guessed initial distribution
    Q = zeros(prim.na, 2, prim.na, 2) #empty transition matrix
    res = Results(val_func, pol_func, pol_func_ind, q, μ, Q) #initialize results struct
    prim, res #return structs
end

@everywhere function Bellman(prim::Primitives, res::Results, q::Float64) #bellman operator which takes primitives, results struct, employment status, and the market price as given
    @unpack β, α, S, Π, A, na = prim
    v_next = SharedArray{Float64}(zeros(na, 2))

    #s = S[s_i] #find current income given employment status index
    @sync @distributed for a_i=1:na
        a = A[a_i] #given current asset index, find asset holding in grid
        val_prev_h = -1e16 #v bad number to start as candidate maximum for employed agent
        val_prev_l = -1e16 #same, but for unemployed agent
        pol_cand_h = 1 #initial policy choice for employed
        pol_cand_l = 1 #initial policy choice for unemployed
        pol_cand_ind_h = 1 #corresponding index for employed
        pol_cand_ind_l = 1 #same for unemployed
        budget_h = S[1] + a #find the employed agent's budget given income and asset holdings
        budget_l = S[2] + a #find the unemployed agent's budget given income and asset holdings
        h_done = 0
        l_done = 0
        for ap_i=1:na #iterate over index of next period asset holdings
            ap = A[ap_i] #find corresponding asset level
            if h_done ==0
                c_h = budget_h - q*ap #given the employed agent's budget and choice of a prime, find resulting consumption
                val_h = -1e16
                if c_h > 0
                    val_h = -2*(1/sqrt(c_h) -1) #set value equal to the utility of consumption
                    for s_i_n=1:2 #iterate over index of employment status in next period
                        val_h += β*Π[1, s_i_n]*res.val_func[ap_i, s_i_n] #add the probability of state s_i_n times the value of assets ap and status s_i_n, discounted by beta
                    end
                end
                if val_h > val_prev_h #check if the value from choosing ap_i is higher than the value from choosing ap_i-1 
                    val_prev_h = val_h
                    pol_cand_h = ap
                    pol_cand_ind_h = ap_i
                else
                    h_done =1
                end
            end
            if l_done ==0
                c_l = budget_l - q*ap #given the employed agent's budget and choice of a prime, find resulting consumption
                val_l = -1e16
                if c_l > 0
                    val_l = -2*(1/sqrt(c_l) -1) #set value equal to the utility of consumption
                    for s_i_n=1:2 #iterate over index of employment status in next period
                        val_l += β*Π[2, s_i_n]*res.val_func[ap_i, s_i_n] #add the probability of state s_i_n times the value of assets ap and status s_i_n, discounted by beta
                    end
                end
                if val_l > val_prev_l #check if the value from choosing ap_i is higher than the value from choosing ap_i-1 
                    val_prev_l = val_l
                    pol_cand_l = ap
                    pol_cand_ind_l = ap_i
                else
                    l_done =1
                end
            end
        end
        res.pol_func[a_i, 1] = pol_cand_h
        res.pol_func_ind[a_i, 1] = pol_cand_ind_h
        res.pol_func[a_i, 2] = pol_cand_l
        res.pol_func_ind[a_i, 2] = pol_cand_ind_l
        v_next[a_i, 1] = val_prev_h
        v_next[a_i, 2] = val_prev_l
    end
    v_next
end

@everywhere function solve_vf(prim::Primitives, res::Results)
    error = 100
    tol = 1e-6 #used 1e-8 for mkt clearing
    n=0
    q = res.q
    #v_next= zeros(prim.na, 2)
    while error > tol
        n+=1
        v_next = Bellman(prim, res, q)
        error = maximum(abs.(res.val_func .- v_next))/maximum(abs.(v_next))
        res.val_func = v_next #update value function held in results vector
        #println(n, "  ",  error) #iteration number and error level
    end
    println("Convergence in $(n) iterations!")
end

@everywhere function Q_finder(prim::Primitives, res::Results) #find Q
    @unpack A, na, Π = prim
    pf_ind = res.pol_func_ind #policy index matrix
    Q = zeros(na, 2, na, 2)
    for sp_i = 1:2
        for ap_i = 1:na
            ap_choosers = findall(==(ap_i), pf_ind) #find all indices a_i, s_i which lead to choice of ap 
            for x in ap_choosers #iterate over each element
                ai = x[1]
                si = x[2]
                Q[ai, si, ap_i, sp_i] = Π[si, sp_i]
            end
        end
    end
    Q
end
            


@everywhere function next_mu_finder(prim::Primitives, res::Results)
    @unpack A, na, Π, Π_st = prim
    μ_0 = res.μ #unpack the distribution mu
    Q = res.Q
    μ_1 = SharedArray{Float64}(zeros(na, 2))
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

@everywhere function invar_dist(prim::Primitives, res::Results)
    @unpack A, na, Π, Π_st = prim
    println("Finding Q:")
    res.Q = Q_finder(prim::Primitives, res::Results)
    println("Found Q")
    error = 100
    tol = 1e-6 #used -10 for mkt clearing
    n=0
    while error > tol
        n+=1
        μ_1 = next_mu_finder(prim, res)
        error = maximum(abs.(μ_1 - res.μ))/maximum(abs.(res.μ)) 
        res.μ = copy(μ_1) #update the distribution stored in results struct
        if mod(n, 100) == 0
            println(n, ": ", error)
        end
    end
    print("Found it in $(n) tries!")
    res.μ
end


@everywhere function excess_demand(prim::Primitives, res::Results)
    @unpack A, na = prim
    #q = res.q
    #prim, res = Initialize()
    @elapsed solve_vf(prim, res)
    μ = invar_dist(prim, res)
    excess = 0.0
    for a_i = 1:na
        for s_i = 1:2
            excess += res.pol_func[a_i, s_i]*μ[a_i, s_i]
        end
    end
    excess
end 

@everywhere function market_clearing(prim::Primitives, res::Results)
    @unpack β = prim
    q_high = 1.0 #upper bound for discount bond rate; since it is a discount bond, it will be less than 1
    q_low = β #by assumption, the discount bond rate must exceed beta
    res.q = (q_low + q_high)/2
    excess = excess_demand(prim, res)
    println("Price: $(res.q), Excess Demand: $(excess)")
    tol = 1e-4 
    while abs(excess) > tol
        if excess > 0
            q_low = res.q
        else
            q_high = res.q
        end
        res.q = (q_low + q_high)/2
        excess = excess_demand(prim, res)
        println("Price: $(res.q), Excess Demand: $(excess)")
    end
    res.q
end

@everywhere function wealth_dist(prim::Primitives, res::Results)
    @unpack A, na = prim
    invar_dist(prim, res)
    μ = res.μ
    wealth_dist = zeros(na, 2)
    for a_i = 1:na
        wealth_i_h = min(a_i + 143, 1000) #since the range has increments of (roughly) 0.007, adding one to the level of assets corresponds to adding 1/0.007 = 143 to the index of asset holdings 
        wealth_i_l = min(a_i + 71, 1000) #similarly, we add 71 to the index a_i to add 0.5 
        #note: minimum operator included to ensure we don't go out of bounds with the index. Since there is zero mass of agents with assets more than 2, this has no practical impact and is only included to ensure we don't have a bounds error in the next two lines
        wealth_dist[wealth_i_h, 1] = μ[a_i, 1] #set measure at wealth level corresponding to a_i when employed to the mass with assets a_i
        wealth_dist[wealth_i_l, 2] = μ[a_i, 2] #same as above, but for unemployed
    end
    wealth_dist
end

function lorenz_curve(prim::Primitives, res::Results, w_dist::Array{Float64,2})
    @unpack A, na = prim
    total_wealth_dist = w_dist[:,1] + w_dist[:,2] #the total mass of agents with wealth level w is the sum of the measures of employed and unemployed with wealth w
    lorenz_num=zeros(na)
    prop_lower = zeros(na)
    lorenz_num[1] = total_wealth_dist[1]*A[1]
    prop_lower[1] = total_wealth_dist[1]
    for a_i=2:na
        a = A[a_i]
        lorenz_num[a_i] = total_wealth_dist[a_i] * a + lorenz_num[a_i-1]
        prop_lower[a_i] = total_wealth_dist[a_i] + prop_lower[a_i-1]
    end
    x_grid = prop_lower ./ prop_lower[na]
    lorenz = lorenz_num ./ lorenz_num[na]
    x_grid, lorenz
end

@everywhere function gini_finder(x_grid::Vector{Float64}, lorenz::Vector{Float64})
    unscaled_gini = 0.0
    for i=2:length(x_grid)
        delta_x = x_grid[i]-x_grid[i-1]
        unscaled_gini += (x_grid[i] - lorenz[i])*delta_x
    end
    gini = unscaled_gini*2 #the area under the 45 degree line is 1/2. The Gini coefficient is the ratio of the area between the 45-degree line and the lorenz curve over the area under the 45-degree line. So we divide the area between the curves by the total area under the 45 degree line, which is 1/2. Equivalently, we multiply by 2
    gini
end


@everywhere function welfare_comparison(prim::Primitives, res::Results)
    @unpack A, na, α, β = prim
    λ = zeros(na, 2)
    μ = res.μ
    wfb = -4.252556
    ab_frac = 1/((1-α)*(1-β))
    val_func = res.val_func
    num = wfb + 1 /((1 - α)*(1 - β))
    denom = val_func .+ (1 ./((1 .- α).*(1 .- β)))

    λ = (num./denom).^(1/(1 .- α)) .- 1
    λ
end
