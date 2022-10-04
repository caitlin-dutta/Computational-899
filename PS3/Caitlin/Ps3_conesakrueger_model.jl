#Comp 899 PS2 Caitlin Dutta
#keyword-enabled structure to hold model primitives
@everywhere using Parameters, Plots, SharedArrays
@everywhere @with_kw struct Primitives
    β::Float64 = 0.97 #discount rate
    γ::Float64 = 0.42 #coefficient of rr a
    w::Float64 = 1.05 #wage
    r::Float64 = 0.05 #rental rate
    b::Float64 = 0.2 #ss benefit
    δ::Float64 = 0.06 #depreciation
    α::Float64 = 0.36 #capital share
    θ::Float64 = 0.11 #SS tax on labor
    σ::Float64 = 2 #Coeff RRA
    a_min::Float64 = .0001 #asset lower bound
    a_max::Float64 = 100 #asset upper bound
    na::Int64 = 2000 #number of asset grid points
    N::Int64 = 66 #model length
    Nr::Int64 = 46 #retirement age
    η::Vector{Float64} = [0.59923239,0.63885106 ,0.67846973 ,0.71808840 ,0.75699959 ,0.79591079 ,0.83482198 ,0.87373318 ,0.91264437 ,0.95155556 ,0.99046676 ,0.99872065
    ,1.0069745 ,1.0152284 ,1.0234823 ,1.0317362 ,1.0399901 ,1.0482440 ,1.0564979 ,1.0647518 , 1.0730057 ,1.0787834 ,1.0845611 ,1.0903388 , 1.0961165 , 1.1018943
    ,1.1076720 , 1.1134497 ,1.1192274 , 1.1250052 ,1.1307829 ,1.1233544 ,1.1159259 ,1.1084974 ,1.1010689 ,1.0936404 ,1.0862119 ,1.0787834 ,1.0713549 ,1.0639264
    ,1.0519200,1.0430000,1.0363000,1.0200000,1.0110000]
    #η::Array{Float64,1} = map(x->parse(Float64,x), readlines("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS3/Caitlin/ef.txt"))
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) #capital grid
    markov::Array{Float64,2} = [0.9261 0.0739 ; 0.0189 0.9811] #transition matrix
    s_grid::Array{Float64,1} = [3, 0.5] #high/low productivity shocks
    ns::Int64 = length(s_grid)
end


#structure that holds model results
@everywhere mutable struct Results
    val_func::SharedArray{Float64,3} #value function
    pol_func::SharedArray{Float64,3} #policy function
    pol_func_ind::SharedArray{Int64,3} #policy function
    labor::SharedArray{Float64, 3} #labor choices
    budg::SharedArray{Float64, 3} #budg choices
    F::SharedArray{Float64, 3} #F dist
end

#F' = F0 + F*s'

#function for initializing model primitives and results
@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    pol_func_ind = SharedArray{Int64}(ones(prim.N, prim.na, prim.ns))
    labor = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    budg = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    F = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #F dist
    res = Results(val_func, pol_func, pol_func_ind, labor, budg, F) #initialize results struct
    prim, res #return deliverables
end

@everywhere function Util_R(prim::Primitives, c::Float64)
    @unpack a_grid, β, r, b, σ, γ, na, ns, N, Nr, markov = prim #unpack model primitives
    util_R = c^((1-σ)*γ)/(1-σ)
    util_R
end

#Bellman Operator
@everywhere function Bellman_retired(prim::Primitives,res::Results)
    @unpack val_func = res #unpack value function
    @unpack a_grid, β, r, b, σ, γ, na, ns, N, Nr, markov = prim #unpack model primitives
    #Backwards iteration from period before retiring
    for n = reverse(Nr:N) #backwards iteration
        if n == N
            for ai = 1:na #loop over a states
            for si = 1:ns
            a = a_grid[ai]
            budget = (1+r)*a + b
            c = budget
            res.val_func[n,ai,si] = Util_R(prim,c) #in pd N agent consumes all remaining a
            res.pol_func[n,ai,si] = 0 #in pd N agent consumes all remaining a
            end
            end
        elseif n != N
            for ai = 1:na #loop over a states
            for si = 1:ns
            a = a_grid[ai]
            candidate_max = -Inf
            budget = (1+r)*a + b
            #choice_lower = 1
            for api = 1:na #loop over potential a'
                ap = a_grid[api]
                c = budget - ap
                if c>0 #check for positivity
                    val = Util_R(prim,c) + β * (res.val_func[n+1,api,si]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[n, ai, si] = ap #update policy function
                        res.pol_func_ind[n, ai, si] = api #update policy function
                    #    choice_lower = api #update lowest possible choice
                    end #if val
                elseif c<= 0
                    val = -Inf
                end #c if
            end #api
        res.val_func[n, ai, si] = candidate_max #update value function
    end #si
    end #ai
    end #if
    end #n
    val_func
end #func

@everywhere function Util_W(prim::Primitives, c::Float64, l::Float64)
    @unpack a_grid, s_grid, β, w, r, b, σ, γ, η, θ, na, ns, N, Nr, markov = prim #unpack model primitives
    util_w = ((c^γ)*(1-l)^(1-γ))^(1-σ)/(1-σ)
    util_w
end

@everywhere function lsupply(prim::Primitives, ap::Float64, aep::Float64, a::Float64)
    @unpack a_grid, s_grid, β, w, r, b, σ, γ, η, θ, na, ns, N, Nr, markov = prim #unpack model primitives
    l = ((γ*(1-θ)*aep*w) - ((1-γ)*((1+r)*a - ap)))  / ((1-θ)*w*aep)
    l
end

#Bellman Operator
@everywhere function Bellman_worker(prim::Primitives,res::Results)
    @unpack val_func, labor, budg = res #unpack value function
    @unpack a_grid, s_grid, β, w, r, b, σ, γ, η, θ, na, ns, N, Nr, markov = prim #unpack model primitives
    #Backwards iteration from period before retiring
    for n = reverse(1:(Nr-1)) #backwards iteration
    for ai = 1:na #loop over a
        for si = 1:ns
        s = s_grid[si]
        a = a_grid[ai]
        candidate_max = -Inf
        #choice_lower = 1
            for api = 1:na #loop over potential a'
                ap = a_grid[api]
                aep = s*η[n]
                l::Float64 = lsupply(prim,ap,aep,a)
                if l < 0
                    l = 0
                end
                if l > 1
                    l = 1
                end
                budget = w*(1-θ)*aep*l + (1+r)*a
                c::Float64 = budget - ap
                if c>0 #check for positivity
                    util_w = Util_W(prim,c,l)
                    val = util_w + β*sum(res.val_func[n+1,api,:].*markov[si,:]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[n, ai, si] = ap #update policy function
                        res.pol_func_ind[n, ai, si] = api #update policy function
                        labor[n,ai,si] = l #update labor
                        budg[n,ai,si] = budget
                    #    choice_lower = api #update lowest possible choice
                    end #if val
                elseif c <= 0
                    val = -Inf
                end #c if
            end #api
        res.val_func[n, ai, si] = candidate_max #update value function
        res.labor[n,ai,si]  = labor[n,ai,si]
        res.budg[n,ai,si]  = budg[n,ai,si]
    end #si
end #ai
end #n
    val_func
end #func


#solve the model
@everywhere function Solve_model(prim::Primitives, res::Results)
    Bellman_retired(prim, res) #spit out new vectors
    Bellman_worker(prim,res)
end

## T*

#find size of each cohort, growth rate means more young than old
#start w guess that initial size = 1, then decay to get every age
#add up and then divide all by sum so total size of mu = 1

    @everywhere function Cohort_size(prim::Primitives, res::Results)
        @unpack N = prim
        c_size = zeros(66)
        pg = 0.011
        c_size[1] = 1
        for i = 2:N
            c_size[i] = c_size[i-1]/(1+pg)
        end
        c_sum = sum(c_size)
        c_size_n = c_size./c_sum
        c_size_n
    end


    @everywhere function create_F(prim::Primitives,res::Results)
        @unpack na, ns, N, markov = prim #unpack model primitives
        @unpack pol_func, pol_func_ind = res #unpack model primitives
        c_size_n = Cohort_size(prim, res)
        F = (zeros(prim.N, prim.na, prim.ns))
        println(sum(c_size_n))
        F[1,1,1] = c_size_n[1]*0.2037
        F[1,1,2] = c_size_n[1]*0.7963
        #[age,asset,state]
        for n = 2:N
            for ai = 1:na
                for si = 1:ns #looping through coming from sh and sl
                    aprevi = pol_func_ind[n-1,ai,si]
                    for spi = 1:ns
                        F[n,aprevi,spi] += (F[n-1,ai,si] * markov[si,spi])/(.011+1)
                    end
                end
            end
        end
        res.F = F
    end

#=
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

=#


##############################################################################
