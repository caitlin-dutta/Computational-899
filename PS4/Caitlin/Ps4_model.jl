#Comp 899 PS2 Caitlin Dutta
#keyword-enabled structure to hold model primitives
using Parameters, Plots, SharedArrays
@with_kw struct Primitives
    β::Float64 = 0.97 #discount rate
    γ::Float64 = 0.42 #coefficient of rr a
    δ::Float64 = 0.06 #depreciation
    α::Float64 = 0.36 #capital share
    θ::Float64 = 0.11 #SS tax on labor
    σ::Float64 = 2 #Coeff RRA

    a_min::Float64 = .0001 #asset lower bound
    a_max::Float64 = 100 #asset upper bound
    na::Int64 = 500 #number of asset grid points
    N::Int64 = 66 #model length
    Nr::Int64 = 46 #retirement age
    η::Vector{Float64} = [0.59923239,0.63885106 ,0.67846973 ,0.71808840 ,0.75699959 ,0.79591079 ,0.83482198 ,0.87373318 ,0.91264437 ,0.95155556 ,0.99046676 ,0.99872065
    ,1.0069745 ,1.0152284 ,1.0234823 ,1.0317362 ,1.0399901 ,1.0482440 ,1.0564979 ,1.0647518 , 1.0730057 ,1.0787834 ,1.0845611 ,1.0903388 , 1.0961165 , 1.1018943
    ,1.1076720 , 1.1134497 ,1.1192274 , 1.1250052 ,1.1307829 ,1.1233544 ,1.1159259 ,1.1084974 ,1.1010689 ,1.0936404 ,1.0862119 ,1.0787834 ,1.0713549 ,1.0639264
    ,1.0519200,1.0430000,1.0363000,1.0200000,1.0110000]

    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) #capital grid
    markov::Array{Float64,2} = [0.9261 0.0739 ; 0.0189 0.9811] #transition matrix
    s_grid::Array{Float64,1} = [3, 0.5] #high/low productivity shocks
    ns::Int64 = length(s_grid)
    pg::Float64 = 0.011

    T::Int64 = 30
end


#structure that holds model results
 mutable struct Results
    val_func::Array{Float64,3} #value function
    pol_func::Array{Float64,3} #policy function
    pol_func_ind::Array{Int64,3} #policy function
    labor::Array{Float64, 3} #labor choices
    budg::Array{Float64, 3} #budg choices
    F::Array{Float64, 3} #F dist
    val_next::Array{Float64, 3} #val_next
    #c_size_n::Array{Float64, 1}
    w::Float64  #wage
    r::Float64  #rental rate
    b::Float64  #ss benefit
end

mutable struct Results4
    gamma::Array{Float64, 4}
    wage::Array{Float64, 1} #for every T there is one w
    rate::Array{Float64, 1}
    kap_agg::Array{Float64, 1}
    lab_agg::Array{Float64, 1}
    val_init::Array{Float64, 3} #value fn at t = 0
    lab_path::Array{Float64, 4}
    pol_path::Array{Float64, 4}
    pol_ind_path::Array{Float64, 4}
    mu::Array{Float64, 4}
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives()
    val_func = Array{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial value function guess
    pol_func = Array{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    pol_func_ind = Array{Int64}(ones(prim.N, prim.na, prim.ns))
    labor = Array{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    budg = Array{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    F = Array{Float64}(zeros(prim.N, prim.na, prim.ns)) #F dist
    val_next = Array{Float64}(zeros(prim.N, prim.na, prim.ns))
    #c_size_n = Cohort_size(prim)
    w = 0.0
    r = 0.0
    b = 0.0
    res = Results(val_func, pol_func, pol_func_ind, labor, budg, F, val_next, w, r, b)
    prim, res
end

function Initialize4(prim::Primitives)
    gamma = Array{Float64}(zeros(prim.T, prim.N, prim.na, prim.ns))
    wage = Array{Float64}(zeros(prim.T))
    rate = Array{Float64}(zeros(prim.T))
    kap_agg = Array{Float64}(zeros(prim.T))
    lab_agg = Array{Float64}(zeros(prim.T))
    val_init = Array{Float64}(zeros(prim.N, prim.na, prim.ns))
    pol_path = Array{Float64}(zeros(prim.T, prim.N, prim.na, prim.ns))
    lab_path = Array{Float64}(zeros(prim.T, prim.N, prim.na, prim.ns))
    pol_ind_path = Array{Float64}(zeros(prim.T, prim.N, prim.na, prim.ns))
    mu = Array{Float64}(zeros(prim.T, prim.N, prim.na, prim.ns))
    res4 = Results4(gamma, wage, rate, kap_agg, lab_agg, val_init, pol_path, lab_path, pol_ind_path, mu)
    res4 #return deliverables
end


function Cohort_size(prim::Primitives)
    c_size = zeros(66)
    c_size[1] = 1
    for i = 2:66
        c_size[i] = c_size[i-1]/(1+0.011)
    end
    c_size_n = c_size./sum(c_size)
    c_size_n
end

function Util_R(prim::Primitives, c::Float64, r::Float64, b::Float64, θ::Float64)
    @unpack σ, γ = prim #unpack model primitives
    util_R = c^((1-σ)*γ)/(1-σ)
    util_R
end

function Util_W(prim::Primitives, c::Float64, l::Float64)
    @unpack σ, γ = prim #unpack model primitives
    util_w = ((c^γ)*((1-l)^(1-γ)))^(1-σ)/(1-σ)
    util_w
end

function lsupply(prim::Primitives, ap::Float64, aep::Float64, a::Float64, w::Float64, r::Float64, θ::Float64)
    @unpack σ, γ, η, θ = prim #unpack model primitives
    lsupply = ((γ*(1-θ)*aep*w) - ((1-γ)*((1+r)*a - ap)))  / ((1-θ)*w*aep)
    lsupply
end

#Bellman Operator

function Bellman(K::Float64, L::Float64, θ::Float64, prim::Primitives, res::Results)
    @unpack a_grid, s_grid, β, α, δ, σ, γ, η, na, ns, N, Nr, markov = prim #unpack model primitives
    c_size_n = Cohort_size(prim)
    w = (1-α)*K^(α)*L^(-α)
    r = α*K^(α-1)*L^(1-α) - δ
    b = (θ*w*L)/sum(c_size_n[Nr:N])
    #Backwards iteration from period before retiring
    for n = reverse(Nr:N) #backwards iteration
        if n == N
            for ai = 1:na #loop over a states
            for si = 1:ns
            a = a_grid[ai]
            budget = (1+r)*a + b
            c = budget
            res.val_func[n,ai,si] = Util_R(prim,c,r,b,θ) #in pd N agent consumes all remaining a
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
                        val = Util_R(prim,c,r,b,θ) + β*(res.val_next[n+1,api,si]) #compute value
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

    #worker
    for n = reverse(1:(Nr-1)) #backwards iteration
    for ai = 1:na #loop over a
        for si = 1:ns
        s = s_grid[si]
        a = a_grid[ai]
        candidate_max = -Inf
        #choice_lower = 1
        aep = s*η[n]
            for api = 1:na #loop over potential a'
                ap = a_grid[api]
                l::Float64 = lsupply(prim,ap,aep,a,w,r, θ)
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
                    val = util_w + β*sum(res.val_next[n+1,api,:].*markov[si,:]) #compute value
                    if val>candidate_max #check for new max value
                        candidate_max = val #update max value
                        res.pol_func[n, ai, si] = ap #update policy function
                        res.pol_func_ind[n, ai, si] = api #update policy function
                        res.labor[n,ai,si] = l #update labor
                        #res.budg[n,ai,si] = budget
                    end #if val
                elseif c <= 0
                    val = -Inf
                end #c if
            end #api
        res.val_func[n, ai, si] = candidate_max #update value function
    end #si
end #ai
end #n

res.val_next = res.val_func

end #func


function Bellman_T(K::Float64, L::Float64, θ::Float64, prim::Primitives, res::Results)
    @unpack a_grid, s_grid, β, α, δ, σ, γ, η, na, ns, N, Nr, markov = prim #unpack model primitives
    c_size_n = Cohort_size(prim)
    w = (1-α)*K^(α)*L^(-α)
    r = α*K^(α-1)*L^(1-α) - δ
    b = (θ*w*L)/sum(c_size_n[Nr:N])
    #Backwards iteration from period before retiring
    for n = reverse(Nr:N) #backwards iteration
        if n == N
            for ai = 1:na #loop over a states
            for si = 1:ns
            a = a_grid[ai]
            budget = (1+r)*a + b
            c = budget
            res.val_func[n,ai,si] = Util_R(prim,c,r,b,θ) #in pd N agent consumes all remaining a
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
                        val = Util_R(prim,c,r,b,θ) + β*(res.val_func[n+1,api,si]) #compute value
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

    #worker
    for n = reverse(1:(Nr-1)) #backwards iteration
    for ai = 1:na #loop over a
        for si = 1:ns
        s = s_grid[si]
        a = a_grid[ai]
        candidate_max = -Inf
        #choice_lower = 1
        aep = s*η[n]
            for api = 1:na #loop over potential a'
                ap = a_grid[api]
                l::Float64 = lsupply(prim,ap,aep,a,w,r, θ)
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
                        res.labor[n,ai,si] = l #update labor
                        #res.budg[n,ai,si] = budget
                    end #if val
                elseif c <= 0
                    val = -Inf
                end #c if
            end #api
        res.val_func[n, ai, si] = candidate_max #update value function
    end #si
end #ai
end #n
res.w = w
res.r = r
res.val_next = res.val_func

end #func

## T*

#find size of each cohort, growth rate means more young than old
#start w guess that initial size = 1, then decay to get every age
#add up and then divide all by sum so total size of mu = 1

function create_F(prim::Primitives,res::Results)
    @unpack na, ns, N, markov,pg = prim #unpack model primitives
    @unpack pol_func_ind = res #unpack model primitives
    c_size_n = Cohort_size(prim)
    F = (zeros(N,na,ns))
    F[1,1,1] = c_size_n[1]*0.2037
    F[1,1,2] = c_size_n[1]*0.7963
    #[age,asset,state]
    for n = 2:N
        for ai = 1:na
            for si = 1:ns #looping through coming from sh and sl
                aprevi = pol_func_ind[n-1,ai,si]
                for spi = 1:ns
                    F[n,aprevi,spi] += (F[n-1,ai,si] * markov[si,spi])/(pg+1)
                end
            end
        end
    end
    res.F = F
end



##
#Initialize backwards shooting
#one run Initialize
#one run Initialize4
#two run Init_SS

function Init_path(prim::Primitives, res::Results, res4::Results4)
    @unpack T = prim
    #starting guess of path for K and L, straight line
    res4.kap_agg[1] = 3.360
    res4.kap_agg[T] = 4.604
    res4.lab_agg[1] = 0.343
    res4.lab_agg[T] = 0.365

    #set for T
    K = res4.kap_agg[T]
    L = res4.lab_agg[T]
    θ = 0.0
    Bellman_T(K,L,θ, prim, res) #wo SS final stage

    res4.wage[T] = res.w
    res4.rate[T]  = res.r
    res4.pol_path[T,:,:,:] = res.pol_func
    res4.lab_path[T,:,:,:] = res.labor
    res.val_next = res.val_func
    res4.pol_ind_path[T,:,:,:] = res.pol_func_ind

    #set up aggregate path guesses
    slope_k = ((res4.kap_agg[T]) - res4.kap_agg[1])/(T)
    for i = 2:(T-1)
        res4.kap_agg[i] = slope_k*i + res4.kap_agg[1]
    end
    slope = (res4.lab_agg[T] - res4.lab_agg[1])/(T)
    for i = 2:(T-1)
        res4.lab_agg[i] = slope*i + res4.lab_agg[1]
    end
end


function Big_Init()
    println("Initialize")
    prim, res = Initialize()
    println("Initialize4")
    res4 = Initialize4(prim)
    println("Init_path")
    Init_path(prim, res, res4)
    prim, res, res4
end

#using JLD2
#save_object("res4.jld2", res4)
#test = load_object("res4.jld2")

## 10:40pm
#Shoot backwards
#using JLD2
#res4 = load_object("C:/Users/caitl/OneDrive/Documents/Second year/Computational 899/PS4/Caitlin/res4.jld2")
function Shoot_back(prim::Primitives, res::Results, res4::Results4)
    @unpack T = prim #unpack model primitives
    #θ::Float64
    time = 0
    for t = reverse(1:T-1)
        time += 1
        println("t = ", time)
        K = res4.kap_agg[t]
        L = res4.lab_agg[t]
    if t > 1
        θ = 0.0
        println("Bellman")
        Bellman(K, L, θ, prim, res)
        res4.wage[t] = res.w #save w, r, F, val
        res4.rate[t] = res.r
        res4.pol_path[t,:,:,:] = res.pol_func
        res4.lab_path[t,:,:,:] = res.labor
        res4.pol_ind_path[t,:,:,:] = res.pol_func_ind
    elseif t == 1
        θ = 0.11
        Bellman(K, L, θ, prim, res)
        #val_next = res.val_func
        res4.wage[t] = res.w #save w, r, F, val
        res4.rate[t] = res.r
        res4.pol_path[t,:,:,:] = res.pol_func
        res4.lab_path[t,:,:,:] = res.labor
        res4.pol_ind_path[t,:,:,:] = res.pol_func_ind
        res4.val_init = res.val_func
    end #end if
    end #t loop
    prim, res, res4
end

function Transition_dist(prim::Primitives, res::Results, res4::Results4)
    @unpack pol_ind_path = res4
    @unpack T, N, na, ns, markov, pg = prim
    #For t = 1 set mu[1] = F.
    c_size_n = Cohort_size(prim)
    mu = zeros(T,N,na,ns)
    for t = 1:T-1
        mu[t,1,1,1] = (c_size_n[1]*0.2037)#*(pg^(t-1))
        mu[t,1,1,2] = (c_size_n[1]*0.7963)#*(pg^(t-1))
    end

    for t = 1:T #we are connecting mu+1 and mu...
    for n = 2:N
        for ai = 1:na
        for si = 1:ns
        if t == 1
            aprevi = trunc(Int, pol_ind_path[1, n-1, ai, si])
        elseif t > 1
            aprevi = trunc(Int, pol_ind_path[t-1, n-1, ai, si])
        end
            for spi = 1:ns
                if t == 1
                    mu[t,n,aprevi,spi] += (mu[1,n-1,ai,si]*markov[si,spi])/(pg+1)
                elseif t > 1
                    mu[t,n,aprevi,spi] += (mu[t-1,n-1,ai,si]*markov[si,spi])/(pg+1)
                end
            end #spi
        end #si
        end #ai
    end #n
    end #t
    res4.mu = mu
end #func


function Shoot_forward(prim, res, res4)
    @unpack T, na, ns, N, a_grid, η, s_grid, Nr = prim

    K_path = zeros(T)
    L_path = zeros(T)

    K_path = res4.kap_agg #initial kap and lab paths
    L_path = res4.lab_agg

    error = 1.0
    tol = 1e-1

    iter = 0
    while error > tol
        iter += 1
        println("iter = ", iter)
        Shoot_back(prim, res, res4)
        Transition_dist(prim, res, res4)
    #    Transition_dist(prim, res, res4)
        K_new = zeros(T)
        L_new = zeros(T)

        for t = 1:T
        for n = 1:N, ai = 1:na, si = 1:ns
            K_new[t] += res4.mu[t,n,ai,si]*a_grid[ai]
        end
        end

        for t = 1:T
        for n = 1:Nr-1, ai = 1:na, si = 1:ns
            L_new[t] += res4.mu[t,n,ai,si]*η[n]*s_grid[si]*res4.lab_path[t,n,ai,si] #should i include prod here?
        end
        end

        err_k = zeros(T)
        err_l = zeros(T)

        for i = 1:T
            err_k[i] = abs(K_new[i] - K_path[i])
            err_l[i] = abs(L_new[i] - L_path[i])
        end

        ek = maximum(err_k)
        el = maximum(err_l)

        error = max(ek, el)

        println("error = ", error, "ek, el = ", ek, " ", el)

        K_path[1] = res4.kap_agg[1] #t = 1unchanged
        L_path[1] = res4.lab_agg[1]

        for t = 2:T
            K_path[t] = 0.6*res4.kap_agg[t] + 0.4*K_new[t]
            L_path[t] = 0.6*res4.kap_agg[t] + 0.4*L_new[t]
        end
        res4.kap_agg = K_path
        res4.lab_agg = K_path

    end #while
    diff = abs(K_path[T] - 4.604) + abs(L_path[T] - 0.365)
    println("diff = ", diff)
    prim, res, res4
end


function Solve_model()
    prim, res, res4 = Big_Init()
    prim, res, res4 = Shoot_back(prim, res, res4)
end



##############################################################################

#=



=#
