#Comp 899 PS2 Caitlin Dutta
#keyword-enabled structure to hold model primitives
using Parameters, Plots, SharedArrays
@with_kw struct Primitives
    β::Float64 = 0.97 #discount rate
    γ::Float64 = 0.42 #coefficient of rr a
    δ::Float64 = 0.06 #depreciation
    α::Float64 = 0.36 #capital share
    θ::Float64 = 0.0 #SS tax on labor
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

    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) #capital grid
    markov::Array{Float64,2} = [0.9261 0.0739 ; 0.0189 0.9811] #transition matrix
    s_grid::Array{Float64,1} = [3, 0.5] #high/low productivity shocks
    ns::Int64 = length(s_grid)
    pg::Float64 = 0.011
end


#structure that holds model results
 mutable struct Results
    val_func::SharedArray{Float64,3} #value function
    pol_func::SharedArray{Float64,3} #policy function
    pol_func_ind::SharedArray{Int64,3} #policy function
    labor::SharedArray{Float64, 3} #labor choices
    budg::SharedArray{Float64, 3} #budg choices
    F::SharedArray{Float64, 3} #F dist
    c_size_n::SharedArray{Float64, 1}
    w::Float64  #wage
    r::Float64  #rental rate
    b::Float64  #ss benefit
end

#F' = F0 + F*s'


#function for initializing model primitives and results
function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    pol_func_ind = SharedArray{Int64}(ones(prim.N, prim.na, prim.ns))
    labor = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    budg = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #initial policy function guess
    F = SharedArray{Float64}(zeros(prim.N, prim.na, prim.ns)) #F dist
    c_size_n = zeros(prim.N)
    w = 0.0
    r = 0.0
    b = 0.0
    res = Results(val_func, pol_func, pol_func_ind, labor, budg, F, c_size_n, w, r, b) #initialize results struct
    prim, res #return deliverables
end

function Cohort_size(prim::Primitives, res::Results)
    c_size = zeros(66)
    c_size[1] = 1
    for i = 2:66
        c_size[i] = c_size[i-1]/(1+0.011)
    end
    res.c_size_n = c_size./sum(c_size)
end



function Util_R(prim::Primitives, c::Float64, r::Float64, b::Float64)
    @unpack σ, γ = prim #unpack model primitives
    util_R = c^((1-σ)*γ)/(1-σ)
    util_R
end

function Util_W(prim::Primitives, c::Float64, l::Float64)
    @unpack σ, γ = prim #unpack model primitives
    util_w = ((c^γ)*((1-l)^(1-γ)))^(1-σ)/(1-σ)
    util_w
end

function lsupply(prim::Primitives, ap::Float64, aep::Float64, a::Float64, w::Float64, r::Float64)
    @unpack σ, γ, η, θ = prim #unpack model primitives
    lsupply = ((γ*(1-θ)*aep*w) - ((1-γ)*((1+r)*a - ap)))  / ((1-θ)*w*aep)
    lsupply
end

#Bellman Operator
function Bellman(K::Float64, L::Float64, prim::Primitives, res::Results)
    #@unpack val_func, labor, budg = res #unpack value function
    @unpack a_grid, s_grid, β, α, δ, σ, γ, η, θ, na, ns, N, Nr, markov = prim #unpack model primitives
    c_size_n = res.c_size_n

    w = (1-α)*K^(α)*L^(-α)
    r = α*K^(α-1)*L^(1-α) - δ
    b = (θ*res.w*L)/sum(c_size_n[Nr:N]) #want to use existing w

    #Backwards iteration from period before retiring
    for n = reverse(Nr:N) #backwards iteration
        if n == N
            for ai = 1:na #loop over a states
            for si = 1:ns
            a = a_grid[ai]
            budget = (1+r)*a + b
            c = budget
            res.val_func[n,ai,si] = Util_R(prim,c,r,b) #in pd N agent consumes all remaining a
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
                        val = Util_R(prim,c,r,b) + β*(res.val_func[n+1,api,si]) #compute value
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
        candidate_maxw = -Inf
        #choice_lower = 1
        aep = s*η[n]
            for api = 1:na #loop over potential a'
                ap = a_grid[api]
                l::Float64 = lsupply(prim,ap,aep,a,w,r)
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
                    if val>candidate_maxw #check for new max value
                        candidate_maxw = val #update max value
                        res.pol_func[n, ai, si] = ap #update policy function
                        res.pol_func_ind[n, ai, si] = api #update policy function
                        res.labor[n,ai,si] = l #update labor
                        res.budg[n,ai,si] = budget
                    end #if val
                elseif c <= 0
                    val = -Inf
                end #c if
            end #api
        res.val_func[n, ai, si] = candidate_maxw #update value function
    end #si
end #ai
end #n
    res.b = b
    res.w = w
    res.r = r

end #func


## T*

#find size of each cohort, growth rate means more young than old
#start w guess that initial size = 1, then decay to get every age
#add up and then divide all by sum so total size of mu = 1

function create_F(prim::Primitives,res::Results)
    @unpack na, ns, N, markov,pg = prim #unpack model primitives
    @unpack pol_func_ind = res #unpack model primitives
    c_size_n = res.c_size_n
    F = (zeros(N,na,ns))
    println(sum(c_size_n))
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
#Convergence on L and K

function Solve_model()
    prim, res = Initialize()
    Cohort_size(prim, res)
    @unpack η, ns, na, N, Nr, s_grid, a_grid = prim
    K = 3.4 #initial guess K
    L = 0.366 #initial guess L

    error = 1
    tol = 0.01

    n = 0

    while error > tol
        n += 1
        Bellman(K, L, prim, res)
        create_F(prim, res)

        K_new = 0
        L_new = 0

        for n = 1:N, ai = 1:na, si = 1:ns
            K_new += res.F[n,ai,si]*a_grid[ai]
        end

        for n = 1:Nr-1, ai = 1:na, si = 1:ns
            L_new += res.F[n,ai,si]*s_grid[si]*η[n]*res.labor[n,ai,si]
        end

        error = max(abs(K-K_new), abs(L-L_new))
        K = 0.6*K + 0.4*K_new
        L = 0.6*L + 0.4*L_new

        println("Iteration,", n, " Error = ", error, " K, L = ", K, " ", L)

    end

    prim, res
end



##############################################################################
