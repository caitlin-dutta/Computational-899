#### the algorithm
# initial guess of K, L
# compute b
# compute prices w, r
# given w, r solve the household problem
# compute the cross-sectional distribution of households
# commpute aggregate capital and labor supply
# iterate on K, L until convergence

@with_kw struct Primitives
    # primatives of the model
    n::Float64 = .011
    β::Float64 = .97
    γ::Float64 = .42
    σ::Float64 = 2
    θ::Float64 = .11
    α::Float64 = .36
    δ::Float64 = .06

    N::Int64 = 66
    Jr::Int64 = 46
    age_grid::Array{Int64,1} = collect(1:1:N)
    n_age::Int64 = length(age_grid)

    na::Int64 = 1000
    a_min::Float64 = .0001
    a_max::Float64 = 100.0
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max))

    z_markov::Array{Float64,2} = [.9261 .0739 ; .0189 .9811]
    z_grid::Array{Float64, 1} = [3.0 ,.5]
    nz::Int64 = length(z_grid)

    η::Vector{Float64} = [0.59923239,0.63885106 ,0.67846973 ,0.71808840 ,0.75699959 ,0.79591079 ,0.83482198 ,0.87373318 ,0.91264437 ,0.95155556 ,0.99046676 ,0.99872065
    ,1.0069745 ,1.0152284 ,1.0234823 ,1.0317362 ,1.0399901 ,1.0482440 ,1.0564979 ,1.0647518 , 1.0730057 ,1.0787834 ,1.0845611 ,1.0903388 , 1.0961165 , 1.1018943
    ,1.1076720 , 1.1134497 ,1.1192274 , 1.1250052 ,1.1307829 ,1.1233544 ,1.1159259 ,1.1084974 ,1.1010689 ,1.0936404 ,1.0862119 ,1.0787834 ,1.0713549 ,1.0639264
    ,1.0519200,1.0430000,1.0363000,1.0200000,1.0110000]
end


mutable struct Results
    #store the results we want to return
    val_func::Array{Float64, 3}
    pol_func::Array{Float64, 3}
    pol_ind::Array{Int64, 3}
    l_opt::Array{Float64, 3}
    F::Array{Float64,3}
    mu::Array{Float64,1}
    w::Float64
    r::Float64
    b::Float64
end

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.na,prim.nz,prim.n_age) #initial value function guess
    pol_func = zeros(prim.na,prim.nz,prim.n_age) #initial policy function guess
    pol_ind = zeros(prim.na,prim.nz,prim.n_age)
    l_opt = zeros(prim.na,prim.nz,prim.n_age)
    F = zeros(prim.na,prim.nz,prim.n_age)
    mu = zeros(prim.N)
    mu[1] = 1
    for j=2:prim.N
        mu[j] = mu[j-1]/(1+prim.n)
    end
    mu = mu/sum(mu)
    w = 0.0
    r = 0.0
    b = 0.0

    res = Results(val_func, pol_func, pol_ind, l_opt, F, mu, w, r, b) #initialize results struct
    prim, res #return deliverables
end

function U_R(c, γ, σ)
    #utility of the old
    U = c^((1-σ)*γ)/(1-σ)
    U
end
#
function U_W(c,l,σ, γ)
    #utility of the young
    U = ((c^γ)*(1-l)^(1-γ))^(1-σ)/(1-σ)
    U
end

function BackwardIteration(K, L, prim::Primitives, res::Results)
    # calculate the value and policy functions of the young and old
    @unpack β, N, n, Jr, θ, z_grid, nz, α, δ, z_markov, age_grid, na, a_grid, η, γ,σ  = prim

    #calculate mu
    mu = res.mu


    # calculate prices
    w = (1-α)*L^(-α)*K^(α)
    r = α*L^(1-α)*K^(α-1) - δ
    b = (θ*res.w*L)/sum(mu[Jr:N])

    # loop over the old
    for j=reverse(Jr:N)
        if j==N
            for a_index = 1:na
                c = (1+r)*a_grid[a_index] + b
                res.pol_func[a_index,:,j] .= 0
                res.val_func[a_index,:,j] .= U_R(c,γ,σ)
            end
        elseif j!=n
            for a_index = 1:na
                budget = (1+r)*a_grid[a_index] + b
                candidate_max = -Inf

                for ap_index = 1:na
                    c = budget - a_grid[ap_index]

                    if c>0
                        val = U_R(c,γ,σ) + β*res.val_func[ap_index,1,j+1]
                        if val> candidate_max
                            candidate_max = val
                            res.pol_func[a_index,:,j] .= a_grid[ap_index]
                            res.pol_ind[a_index,:,j] .= ap_index
                        end
                    elseif c<=0
                        val = -Inf
                    end
                end
                res.val_func[a_index,:,j] .= candidate_max
            end
        end
    end

    # loop over the young
    for j = reverse(1:Jr-1)
        for a_index = 1:na, z_index = 1:nz
            candidate_max = -Inf

            for ap_index = 1:na
                #calculate labor supply
                l = (γ*(1-θ)*(z_grid[z_index])*η[j]*w - (1-γ)*((1+r)*a_grid[a_index]-a_grid[ap_index]))/((1-θ)*z_grid[z_index]*w*η[j])
                if l > 1
                    l=1
                end
                if l < 0
                    l=0
                end

                #calculate consumption
                c = w*(1-θ)*(z_grid[z_index])*η[j]*l + (1+r)*a_grid[a_index]-a_grid[ap_index]

                #calculate val
                if c>0
                    val = U_W(c,l,σ,γ) + β*sum(res.val_func[ap_index,:,j+1].*z_markov[z_index,:])
                    if val > candidate_max
                        candidate_max = val
                        res.pol_func[a_index, z_index, j] = a_grid[ap_index]
                        res.pol_ind[a_index, z_index, j] = ap_index
                        res.l_opt[a_index, z_index, j] = l
                    end
                elseif c <= 0
                    val = -Inf
                end
            end
            res.val_func[a_index, z_index, j] = candidate_max
        end
    end
    res.b = b
    res.w = w
    res.r = r
end


function distribution(prim::Primitives, res::Results)
    # calculate the distribution
    @unpack N, n, na, nz, Jr, z_markov = prim
    @unpack pol_ind = res

    mu = res.mu

    F = zeros(na,nz,N)
    F[1,1,1] = mu[1]*0.2037
    F[1,2,1] = mu[1]*0.7963

    for j = 2:N
        for a_index=1:na, z_index = 1:nz
            prev = pol_ind[a_index,z_index,j-1]

            for zp_index = 1:nz
                    F[prev,zp_index,j] += F[a_index,z_index,j-1]*z_markov[z_index,zp_index]/(1+n)
            end

        end
    end

    res.F = F
end

function SolveModel()
    # guess K, L and iterate until convergence
    prim, res = Initialize()
    @unpack η, nz, na, N, Jr,z_grid,a_grid = prim

    K = 3.4
    L = 0.366
    error = 1
    tol = .001

    # BackwardIteration(K,L,prim,res)
    # distribution(prim, res)

    while error>tol
        BackwardIteration(K,L,prim,res)
        distribution(prim, res)
        L_new = 0
        K_new = 0

        for j=1:N, m=1:na, z=1:nz
            K_new += res.F[m,z,j]*a_grid[m]
        end
        for j=1:Jr-1, m=1:na, z=1:nz
            L_new += res.F[m,z,j]*η[j]*z_grid[z]*res.l_opt[m,z,j]
        end
        error = max(abs(K-K_new), abs(L-L_new))
        K = .6*K + .4*K_new
        L = .6*L + .4*L_new
        println(K," ", L)
    end

    prim, res
end
