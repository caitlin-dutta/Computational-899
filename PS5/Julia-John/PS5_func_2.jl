# --------------------------------------------------------------------------
# File Name: ks_model.jl
# Author: Philip Coyle
# --------------------------------------------------------------------------
## Housekeeping
@with_kw struct Params
    cBET::Float64 = 0.99
    cALPHA::Float64 = 0.36
    cDEL::Float64 = 0.025
    cLAM::Float64 = 0.5

    N::Int64 = 5000
    T::Int64 = 11000
    burn::Int64 = 1000

    tol_vfi::Float64 = 1e-4
    tol_coef::Float64 = 1e-4
    tol_r2::Float64 = 1.0 - 1e-2
    maxit::Int64 = 10000
end

@with_kw struct Grids
    k_lb::Float64 = 0.001
    k_ub::Float64 = 20.0
    n_k::Int64 = 40
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0
    K_ub::Float64 = 15.0
    n_K::Int64 = 30
    K_grid::Array{Float64,1} = range(K_lb, stop = K_ub, length = n_K)

    eps_l::Float64 = 0.0
    eps_h::Float64 = 0.3271
    n_eps::Int64 = 2
    eps_grid::Array{Float64,1} = [eps_h, eps_l]

    z_l::Float64 = 0.99
    z_h::Float64 = 1.01
    n_z::Int64 = 2
    z_grid::Array{Float64,1} = [z_h, z_l]
end

@with_kw  struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0 # Duration (Good Times)
    u_b::Float64 = 0.1 # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0 # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pgb::Float64 = 1.0 - (d_b-1.0)/d_b
    pbg::Float64 = 1.0 - (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb10::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    markov::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                                pbg*Mbg pbb*Mbb]
end

mutable struct Results
    pf_k::Array{Float64,4}
    pf_v::Array{Float64,4}

    a0::Float64
    a1::Float64
    b0::Float64
    b1::Float64

    R2::Float64

    Z::Vector{Float64}
    E::Array{Float64,2}
end

function Initialize()
    P = Params()
    G = Grids()
    S = Shocks()
    P,G,S
end

function Init_R()
    pf_k = zeros(G.n_k, G.n_eps, G.n_K, G.n_z )
    pf_v = zeros(G.n_k, G.n_eps, G.n_K, G.n_z )
    a0 = 0.095
    b0 = 0.085
    a1 = 0.999
    b1 = 0.999
    R2 = 0.0
    E,Z = draw_shocks(S, P.N, P.T) 
    R = Results(pf_k, pf_v, a0, a1, b0, b1, R2, Z, E)
    R
end

function draw_shocks(S::Shocks, N::Int64,T::Int64)
    @unpack pgg, pbb, Mgg, Mgb, Mbg, Mbb = S

    # Shock
    Random.seed!(12032020)
    dist = Uniform(0, 1)

    # Allocate space for shocks and initialize
    idio_state = zeros(N,T)
    agg_state = zeros(T)
    idio_state[:,1] .= 1
    agg_state[1] = 1

    for t = 2:T
        agg_shock = rand(dist)
        if agg_state[t-1] == 1 && agg_shock < pgg
            agg_state[t] = 1
        elseif agg_state[t-1] == 1 && agg_shock > pgg
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock < pbb
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock > pbb
            agg_state[t] = 1
        end

        for n = 1:N
            idio_shock = rand(dist)
            if agg_state[t-1] == 1 && agg_state[t] == 1
                p11 = Mgg[1,1]
                p00 = Mgg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 1 && agg_state[t] == 2
                p11 = Mgb[1,1]
                p00 = Mgb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 1
                p11 = Mbg[1,1]
                p00 = Mbg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 2
                p11 = Mbb[1,1]
                p00 = Mbb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            end
        end
    end

    return idio_state, agg_state
end

#function which takes aggregate capital, labor, and productivity factor and finds interest rate and wage
function prices(P::Params, K::Float64, L::Float64, z::Float64)
    @unpack cALPHA = P
    α = cALPHA
    r = α*z*(K/L)^(α-1)
    w = (1-α)*z*(K/L)^(α)
    r, w
end

function Bellman(P::Params, G::Grids, S::Shocks, R::Results)
    @unpack cBET, cALPHA, cDEL = P
    @unpack n_k, k_grid, n_eps, eps_grid, eps_h, K_grid, n_K, n_z, z_grid = G
    @unpack u_g, u_b, markov = S
    @unpack pf_k, pf_v, a0, a1, b0, b1= R

    pf_k_up = zeros(n_k, n_eps, n_K, n_z)
    pf_v_up = zeros(n_k, n_eps, n_K, n_z)

    # In Julia, this is how we define an interpolated function.
    # Need to use the package "Interpolations".
    # (If you so desire, you can write your own interpolation function too!)
    k_interp = interpolate(k_grid, BSpline(Linear()))
    v_interp = interpolate(pf_v, BSpline(Linear()))

    for (i_z, z_today) in enumerate(z_grid)
        for (i_K, K_today) in enumerate(K_grid)
            if i_z == 1
                K_tomorrow = exp(R.a0 + R.a1*log(K_today)) #predicted level of capital tomorrow given we're in good state :)
                L_today = (1-u_g)*eps_h
            elseif i_z == 2
                K_tomorrow = exp(R.b0 + R.b1*log(K_today)) #predicted level of capital tomorrow given we're in bad state :(
                L_today = (1-u_b)*eps_h
            end
            r_today, w_today = prices(P, K_today, L_today, z_today)
            # See that K_tomorrow likely does not fall on our K_grid...this is why we need to interpolate!
            i_Kp = get_index(K_tomorrow, K_grid)

            for (i_eps, eps_today) in enumerate(eps_grid)
                row = i_eps + n_eps*(i_z-1)

                for (i_k, k_today) in enumerate(k_grid)
                    budget_today = r_today*k_today + w_today*eps_today + (1.0 - cDEL)*k_today

                    # We are defining the continuation value. Notice that we are interpolating over k and K.
                    v_tomorrow(i_kp) = markov[row,1]*v_interp(i_kp,1,i_Kp,1) + markov[row,2]*v_interp(i_kp,2,i_Kp,1) +
                                        markov[row,3]*v_interp(i_kp,1,i_Kp,2) + markov[row,4]*v_interp(i_kp,2,i_Kp,2)


                    # We are now going to solve the HH's problem (solve for k).
                    # We are defining a function val_func as a function of the agent's capital choice.
                    val_func(i_kp) = log(budget_today - k_interp(i_kp)) +  cBET*v_tomorrow(i_kp)

                    # Need to make our "maximization" problem a "minimization" problem.
                    obj(i_kp) = -val_func(i_kp)
                    lower = 1.0
                    upper = get_index(budget_today, k_grid)

                    # Then, we are going to maximize the value function using an optimization routine.
                    # Note: Need to call in optimize to use this package.
                    opt = optimize(obj, lower, upper)

                    k_tomorrow = k_interp(opt.minimizer[1])
                    v_today = -opt.minimum

                    # Update PFs
                    R.pf_k[i_k, i_eps, i_K, i_z] = k_tomorrow
                    pf_v_up[i_k, i_eps, i_K, i_z] = v_today
                end
            end
        end
    end

    return pf_v_up
end

function get_index(val::Float64, grid::Array{Float64,1})
    n = length(grid)
    index = 0
    if val <= grid[1]
        index = 1
    elseif val >= grid[n]
        index = n
    else
        index_upper = findfirst(x->x>val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    return index
end

function iterate(P::Params, G::Grids, S::Shocks, R::Results)
    @unpack tol_vfi = P
    @unpack n_k, n_K = G
    n = 0
    error = 100
    R.pf_v = zeros(n_k, 2, n_K, 2) #empty value function guess
    println("Finding VF...")
    while error > tol_vfi
        n +=1
        v_next = Bellman(P, G, S, R) #iterate bellman equation
        error = maximum(abs.(R.pf_v - v_next))/maximum(abs.(v_next)) #find max error
        #println("n = $(n), error = $(error)")
        R.pf_v = v_next #update policy function
    end
    println("Found VF!")
end

#this function creates a pseudopanel given the random shocks and policy functions
function create_panel(P::Params, G::Grids, S::Shocks, R::Results)
    @unpack N, T = P
    @unpack k_grid, K_grid, n_k, n_K, k_ub, k_lb, K_ub, K_lb = G
    @unpack pf_v, pf_k, Z, E = R
    #I created a separate interpolation for each (e,z) combo - there may be a better, more concise way, but I got this to work and don't really want to mess around with it
    pf_int_11 = interpolate((k_grid, K_grid), pf_k[:, 1, :, 1], Gridded(Linear()))
    pf_int_12 = interpolate((k_grid, K_grid), pf_k[:, 1, :, 2], Gridded(Linear()))
    pf_int_21 = interpolate((k_grid, K_grid), pf_k[:, 2, :, 1], Gridded(Linear()))
    pf_int_22 = interpolate((k_grid, K_grid),  pf_k[:, 2, :, 2], Gridded(Linear()))
    pf_int = [[pf_int_11, pf_int_21] [pf_int_12,  pf_int_22]] #create a lil array for each interpolated policy function
    K = zeros(T) #empty aggregate capital array
    V = zeros(N, T) #empty capital choice grid
    K[1] = 11.55 #initial aggregate capital guess from steady state
    V[:, 1] .= 11.55 #initial individual capital guess 
    for t = 2:T #iterate over all time periods and individuals
        for n =1:N
            k_prev = max(k_lb, min(k_ub, V[n, t-1])) #find capital holdings from previous period, make sure it's on grid
            K_prev = max(K_lb, min(K_ub, K[t-1]))#find aggregate capital holdings from previous period, make sure it's on grid
            V[n,t] = pf_int[Int(E[n,t-1]), Int(Z[t-1])](k_prev, K_prev) #evaluate policy function given previous values to get next period's capital
        end
        K[t] = sum(V[:,t])/N #find aggregate capital at time t
    end
    K, V
end

#this function finds OLS estimates for both states
function coeff_estimates(P::Params, G::Grids, S::Shocks, R::Results, K_full::Vector{Float64})
    @unpack T = P
    Z = R.Z[1000:T] #throw out first 1000 observations
    K = K_full[1000:T]
    dat = zeros(T-1001, 4) #empty array to store data
    for t in 1:length(dat[:, 1])
        dat[t,1] = log(K[t+1])
        dat[t, 2] = 1.0 #want regression with intercept, so need column of ones
        dat[t, 3] = log(K[t])
        dat[t, 4] = Z[t] #include aggregate state so we can create conditional sample
    end
    dat_g = dat[dat[:,4] .== 1.0, :] #find subset of data which corresponds to good state
    dat_b = dat[dat[:,4] .== 2.0, :] #find subsed of data corresponding to bad state
    #Y and X for good states
    Y_g = dat_g[:,1]
    X_g = dat_g[:,2:3]
    #Y and X for bad states
    Y_b = dat_b[:,1]
    X_b = dat_b[:,2:3]
    #find OLS coefficients
    β_g = inv(X_g'X_g)*X_g'Y_g
    β_b = inv(X_b'X_b)*X_b'Y_b
    #find prediction for each state
    pred_yg = X_g*β_g
    pred_yb = X_b*β_b
    #find residual sum of squares by state
    err_g = sum((Y_g - pred_yg).^2)
    err_b = sum((Y_b - pred_yb).^2)
    #find total sum of squares
    ssy_g = sum((Y_g .- mean(Y_g)).^2)
    ssy_b = sum((Y_b .- mean(Y_b)).^2)
    #find overall R^2
    R2 = 1- (err_g + err_b)/(ssy_g + ssy_b)
    #stack coefficients
    coeff_vec = vcat(β_g, β_b)
    coeff_vec, R2, X_b, Y_b, X_g, Y_g, pred_yg, pred_yb
end

function coeff_finder(P::Params, G::Grids, S::Shocks, R::Results)
    @unpack tol_coef, tol_r2 = P
    iterate(P, G, S, R) #do value function iteration
    K, V = create_panel(P, G, S, R) #create pseudopanel
    c_new, R2, xb, yb, xg, yg = coeff_estimates(P, G, S, R, K) #find coefficient estimates for each state
    print(c_new) #see what they are!
    error = abs(c_new[1] - R.a0)+ abs(c_new[2] - R.a1)+ abs(c_new[3] - R.b0)+ abs(c_new[4] - R.b1) #compute total error
    λ = 0.6 #weight parameter
    #find convex combination of old coefficients and new ones
    R.a0 = λ*R.a0 + (1-λ)*c_new[1]
    R.a1 = λ*R.a1 + (1-λ)*c_new[2]
    R.b0 = λ*R.b0 + (1-λ)*c_new[3]
    R.b1 = λ*R.b1 + (1-λ)*c_new[4]
    R.R2 = R2
    n = 0
    while error > tol_coef || R.R2 < tol_r2 #iterate until both convergence criteria are met
        n += 1
        println("Iteration $(n), error = $(error)")
        iterate(P, G, S, R) #do value function iteration
        K, V = create_panel(P, G, S, R) #generate corresponding pseudopanel
        c_new, R2 = coeff_estimates(P, G, S, R, K) #find new coeffs
        println("$(R.a0) $(R.a1) $(R.b0) $(R.b1) $(R.R2)") #check out their progress :)
        error = abs(c_new[1] - R.a0)+ abs(c_new[2] - R.a1)+ abs(c_new[3] - R.b0)+ abs(c_new[4] - R.b1) #find max error
        #find convex combination to get new updated coefficient
        R.a0 = λ*R.a0 + (1-λ)*c_new[1]
        R.a1 = λ*R.a1 + (1-λ)*c_new[2]
        R.b0 = λ*R.b0 + (1-λ)*c_new[3]
        R.b1 = λ*R.b1 + (1-λ)*c_new[4]
        R.R2 = R2
    end
    println("Convergence! a0 = $(R.a0), a1 = $(R.a1), b0 = $(R.b0), b1 = $(R.b1), R2 = $(R.R2)")
end
    
    

