
@with_kw struct Primitives
    T::Int64 = 200 #length of time series
    H::Int64 = 10 #number of simulated series
    ρ :: Float64 = 0.5 #true rho for data generating process
    σ :: Float64 = 1.0 #true variance of data generating process
    seed :: Int64  #seed for random number generator, to be set when struct is initialized
    x_seq::Vector{Float64} = process_generator(T, ρ, σ, seed) #generate 'true' x sequence
    e_mat::Array{Float64} = e_generator(T, H, seed) #generate matrix of normal shocks where each column is one draw of normal random shocks of length T
    M_T ::Vector{Float64} = compute_M_T(x_seq) #find the true moment of the data generating process
end 

#function to initialize the primitives struct which accepts a random number seed as input
function Initialize(seed_no)
    prim_h = Primitives(seed = seed_no)
    prim_h
end

#function to generate the true process
function process_generator(T, rho, sigma,seed)
    Random.seed!(seed); #set random seed
    val = zeros(T) #initialize empty array
    dist = Normal(0, sigma) #specify normal distribution with mean zero and variance sigma
    shocks = rand(dist, T) #draw vector of shocks of length T from distribution
    val[1] = shocks[1] #initial value
    for i=2:T #iteratively construct sequence
        val[i] = rho*val[i-1] + shocks[i] #each value is rho times the previous value plus the current shock
    end
    val
end

#generate matrix of normal shocks to be used to construct simulated data; each of the H columns is a length T vector of random shocks
function e_generator(T,H, seed)
    Random.seed!(seed);
    e_mat = randn(T,H)
    e_mat
end

#given arbitrary parameters rho and sigma, generate simulated data
function sim_process(prim, rho, sigma)
    @unpack e_mat,T, H= prim
    sim_array = zeros(T, H)
    if sigma < 0 #don't know if this ends up being necessary, but I just created a very bad sequence if a negative sigma is passed so I don't take the square root of a negative number. Better safe than sorry
        sim_array = fill(500.0, T, H)
    else
        for i=2:T
            sim_array[i,:] .= rho .*sim_array[i-1,:] + sqrt(sigma)*e_mat[i,:] #generate process in the same way as above, but now we have N(0,1) shocks so we must scale by the square root of the desired variance 
        end
    end
    sim_array
end

#smol function to compute the first-order autocovariance of a time series
function autocov_1(seq)
    val = 0
    zbar = mean(seq) #find the mean
    for i=2:length(seq) 
        val += (seq[i] - zbar)*(seq[i-1] - zbar)/(length(seq)) #iteratively add (z_i - zbar)(z_{i-1} - z_bar) and divide by the length of the sequence to get the acf evaluated at 1
    end
    val
end

#this function finds the moment vector for the true data 
function compute_M_T(x_seq)
    val = zeros(3)
    val[1] = mean(x_seq) #first element will be eual to the mean of the sequence
    val[2] = var(x_seq) #second element of the moment vector will be the variance
    val[3] = autocov_1(x_seq) #third element of the moment vector will be the first order autocovariance
    val
end

#this function generates H simulated sequences of length T according to the supplied rho and sigma and finds the mean of their moment vectors
function compute_M_TH(prim, rho, sigma, moment_spec)
    @unpack H, e_mat = prim
    val = zeros(3)
    y_seq = sim_process(prim, rho, sigma) #given parameters rho and sigma, generate H sequences of length T
    for h=1:H #iterate over each generated sequence to get the mean of each sequence's moment vector
        val[1] += mean(y_seq[:,h])/H #find the average of the means of each simulated sequence
        val[2] += var(y_seq[:,h])/H #find the average of the variances
        val[3] += autocov_1(y_seq[:,h])/H #find the average of the acf's
    end
    if moment_spec == 1 #if we want to only use the mean and variance moments
        val_return = [val[1], val[2]] #select only the mean and variance components of the moment vector
    elseif moment_spec == 2 #if we want only the variance and acf
        val_return = [val[2], val[3]] #select only the variance and acf components
    else #if we want the full moment vector
        val_return = val
    end 
    val_return #return the selected vector
end

#this function finds the objective function for supplied coefficient vector b_vec, weight matrix W, and moment selection indicated by moment_spec
function compute_J_TH(prim, b_vec, W, moment_spec)
    @unpack M_T = prim
    rho = b_vec[1] #extract the rho coefficient
    sigma = b_vec[2] #extract the sigma^2 coefficient
    M_TH = compute_M_TH(prim, rho, sigma, moment_spec) #find the simulated moment given the supplied coefficients
    if moment_spec == 1 #if we want to only use the mean and variance moments of the true moment
        M_T_rest = [M_T[1], M_T[2]]
    elseif moment_spec ==2 #if we want to only use the variance and autocovariance
        M_T_rest = [M_T[2], M_T[3]]
    else #if we want to use the full moment
        M_T_rest = M_T
    end
    g_TH = M_T_rest - M_TH #find g_TH, which is the difference between the data moment and the simulated moment
    obj = g_TH'*W*g_TH #calculate objective function
    obj
end

#function to plot the objective function for range of parameter values
function plot_JTH(prim, x_low, x_high, y_low, y_high, W, moment_spec)
    n_grid = 50
    x_range = collect(range(start=x_low, stop=x_high, length=n_grid))
    y_range = collect(range(start=y_low, stop=y_high, length=n_grid))
    val = zeros(n_grid,n_grid)
    for i=1:n_grid, j=1:n_grid
        val[i,j] = compute_J_TH(prim, [x_range[i], y_range[j]], W, moment_spec)
    end
    jth_plot = plot(x_range,y_range,val,xlabel="rho", ylabel="sigma", zlabel="Objective function",st=:surface, c=:deep,camera=(30,15))
    jth_plot
end

#this function computes Gamma(j) for all lags from 0 up to the maximum lag specified. It takes a sequence y_h and the moment function
function autocov_func(y_h, M_TH, max_lag, moment_spec)
    if moment_spec == 1 || moment_spec ==2 #if we only want to use two of the moments, Gamma will be a 2x2 matrix
        Γ = zeros(1+max_lag,2,2)
    else #if we use all three moments, Gamma will be a 3x3 matrix
        Γ = zeros(1+max_lag,3,3)
    end
    y_mean = mean(y_h) #find the mean of the sequence
    for i=1:max_lag+1 #iterate over all lags from 0 to max_lag (we'll have to subtract 1 to get the lag)
        for j=i+2:length(y_h)-1 #iterate over observations (being careful with indices)
            if moment_spec == 1 #for the case where we use only mean and variance
                my = [y_h[j], (y_h[j] - y_mean)^2] #find m_2(y_j)
                my_lag = [y_h[j-i+1], (y_h[j-i+1] - y_mean)^2] #find m_2(y_{j-(i-1)} = m_2(y_{j-i +1})). because lag index starts at 1, we need to subtract 1 from i
                Γ[i,:,:] += (my - M_TH)*(my_lag - M_TH)'./(length(y_h)) #construct the average value by incrementing by the current observation divided by the length
            elseif moment_spec ==2 #for the case where we use only variance and first order autocovariance
                my = [(y_h[j] - y_mean)^2, (y_h[j]-y_mean)*(y_h[j-1] - y_mean)] #find m_2(y_j)
                my_lag = [(y_h[j-i+1] - y_mean)^2, (y_h[j-i+1]-y_mean)*(y_h[j-i] - y_mean)]#find m_2(y_{j-(i-1)} = m_2(y_{j-i +1})). because lag index starts at 1, we need to subtract 1 from i
                Γ[i,:,:] += (my - M_TH)*(my_lag - M_TH)'./(length(y_h))#construct the average value by incrementing by the current observation divided by the length
            else #for the case where we use mean, variance, and first order autocovariance
                my = [y_h[j], (y_h[j] - y_mean)^2, (y_h[j]-y_mean)*(y_h[j-1] - y_mean)] #find m_3(y_j)
                my_lag = [y_h[j-i+1], (y_h[j-i+1] - y_mean)^2, (y_h[j-i+1]-y_mean)*(y_h[j-i] - y_mean)]#find m_3(y_{j-(i-1)} = m_3(y_{j-i +1})). because lag index starts at 1, we need to subtract 1 from i
                Γ[i,:,:] += (my - M_TH)*(my_lag - M_TH)'./(length(y_h))#construct the average value by incrementing by the current observation divided by the length
            end
        end
    end
    Γ
end

function autocov_TH(prim, rho, sigma, max_lag, moment_spec)
    @unpack e_mat, H = prim
    y_seq = sim_process(prim, rho, sigma) ##given parameters rho and sigma, generate H sequences of length T
    if moment_spec == 1 || moment_spec ==2 #if we want either only the mean or variance or only the variance or acf, it will be a 2x2 matrix
        Γ_TH = zeros(max_lag+1, 2,2)
    else #otherwise, if we use the full moment, we will need a 3x3 matrix
        Γ_TH = zeros(max_lag + 1, 3,3)
    end
    M_TH = compute_M_TH(prim, rho, sigma, moment_spec) #find the simulated moment vectors
    for h=1:H #find average of the acf's across each simulated dataset
        Γ_TH += autocov_func(y_seq[:,h], M_TH, max_lag, moment_spec) ./H
    end
    Γ_TH
end

#all props to Ken West and Whitney Newey for this one - this finds the Newey-West estimator for the asymptotic variance covariance matrix of the simulated data
function thx_Ken_West(prim, b, max_lag, moment_spec)
    @unpack H = prim
    rho = b[1]
    sigma = b[2]
    Γ_TH = autocov_TH(prim, rho, sigma, max_lag, moment_spec) #find acf at for each lag value
    S = Γ_TH[1,:,:] #initially set S equal to Γ(0) - the acf evaluated at lag 0 (corresponding to index 1)
    for i=2:max_lag+1 #iterate over higher lags
        S += (1 - i/(max_lag + 1)) .*(Γ_TH[i,:,:] + Γ_TH[i,:,:]') 
    end
    S = (1 + 1/H) .* S #rescale
    S
end

function delta_g(prim, b_2, W_hat, moment_spec)
    @unpack T = prim
    rho = b_2[1]
    sigma = b_2[2]
    #compute numerical derivatives of the moment function with respect to rho and sigma
    dMdrho = -(compute_M_TH(prim, rho, sigma, moment_spec) - compute_M_TH(prim, rho - 1e-12, sigma, moment_spec))./(1e-12)
    dMdsigma = -(compute_M_TH(prim, rho, sigma, moment_spec) - compute_M_TH(prim, rho, sigma - 1e-12, moment_spec ))./(1e-12)
    #horizontally combine the above vectors to get the ∇_g matrix
    ∇_g = hcat(dMdrho, dMdsigma)
    #find the variance-covariance matrix for the coefficients
    vcov = inv(∇_g'*W_hat*∇_g)./T
    #find the standard errors for the coefficients
    se = sqrt.(diag(vcov))
    ∇_g, vcov, se
end

function J_test(prim, b_2, W_hat, moment_spec)
    @unpack T,H = prim
    #find objective function given coefficient vector b_2 and weight matrix W_hat
    J_TH = compute_J_TH(prim, b_2, W_hat, moment_spec)
    #compute test statistic
    J = T *(H/(1+H))*J_TH
    J
end

function problem_solver(prim,moment_spec, plot_ind)
    #start with initial guess of identity matrix for W
    if moment_spec ==1 || moment_spec ==2 #if we only want two moments
        W_0 = diagm(ones(2))
    else
        W_0 = diagm(ones(3)) #if we want all three moments
    end
    if plot_ind ==1 #if we want to plot
        jth_plot = plot_JTH(prim, 0.35, 0.65, 0.8, 1.2, W_0,moment_spec)
        savefig(jth_plot, "jthplot_$(moment_spec)_init.png")
    end
    #find b_1 by minimizing the objective function evaluated at identity matrix 
    opt_b = optimize(b -> compute_J_TH(prim, b, W_0, moment_spec), [0.5, 1.0])
    b_1 = opt_b.minimizer #find the minimizer
    S_TH = thx_Ken_West(prim, b_1,4 ,moment_spec) #find asymptotic v-cov matrix
    W_TH = inv(S_TH) #invert v-cov matrix to find the optimal weight matrix for next stage
    if plot_ind ==1 #if we want to plot
        jth_plot_2 = plot_JTH(prim, 0.35, 0.65, 0.8, 1.2, W_TH,moment_spec)
        savefig(jth_plot_2, "jthplot_$(moment_spec)_2nd.png")
    end
    #given new weight matrix, find the new optimal coefficient vector
    opt_b2 = optimize(b -> compute_J_TH(prim, b, W_TH,moment_spec), [0.5, 1.0])
    b_2 = opt_b2.minimizer
    #find ∇_g, the variance covariance, and standard errors for the coefficient vector
    ∇_g_2, vcov_2, se_2 =  delta_g(prim, b_2, W_TH, moment_spec)
    #compute the J test statistic
    J = J_test(prim, b_2, W_TH, moment_spec)
    b_1, b_2, W_TH, ∇_g_2, vcov_2, se_2, J
end


#find results for each moment vector specification. 
function res_finder(seed_no, verbose, plot_ind)
    prim = Initialize(seed_no) #initialize struct with specified seed number
    #create empty arrays to store results
    b1_res = zeros(3,2)
    b2_res = zeros(3,2)
    W_res = [zeros(2,2), zeros(2,2), zeros(3,3)]
    dg_res = [zeros(2,2),zeros(2,2), zeros(3,2)]
    vcov_res = [zeros(2,2),zeros(2,2), zeros(3,3)]
    se2_res = [[],[], []]
    J_res = zeros(3)
    #loop over each moment vector specification
    for i=1:3
        b1_res[i,:], b2_res[i,:], W_res[i], dg_res[i], vcov_res[i], se2_res[i], J_res[i] = problem_solver(prim,i, plot_ind)
    end
    if verbose == 1 #if we ask for more outputs, return more outputs
        return b1_res, b2_res, W_res, dg_res, vcov_res, se2_res, J_res
    else #otherwise, keep the chatter to a minimum
        return b1_res, b2_res
    end
end

#this function repeats the above analysis B times and stores the coefficients 
function bootstrapper(B)
    ρ1_sample = zeros(B)
    σ1_sample = zeros(B)
    ρ2_sample = zeros(B)
    σ2_sample = zeros(B)
    for b=1:B
        b1, b2 = problem_solver(Primitives(seed=b), 3, 0)
        ρ1_sample[b] = b1[1]
        σ1_sample[b] = b1[2]
        ρ2_sample[b] = b2[1]
        σ2_sample[b]= b2[2]
    end
    ρ1_sample, σ1_sample, ρ2_sample, σ2_sample
end
