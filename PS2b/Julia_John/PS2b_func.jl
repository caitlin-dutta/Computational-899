


@with_kw struct Data 
    df :: DataFrame = DataFrame(load("Mortgage_performance_data.dta"))
    #select the relevant columns from the data and convert into array 
    X ::Array{Float64} = identity.(Array(select(df, [:score_0, :rate_spread, :i_large_loan, :i_medium_loan, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5])))
    #add a column of ones to handle the intercept term
    #X ::Array{Float64}= hcat(ones(size(X_pre,1)), X_pre)
    #construct time-varying characteristics and convert to array
    Z :: Array{Float64} = identity.(Array(select(df,[:score_0, :score_1, :score_2])))
    #extract the column of Y variables, convert into array
    Y ::Array{Float64} = identity.(Array(select(df,[:i_close_0, :i_close_1, :i_close_2])))
    #construct the variable T based on Y
    T :: Array{Float64} = T_maker(Y) 
    #load in our quadrature weights for 1 and 2 dimensional integration
    d1_quad :: Array{Float64} = identity.(Array(DataFrame(CSV.File("KPU_d1_l20.asc", header=false))))
    d2_quad :: Array{Float64} = identity.(Array(DataFrame(CSV.File("KPU_d2_l20.asc", header=false))))
    #precompute first part of quadrature nodes 
    d1_nodes_pre :: Array{Float64} = - log.(1 .- d1_quad[:,1])
    d2_nodes_pre_0 :: Array{Float64} = - log.(1 .- d2_quad[:,1])
    d2_nodes_pre_1 :: Array{Float64} = - log.(1 .- d2_quad[:,2])
    #precompute jacobians for quadrature
    d1_jacob :: Array{Float64} = 1 ./(1 .- d1_quad[:,1])
    d2_jacob_0 :: Array{Float64} = 1 ./ (1 .- d2_quad[:,1])
    d2_jacob_1 :: Array{Float64} = 1 ./ (1 .- d2_quad[:,2])
end

#constructs the variable T based on Y
#I know if statements are passe - but I don't care!
function T_maker(Y)
    n = size(Y,1)
    T = zeros(n)
    for i=1:size(Y,1)
        if Y[i, 1] == 1
            T[i] =1
        elseif Y[i,1]==0 && Y[i,2]==1
            T[i] = 2
        elseif Y[i,1]==0 && Y[i,2]==0 && Y[i,3]==1
            T[i]=3
        else 
            T[i]=4
        end
    end
    T
end

function quadrature_d1(data::Data, inside, bound, ρ, σ)
    @unpack d1_quad, d1_jacob, d1_nodes_pre = data
    dist = Normal(0,1)
    n=size(d1_quad,1)
    val = 0.0
    #transform initial quadrature nodes to lie in (-inf, bound) by adding the integration bound to the precomputed transformed quadrature nodes
    nodes = d1_nodes_pre .+ bound
    #find jacobian of transformation 
    jacob = d1_jacob 
    #find the weight assigned to this node
    weight = d1_quad[:,2]
    #find the sum of the likelihood for each quadrature draw
    val = sum(weight .* cdf.(dist, inside .- ρ.*nodes) .* (pdf.(dist, nodes ./ σ) ./ σ) .* jacob)
    val
end

function quadrature_d2(data::Data, inside, bound_0, bound_1, ρ, σ)
    @unpack d2_quad, d2_nodes_pre_0, d2_nodes_pre_1, d2_jacob_0, d2_jacob_1 = data
    dist = Normal(0,1)
    n=size(d2_quad,1)
    val = 0.0
    #add the integral bounds to the precomputed transformed quadrature nodes
    nodes_0 = d2_nodes_pre_0 .+ bound_0
    nodes_1 = d2_nodes_pre_1 .+ bound_1
    #unpack the stored jacobians of the transformation
    jacob_0 = d2_jacob_0
    jacob_1 = d2_jacob_1
    #find the weights for the integration
    weights = d2_quad[:,3]
    #compute the density part of the likelihood
    dens = pdf.(dist, nodes_1 .- ρ .* nodes_0) .* pdf.(dist, nodes_0 ./ σ) ./ σ
    #find the sum of the likelihood for each quadrature draw
    val = sum( weights .* cdf.(dist, inside .- ρ .* nodes_1) .* dens .* jacob_0 .* jacob_1)
    val
end

function quadrature_d2_4(data::Data, inside, bound_0, bound_1, ρ, σ)
    @unpack d2_quad, d2_nodes_pre_0, d2_nodes_pre_1, d2_jacob_0, d2_jacob_1 = data
    dist = Normal(0,1)
    n=size(d2_quad,1)
    val = 0.0
    #nodes_0 = log.( d2_quad[:,1]) .+ bound_0
    #nodes_1 = log.( d2_quad[:,2]) .+ bound_1
    #jacob_0 = 1 ./(d2_quad[:,1])
    #jacob_1 = 1 ./(d2_quad[:,2])
    nodes_0 = d2_nodes_pre_0 .+ bound_0
    nodes_1 = d2_nodes_pre_1 .+ bound_1
    jacob_0 = d2_jacob_0
    jacob_1 = d2_jacob_1
    weights = d2_quad[:,3]
    dens = pdf.(dist, nodes_1 .- ρ .* nodes_0) .* pdf.(dist, nodes_0 ./ σ) ./ σ
    val = sum( weights .* (1 .- cdf.(dist, inside .- ρ .* nodes_1)) .* dens .* jacob_0 .* jacob_1)
    val
end

function likelihood_quad(data::Data, X, Z, T, α, β, γ, ρ)
    dist = Normal(0,1)
    σ = 1/(1-ρ) #since sigma^2 = 1/(1-rho)^2, the standard deviation is 1/(1-rho)
    #find the individual-specific truncation point
    ind_0 = -α[1] .- X'*β .- Z[1]*γ
    ind_1 = -α[2] .- X'*β .- Z[2]*γ
    ind_2 = -α[3] .- X'*β .- Z[3]*γ
    val = 0.0
    if T ==1 #likelihood they pay at t=0
        val = cdf(dist, ind_0/σ)
    elseif T == 2 #likelihood they don't pay at t=0 but do pay at t=1
        val = quadrature_d1(data, ind_1, ind_0, ρ, σ)
    elseif T == 3 #likelihood they don't pay in t=0 or 1 but do pay in t=2
        val = quadrature_d2(data, ind_2, ind_0, ind_1, ρ, σ)
    else #likelihood they don't pay in t=0,1, or 2
        val = quadrature_d2_4(data, ind_2, ind_0,ind_1, ρ, σ)
    end
    val
end

#generates halton sequences of length n_draw using specified base
function halton_draws(n_draw, base)
    #use a burn-in period for the sequence - here, it'll be 200 with 100 draws. I don't know what is optimal for different values of draws, but I don't think it'll matter too much. The resulting sequences look very uniform (and normal rv's generated using them look pretty normal)
    burn = n_draw*2
    #find the Halton sequence with the burn in plus the length of the draw requested
    halt_pre = Halton(base, length=n_draw + burn)
    #only use the past n_draw elements
    halt = halt_pre[burn+1:n_draw + burn]
    halt
end


function likelihood_GHK(X, Z, T, α, β, γ, ρ, shock_mat)
    #define sigma using passed rho
    σ = 1/(1-ρ)
    #initial value of zero
    val=0.0
    #initialize distribution
    dist=Normal(0,1)
    #individual-specific component of truncation points
    b_0 = -α[1] .- X'*β .- Z[1]*γ
    b_1 = -α[2] .- X'*β .- Z[2]*γ
    b_2 = -α[3] .- X'*β .- Z[3]*γ
    #first truncation point for eps_{i0}
    trunc_0 = b_0/σ
    #create a sample from truncated N(0,1) on support (trunc_0, ∞)
    η_0 = quantile.(dist, cdf(dist, trunc_0) .+ shock_mat[:,1] .* (1 .- cdf(dist, trunc_0)))
    #rescale by sigma to get the draw of eps_{i0}
    ε_0 = σ.* η_0
    #find the truncation point for ε_{i1} using the drawn value of ε_{i0} 
    trunc_1 = b_1 .- ρ.*ε_0
    #construct the random sample of η_{i1} from a truncated normal(0,1) with support (trunc_1, ∞) 
    η_1 = quantile.(dist, cdf.(dist,trunc_1) .+ shock_mat[:,2] .*(1 .- cdf.(dist, trunc_1)))
    #use the definition of ε_{i1} to construct it based on ε_{i0} and the drawn η_{i1} 
    ε_1 = ρ.*ε_0 + η_1
    #find the truncation point for eps_{i2} 
    trunc_2 = b_2 .- ρ.*ε_1
    #find the cdf evaluated at each truncation point
    cdf_0 = cdf(dist, trunc_0)
    cdf_1 = cdf(dist, trunc_1)
    cdf_2 = cdf(dist, trunc_2)
    if T==1  #likelihood they pay at t=0
        val = cdf_0
    elseif T==2 #likelihood they don't pay at t=0 but do pay at t=1
        val = sum((1 .- cdf_0).*cdf_1)/size(shock_mat,1)
    elseif T==3 #likelihood they don't pay at t=0 or 1 but do pay at 2
        val = sum((1 .-cdf_0).*(1 .-cdf_1).*cdf_2)/size(shock_mat,1)
    else #likelihood they never pay
        val = sum((1 .-cdf_0).*(1 .-cdf_1).*(1 .-cdf_2))/size(shock_mat,1)
    end
    val
end

function likelihood_AR(X, Z, T, α, β, γ, ρ, shock_mat)
    #define sigma using passed rho
    σ = 1/(1-ρ)
    val = 0.0
    dist = Normal(0,1)
    #individual specific component of truncation points
    b_0 = -α[1] .- X'*β .- Z[1]*γ
    b_1 = -α[2] .- X'*β .- Z[2]*γ
    b_2 = -α[3] .- X'*β .- Z[3]*γ
    #randomly draw N(0, sigma^2) random variables using first Halton sequence
    ε_0 = σ .* quantile.(dist, shock_mat[:,1])
    #randomly generate ε_1 as the sum of rho times ε_0 plus an N(0,1) random variable
    ε_1 = ρ .* ε_0 .+ quantile.(dist, shock_mat[:,2])
    #indicator for whether ε_0 > b_0 (i.e. agent doesn't pay in t=0)
    ε_0_acc = (ε_0 .> b_0) 
    #indicator for whether ε_1 > b_1 (i.e. agent doesn't pay in t=1)
    ε_1_acc = (ε_1 .> b_1)
    #joint indicator for whether the agent doesn't pay back in either t=0 or 11
    ε_01_acc = ε_0_acc .* ε_1_acc
    if T==1
        #no need to accept/reject here, just use the cdf
        val = cdf(dist,b_0/σ)
    elseif T==2
        #check if nonzero length of sample
        if length((ε_0_acc .== 1)) > 0
            #only count observations which have indicators equal to 1
            #for each accepted sample, add the inside of the integral (except for the density), divide by number of accepted samples
            val = sum(ε_0_acc .* cdf.(dist, b_1 .- ρ.*ε_0))/length((ε_0_acc .== 1))
        end
    elseif T==3
        if length((ε_01_acc .== 1)) > 0
            #only count observations which have indicators equal to 1
            #for each accepted sample, add the inside of the integral (except for the density), divide by number of accepted samples
            val = sum(ε_01_acc .* cdf.(dist, b_2 .- ρ.*ε_1))/length((ε_01_acc .== 1))
        end
    else #T ==4
        if length((ε_01_acc .== 1)) > 0
            #only count observations which have both indicators equal to 1
            #for each accepted sample, add the inside of the integral (except for the density), divide by number of accepted samples
            val = sum( ε_01_acc.* (1 .- cdf.(dist, b_2 .- ρ .* ε_1)) )/length((ε_01_acc .== 1))
        end
    end
    val
end

function log_like_quad(data::Data, θ)
    @unpack X, Z, T, d1_quad, d2_quad = data
    L = 0.0
    α = θ[1:3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]
    #L_vec = zeros(length(T))
    for i=1:length(T)
        L_i = likelihood_quad(data, X[i,:], Z[i,:], T[i], α, β, γ, ρ)
        L += log(L_i)
        #L_vec[i] = L_i
    end
    L#, L_vec
end
#function which returns both LL and the vector of likelihoods
function log_like_quad_verbose(data::Data, θ)
    @unpack X, Z, T, d1_quad, d2_quad = data
    L = 0.0
    α = θ[1:3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]
    L_vec = zeros(length(T))
    for i=1:length(T)
        L_i = likelihood_quad(data, X[i,:], Z[i,:], T[i], α, β, γ, ρ)
        L += log(L_i)
        L_vec[i] = L_i
    end
    L, L_vec
end


function log_like_GHK(data::Data, θ, n_draw)
    @unpack X, Z, T = data
    L = 0.0
    α = θ[1:3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]
    L_vec = zeros(length(T))
    #generate matrix of halton sequence draws
    shock_mat = [halton_draws(n_draw, 7) halton_draws(n_draw, 11)]
    for i=1:length(T)
        L_i = likelihood_GHK(X[i,:], Z[i,:], T[i], α, β, γ, ρ, shock_mat)
        L += log(L_i)
        L_vec[i] = L_i
    end
    L, L_vec
end

function log_like_AR(data::Data, θ, n_draw)
    @unpack X, Z, T = data
    L = 0.0
    α = θ[1:3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]
    L_vec = zeros(length(T))
    shock_mat = [halton_draws(n_draw, 7) halton_draws(n_draw, 11)]
    for i=1:length(T)
        L_i = likelihood_AR(X[i,:], Z[i,:], T[i], α, β, γ, ρ, shock_mat)
        L_vec[i] = L_i
        if L_i > 0
            L += log.(L_i)
        else
            L += -1e12
        end
    end
    L, L_vec
end


#function which computes log likelihoods for each type, creates plots, and saves them
function likelihood_comparison(data::Data, l_types, θ, ndraws)
    for i=1:length(l_types)
        type = l_types[i]
        ll_quad, lv_quad = log_like_quad_verbose(data, θ)
        if i == 1
            ll, l_vec = log_like_quad_verbose(data, θ)
        elseif i ==2
            ll, l_vec = log_like_GHK(data, θ, ndraws)
        else
            ll, l_vec = log_like_AR(data, θ, ndraws)
        end
        if i > 1
            comp_plot = plot([lv_quad[T1], lv_quad[T2], lv_quad[T3], lv_quad[T4]], [l_vec[T1], l_vec[T2], l_vec[T3], l_vec[T4]], xlabel="Individual likelihood, quadrature", ylabel="Individual likelihood, $(type)", legend=:topleft, title="Simulated likelihoods: Quadrature vs $(type)", labels=["T = 1" "T = 2" "T = 3" "T = 4"], seriestype=:scatter)
            plot!([0.0, 1.0], [0.0, 1.0], l=2, labels="45-degree line")
            savefig(comp_plot, "$(type)_comp.png")
        end
        println("Simulated likelihood using $(type): $(ll)")
        ll_plot = histogram([l_vec[T1], l_vec[T2], l_vec[T3], l_vec[T4]],labels=["T = 1" "T = 2" "T = 3" "T = 4"], title="Histogram of likelihoods by outcome, $(type)", legend=:topleft)
        savefig(ll_plot, "$(type)_ll.png")
    end
end
