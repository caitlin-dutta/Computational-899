using StatFiles, CSV, Parameters, DataFrames, ForwardDiff, Optim, LatexPrint, BenchmarkTools, Plots, Random, Distributions, HaltonSequences, NLSolversBase


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
    d1_nodes_pre :: Array{Float64} = - log.(1 .- d1_quad[:,1])
    d2_nodes_pre_0 :: Array{Float64} = - log.(1 .- d2_quad[:,1])
    d2_nodes_pre_1 :: Array{Float64} = - log.(1 .- d2_quad[:,2])
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
    @unpack d1_quad, d1_jacob = data
    dist = Normal(0,1)
    n=size(d1_quad,1)
    val = 0.0
    #transform initial quadrature nodes to lie in (-inf, bound)
    nodes = - log.(1 .- d1_quad[:,1]) .+ bound
    #nodes = log.(d1_quad[:,1]) .+ bound
    #find jacobian of transformation 
    jacob = d1_jacob #1 ./(1 .- d1_quad[:,1])
    #jacob = 1 ./(d1_quad[:,1])
    weight = d1_quad[:,2]
    #for i=1:n
    #    val += weight[i]*cdf(dist, -inside - ρ*nodes[i])*(pdf(dist,nodes[i]/σ)/σ)*jacob[i]
    #end
    val = sum(weight .* cdf.(dist, inside .- ρ.*nodes) .* (pdf.(dist, nodes ./ σ) ./ σ) .* jacob)
    val
end

function quadrature_d2(data::Data, inside, bound_0, bound_1, ρ, σ)
    @unpack d2_quad, d2_nodes_pre_0, d2_nodes_pre_1, d2_jacob_0, d2_jacob_1 = data
    dist = Normal(0,1)
    n=size(d2_quad,1)
    val = 0.0
    #nodes_0 = - log.(1 .- d2_quad[:,1]) .+ bound_0
    #nodes_1 = - log.(1 .- d2_quad[:,2]) .+ bound_1
    #jacob_0 = 1 ./(1 .- d2_quad[:,1])
    #jacob_1 = 1 ./(1 .- d2_quad[:,2])
    nodes_0 = d2_nodes_pre_0 .+ bound_0
    nodes_1 = d2_nodes_pre_1 .+ bound_1
    jacob_0 = d2_jacob_0
    jacob_1 = d2_jacob_1
    weights = d2_quad[:,3]
    #for i=1:n
    #    val += weights[i]*cdf(dist, -inside - ρ*nodes_1[i])*(pdf(dist, nodes_1[i]-ρ*nodes_0[i])*pdf(dist, nodes_0[i]/σ)/σ)*jacob_0[i]*jacob_1[i]
    #end
    dens = pdf.(dist, nodes_1 .- ρ .* nodes_0) .* pdf.(dist, nodes_0 ./ σ) ./ σ
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
    ind_0 = α[1] .+ X'*β .+ Z[1]*γ
    ind_1 = α[2] .+ X'*β .+ Z[2]*γ
    ind_2 = α[3] .+ X'*β .+ Z[3]*γ
    val = 0.0
    if T ==1 
        val = cdf(dist, -ind_0/σ)
    elseif T == 2
        val = quadrature_d1(data, -ind_1, -ind_0, ρ, σ)
    elseif T == 3
        val = quadrature_d2(data, -ind_2, -ind_0, -ind_1, ρ, σ)
    else
        val = quadrature_d2_4(data, -ind_2, -ind_0,-ind_1, ρ, σ)
    end
    val
end

function halton_draws(n_draw, base)
    burn = n_draw*2
    halt_pre = Halton(base, length=n_draw + burn)
    halt = halt_pre[burn+1:n_draw + burn]
    #shocks = quantile.(Normal(0,1), halt)
    halt
end


function likelihood_GHK(X, Z, T, α, β, γ, ρ, shock_mat)
    σ = 1/(1-ρ)
    val=0.0
    dist=Normal(0,1)
    ind_0 = α[1] .+ X'*β .+ Z[1]*γ
    ind_1 = α[2] .+ X'*β .+ Z[2]*γ
    ind_2 = α[3] .+ X'*β .+ Z[3]*γ
    trunc_0 = -ind_0/σ
    sample_0 = quantile.(dist, cdf(dist, trunc_0) .+ shock_mat[:,1] .* (1 .- cdf(dist, trunc_0)))
    eps_0 = σ.* sample_0
    trunc_1 = -ind_1 .- ρ.*eps_0
    sample_1 = quantile.(dist, cdf.(dist,trunc_1) .+ shock_mat[:,2] .*(1 .- cdf.(dist, trunc_1)))
    eps_1 = ρ.*eps_0 + sample_1 
    trunc_2 = -ind_2 .- ρ.*eps_1
    cdf_0 = cdf(dist, trunc_0)
    cdf_1 = cdf(dist, trunc_1)
    cdf_2 = cdf(dist, trunc_2)
    if T==1
        val = cdf_0
    elseif T==2
        val = sum((1 .- cdf_0).*cdf_1)/size(shock_mat,1)
    elseif T==3
        val = sum((1 .-cdf_0).*(1 .-cdf_1).*cdf_2)/size(shock_mat,1)
    else #T ==4
        val = sum((1 .-cdf_0).*(1 .-cdf_1).*(1 .-cdf_2))/size(shock_mat,1)
    end
    val
end

function likelihood_AR(X, Z, T, α, β, γ, ρ, shock_mat)
    σ = 1/(1-ρ)
    val = 0.0
    dist = Normal(0,1)
    ind_0 = α[1] .+ X'*β .+ Z[1]*γ
    ind_1 = α[2] .+ X'*β .+ Z[2]*γ
    ind_2 = α[3] .+ X'*β .+ Z[3]*γ
    b_0 = -ind_0
    b_1 = - ind_1
    b_2 = - ind_2
    ε_0 = σ .* quantile.(dist, shock_mat[:,1])
    ε_1 = ρ .* ε_0 .+ quantile.(dist, shock_mat[:,2])
    ε_0_acc = (ε_0 .> b_0) 
    ε_1_acc = (ε_1 .> b_1)
    ε_01_acc = ε_0_acc .* ε_1_acc
    if T==1
        val = cdf(dist,b_0/σ)
    elseif T==2
        if length((ε_0_acc .== 1)) > 0
            val = sum(ε_0_acc .* cdf.(dist, b_1 .- ρ.*ε_0))/length((ε_0_acc .== 1))
        end
    elseif T==3
        if length((ε_01_acc .== 1)) > 0
            val = sum(ε_01_acc .* cdf.(dist, b_2 .- ρ.*ε_1))/length((ε_01_acc .== 1))
        end
    else #T ==4
        if length((ε_01_acc .== 1)) > 0
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
    shock_mat = [halton_draws(n_draw, 7) halton_draws(n_draw, 11)  halton_draws(n_draw, 13) ]
    for i=1:length(T)
        L += log.(likelihood_GHK(X[i,:], Z[i,:], T[i], α, β, γ, ρ, shock_mat))
        L_vec[i] = likelihood_GHK(X[i,:], Z[i,:], T[i], α, β, γ, ρ, shock_mat)
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
    shock_mat = [halton_draws(n_draw, 7) halton_draws(n_draw, 11)  halton_draws(n_draw, 13) ]
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

θ_init = vcat([0,-1,-1], zeros(15), 0.3, 0.5)
@elapsed ll, test = log_like_quad(Data(), θ_init)
opt = optimize(θ -> -log_like_quad(Data(), θ), θ_init, LBFGS(), Optim.Options(show_trace = true, show_every = 1,
iterations=100, g_tol=1e-3); autodiff=:forward)
θ_1 = opt.minimizer
opt_1 = optimize(θ -> -log_like_quad(Data(), θ), θ_1, LBFGS(), Optim.Options(show_trace = true, show_every = 1,
iterations=100, g_tol=1e-3); autodiff=:forward)
θ_opt = opt_1.minimizer
T1 = (Data().T .== 1)
T2 = (Data().T .== 2)
T3 = (Data().T .== 3)
T4 = (Data().T .== 4)

like_types = ["Quadrature", "GHK", "Accept_Reject"]
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
likelihood_comparison(Data(), like_types, θ_init, 100)

@belapsed ll, l_vec = log_like_quad_verbose(Data(), θ_init)
##quadrature took 1.288 seconds
@belapsed ll, l_vec = log_like_GHK(Data(), θ_init,100)
#GHK took 1.644 seconds
@belapsed ll, l_vec = log_like_AR(Data(), θ_init,100)
#AR took 0.482

#histogram([test_0[T1], test_0[T2], test_0[T3], test_0[T4]])
#histogram([test_0_g[T1], test_0_g[T2], test_0_g[T3], test_0_g[T4]])
#histogram([test_0_AR[T1], test_0_AR[T2], test_0_AR[T3], test_0_AR[T4]])

plot([test_0[T1], test_0[T2], test_0[T3], test_0[T4]], [test_0_AR[T1], test_0_AR[T2], test_0_AR[T3], test_0_AR[T4]], seriestype=:scatter)
plot([test_0[T1], test_0[T2], test_0[T3], test_0[T4]], [test_0_AR[T1], test_0_g[T2], test_0_g[T3], test_0_g[T4]], seriestype=:scatter)
#plot([test_0[T1] - test_0_g[T1], test_0[T2] - test_0_g[T2], test_0[T3] - test_0_g[T3], test_0[T4] - test_0_g[T4]], seriestype=:scatter)
#plot([test_0[T1] - test_0_AR[T1], test_0[T2] - test_0_AR[T2], test_0[T3] - test_0_AR[T3], test_0[T4] - test_0_AR[T4]], seriestype=:scatter)
lap(round.(θ_opt, digits=3))

ll_opt, ll_v_opt = ll, test = log_like_quad(Data(), θ_opt)
histogram([ll_v_opt[T1], ll_v_opt[T2], ll_v_opt[T3], ll_v_opt[T4]])
histogram([ll_v_opt[T4], test_0[T4]])
histogram([test_0_g[T1], test_0_g[T2], test_0_g[T3], test_0_g[T4]])
histogram([test_0_AR[T1], test_0_AR[T2], test_0_AR[T3], test_0_AR[T4]])

plot(test_0, [ll_v_opt, test_0], seriestype=:scatter)

plot([test_0[T1], test_0[T2], test_0[T3], test_0[T4]], [test_0_AR[T1], test_0_AR[T2], test_0_AR[T3], test_0_AR[T4]], seriestype=:scatter)
plot([test_0[T1], test_0[T2], test_0[T3], test_0[T4]], [test_0_AR[T1], test_0_g[T2], test_0_g[T3], test_0_g[T4]], seriestype=:scatter)