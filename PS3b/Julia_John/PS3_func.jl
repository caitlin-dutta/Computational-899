
@with_kw struct Data 
    char_df :: DataFrame = DataFrame(load("Car_demand_characteristics_spec1.dta"))
    IV_df :: DataFrame = DataFrame(load("Car_demand_iv_spec1.dta")) 
    type_df :: DataFrame = DataFrame(load("Simulated_type_distribution.dta")) 
    X :: Array{Float64} = identity.(Array(char_df[!, Not([:Model_id, :Year, :Version_id, :share, :delta_iia])]))
    shares :: Array{Float64} = identity.(Array(char_df[!, "share"]))
    price :: Array{Float64} = identity.(Array(char_df[!, "price"]))
    #select the relevant columns from the data and convert into array 
    char ::Array{Float64} = Array(char_df) 
    δ_iia ::Vector{Float64} = identity.(Array(char_df[!, "delta_iia"]))
    Z ::Array{Float64} = hcat(X[:,2:size(X,2)], Array(IV_df[!, [:i_import, :diffiv_local_0, :diffiv_local_1, :diffiv_local_2, :diffiv_local_3, :diffiv_ed_0 ]]))
    income :: Array{Float64} = Array(type_df)
    min_year :: Int64 = Int64(minimum(char_df[!, "Year"]))
    max_year :: Int64 = Int64(maximum(char_df[!, "Year"]))
    year_range = collect(min_year:1:max_year)
    mkt_index :: BitMatrix = mkt_indices(char_df, year_range, max_year, min_year)
end

#finds the indices of all products sold in each year, stores as matrix
function mkt_indices(char_df, year_range, max_year, min_year)
    year_indices = zeros(size(char_df,1), max_year- min_year + 1)
    for (index, year) in enumerate(year_range)
        year_indices[:,index] .= (char_df[!, "Year"] .== year)
    end
    val = (year_indices .== 1)
    val
end


function μ_comp(income, price, λ_p)
    μ = zeros(length(price), length(income))
    μ = λ_p * price * income'
    #for i=1:length(income)
    ##    μ[:,i] .= λ_p .* income[i] .* price
    #end
    μ
end

function δ_comp(X, price, α, β)
    δ = X * β .+ α .* price
    δ
end 

function ind_share_pred(μ, δ)
    delta_mu = δ .+  μ 
    share_num = exp.(delta_mu)
    share_denom = 1 .+ sum(share_num, dims=1)
    share = share_num ./ share_denom
    share
end

function share_pred(income, μ, δ)
    ind_share = ind_share_pred(μ, δ)
    R = length(income)
    share = sum(ind_share, dims=2) ./ R
    share
end


function jacobian(σ)
    J_t = size(σ, 1)
    R = size(σ, 2)
    Δ = (1/R).* diagm(ones(J_t)) .* (σ*(1 .- σ)') - (1/R) .* (1 .- diagm(ones(J_t))) .* (σ*σ') 
    Δ
end

function BLP_contraction_mkt(income, price, shares, δ_t, λ_p, ε_newt; verbose=nothing)
    ε = 1e-12
    μ = μ_comp(income, price, λ_p)
    δ_0 = δ_t
    σ_ind = ind_share_pred(μ, δ_0)
    σ_pred = share_pred(income, μ, δ_0)
    δ_1 = δ_0 + log.(shares) - log.(σ_pred)
    error = maximum(abs.(δ_1 - δ_0))
    δ_0 = copy(δ_1)
    n =1
    error_vec = zeros(0)
    while error > ε
        #if ε_newt = 0, we will always do the contraction. If ε_newt > 0, we will switch to do Newton when the error is sufficiently low
        if error > ε_newt
            σ_pred = share_pred(income, μ, δ_0)
            δ_1 = δ_0 + log.(shares) - log.(σ_pred)
            error = maximum(abs.(δ_1 - δ_0))
            append!(error_vec, error)
            δ_0 = copy(δ_1)
        else
            σ_ind = ind_share_pred(μ, δ_0)
            σ_pred = share_pred(income, μ, δ_0)
            Δ = jacobian(σ_ind)
            δ_1 = δ_0 +(inv(Δ ./ σ_pred))*(log.(shares) - log.(σ_pred))
            error = maximum(abs.(δ_1 - δ_0))
            append!(error_vec, error)
            δ_0 = copy(δ_1)
        end
    end
    #println("Convergence!")
    if verbose==1
        return δ_1, error_vec
    else
        return δ_1
    end
end 

function BLP_contraction(data::Data, λ_p, ε_newt)
    @unpack char, X, shares, δ_iia, income, year_range, mkt_index= data
    δ = zeros(size(char,1))
    for (index, year) in enumerate(year_range) 
        mkt_indicator = mkt_index[:, index]
        X_t = X[mkt_indicator,:]
        price_t = X_t[:,1]
        δ_t = δ_iia[mkt_indicator]
        shares_t = shares[mkt_indicator]
        δ[mkt_indicator] = BLP_contraction_mkt(income, price_t, shares_t, δ_t, λ_p, ε_newt, verbose=0)
    end
    δ
end

function IV(X, Z, W, δ)
    β =inv((X' * Z) * W * (Z' * X)) * (X' * Z) * W * Z' * δ
    β
end

function compute_ρ(data, W, λ_p)
    @unpack X, Z = data
    δ = BLP_contraction(data, λ_p,1)
    β_IV = IV(X, Z, W ,δ)
    ρ = δ - X*β_IV
    ρ
end

function GMM_objective(data, λ_p_vec; method=nothing, λ_hat=nothing)
    @unpack Z = data
    λ_p = λ_p_vec[1] 
    if method == "two-step"
        W_1 = inv(Z' * Z)
        ρ_1 = compute_ρ(data, W_1, λ_hat)
        W_2 = inv((Z .* ρ_1)' * (Z .* ρ_1))
        ρ_2 = compute_ρ(data, W_2, λ_p)
        val = ρ_2' * Z * W_2 * Z' * ρ_2
        return val
    else
        W = inv(Z' * Z)
        ρ = compute_ρ(data, W, λ_p)
        val = ρ' * Z * W * Z' * ρ
        return val
    end
end

function λ_grid_search(data; method_ind=nothing, λhat = nothing)
    @unpack X, Z = data
    λ_grid = collect(0.0:0.01:1.0)
    val_array = zeros(length(λ_grid),2)
    for (i,λ_p) in enumerate(λ_grid)
        #ρ = compute_ρ(data, W, λ_p)
        println(i/length(λ_grid))
        val_array[i,:] .= [GMM_objective(data, λ_p, method=method_ind, λ_hat = λhat ),  λ_p]
    end
    val_array
end

