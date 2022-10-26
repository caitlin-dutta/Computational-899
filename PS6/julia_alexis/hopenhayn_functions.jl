@with_kw struct Primitives
    β::Float64 = 0.8
    θ::Float64 = 0.64
    s::Vector{Float64} = [0.000398, 3.58, 6.82, 12.18, 18.79] # incumbent productivty shocks
    ns::Int64 = length(s)
    markov::Array{Float64,2} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                                0.1997 0.7201 0.0420 0.0326 0.0056;
                                0.2000 0.2000 0.5555 0.0344 0.0101;
                                0.2000 0.2000 0.2502 0.3397 0.0101;
                                0.2000 0.2000 0.2500 0.3400 0.0100]
    ν::Vector{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
    c_f::Float64 = 10
    c_e::Float64 = 5
    A::Float64 = 0.005
end

mutable struct Results
    policy::Array{Int64,1}
    W::Array{Float64,1}
    p::Float64
    labor_choice::Array{Float64,1}
    μ::Array{Float64,1}
    M::Float64
end

function Initialize()
    prim = Primitives()

    policy = zeros(prim.ns)
    W = zeros(prim.ns)
    labor_choice = zeros(prim.ns)
    p=0.0
    μ=zeros(prim.ns)
    M=0.0

    res = Results(policy, W, p, labor_choice, μ,M)
    prim, res
end

function LaborDemand(p, s, θ)
    n = (θ*p*s)^(1/(1-θ)) #from the first order condition
    n
end

function AggLaborDemand(prim, res, M)
    @unpack ν = prim
    @unpack labor_choice,μ = res
    Ld=0.0
    Ld = sum(labor_choice.*μ) + M*sum(labor_choice.*ν)
    Ld
end

function AggLaborSupply(Π, prim)
    @unpack A = prim
    Ns = 1/A - Π
    Ns
end

function AggProfits(prim, res, M)
    @unpack s,θ,c_f,ν, c_e= prim
    @unpack p,labor_choice,μ = res
    π_small = p*s .* labor_choice.^θ - labor_choice .- p*c_f
    Π = sum(π_small.* (μ + M*ν)) - M*p*c_e #?????
end

function FirmValue(p, prim::Primitives, res::Results)
    @unpack W = res
    @unpack θ, c_f, β, s, markov, ns = prim

    v_next = zeros(ns)
    n = zeros(ns)
    π = zeros(ns)

    #Dynamic programming problem
    for si=1:length(s)
        s_val= s[si]
        n[si] = LaborDemand(p, s_val, θ)
        π[si] = p*s[si]*n[si]^θ - n[si] - p*c_f
        res.labor_choice[si] = n[si]

        #policy is whether to exit or not
        val_stay = π[si] + β * sum(W .* markov[si,:])
        val_exit = π[si]
            if val_stay>=val_exit
                v_next[si] = val_stay
                res.policy[si] = 0
            elseif val_stay<val_exit
                v_next[si] = val_exit
                res.policy[si] = 1
            end
        end
    v_next
end
        #compute optimal choices


function V_iterate(p, prim::Primitives, res::Results; tol::Float64 = 1e-2, err::Float64 = 100.0)
    counter = 0
    res.W = zeros(prim.ns)

    while err > tol
        v_next = FirmValue(p,prim,res)
        err = maximum(abs.(v_next - res.W))
        res.W = v_next
        counter += 1
    end

end

#p0 =.738
function Prices(prim::Primitives, res::Results; tol = 1e-6)
    @unpack ν, c_e = prim
    # loop to find p*
    p = 0.738
    p_high = 1
    p_low = 0
    n = 0
    EC = 100 #initial value for loop

    while abs(EC) > tol && n < 1000
        n+=1
        # calculate firms value function
        V_iterate(p, prim, res)

        EC = sum(res.W .* ν)/p - c_e

        if EC>0 #if EC > 0, lower p_i
            p_high = p
        elseif EC<0 #if EC < 0, raise p_i
            p_low = p
        end

        p = (p_low+p_high)/2
    end
    res.p = p
end

function t_star(prim::Primitives, res::Results, M; tol::Float64 = 1e-8)
        @unpack ns, markov, ν = prim
        @unpack policy = res

        μ0 = ones(ns)./ns #initial guess

        iter = 1
        supnorm = 1

        while supnorm > tol
            # iterate on the distribution using the policy function & markov process
            μ1 = zeros(ns)
                for si=1:ns
                    for si_prime=1:ns
                             μ1[si_prime] += (1-policy[si])*markov[si,si_prime]*μ0[si] + (1-policy[si])*markov[si,si_prime]*M*ν[si]
                    end
                end

            #calculate the supnorm
            supnorm = maximum(abs.(μ0 - μ1))
            iter = iter+1
            # update cross-sectional dist
            μ0=μ1
        end
        μ0
end


function LM_clearing(prim::Primitives, res::Results; tol = 1e-4)
    #loop to find SS law of motion and mass of entrants
    M_low = 0.0
    M_high = 100.0
    M = 50.0
    n = 0
    LMC = 100 #initial value for loop

    while abs(LMC) > tol && n < 1000
        #iterate over μ_i, M_i until the labor market clears
        n+=1

        #calculate the distribution
        μ = t_star(prim,res,M)
        res.μ = μ

        #calculte labor market clearing
        Ld = AggLaborDemand(prim,res,M)
        Π = AggProfits(prim,res, M)
        Ns = AggLaborSupply(Π,prim)
        LMC = Ld - Ns

        #adjust M
        if LMC>=0
            M_high = M
        elseif LMC<0
            M_low = M
        end
        M = (M_low+M_high)/2
        println(Ld)
    end
    res.M = M

end

function SolveModel()
    prim, res = Initialize()
    Prices(prim, res)
    LM_clearing(prim,res)
    prim, res
end
