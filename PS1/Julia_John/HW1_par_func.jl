#using Parameters, Plots #read in necessary packages
@everywhere @with_kw struct Primitives
    β::Float64 = 0.99 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 1000, stop= 75.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
    Pi::Array{Float64,2} = [0.977  0.023; 0.074  0.926] #initialize productivity transition grid
    prod_mat::Array{Float64,1} = [1.25, 0.2]#productivity matrix
    zk::Int64 = 2#length of productivity vector
end

@everywhere mutable struct Results
    val_func::SharedArray{Float64,2}  #value function
    pol_func::SharedArray{Float64,2} #policy function
end

@everywhere function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = SharedArray{Float64}(zeros(prim.nk, 2)) #initial value function guess
    pol_func = SharedArray{Float64}(zeros(prim.nk,2)) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return structs
end

@everywhere function solve_model(prim::Primitives, res::Results)
    error = 100
    n = 0
    tol = 1e-4
    while error>tol
        n+=1
        v_next = Bellman(prim, res) #find new value function
        error = maximum(abs.(res.val_func .- v_next))/abs(v_next[prim.nk, 1])
        res.val_func = v_next #update value function held in results vector
        #println(n, "  ",  error) #iteration number and error level

        #if either we reach the max number of iterations or our error is less than the tolerance parameter, stop and output the iteration number and max error
        if mod(n, 5000) == 0 || error <tol
            println(" ")
            println("*************************************************")
            println("AT ITERATION = ", n)
            println("MAX DIFFERENCE = ", error)
            println("*************************************************")
        end
    end
end

#Bellman operator (for specified productivity). Takes the primitives struct and results struct as arguments
@everywhere function Bellman(prim::Primitives, res::Results)
    @unpack β,δ,θ,nk, k_grid, Pi, prod_mat,zk = prim
    v_next = SharedArray{Float64}(nk,2)
    @sync @distributed for i_k = 1:nk #iterate over capital grid
        candidate_max_h = -1e10 #very low number 
        candidate_max_l = -1e10 #very low number 
        k = k_grid[i_k]#convert state indices to state values
        budget_h = 1.25*k^θ + (1-δ)*k #budget given high prod state
        budget_l = 0.2*k^θ + (1-δ)*k #budget given low prod state

        for i_kp = 1:nk #loop over choice of k_prime
            kp = k_grid[i_kp]
            c_h = budget_h - kp #consumption when high
            c_l = budget_l - kp #consumption when high
            if c_h>0 #check to make sure that consumption is positive
                val = log(c_h) #utility from consumption choice
                for i in 1:zk #find expectation of value function in the next period
                    val += β*Pi[1,i]*res.val_func[i_kp, i]
                end
                if val>candidate_max_h #check for new max value
                    candidate_max_h = val
                    res.pol_func[i_k,1] = kp #update policy function
                end
            end
            if c_l>0 #check to make sure that consumption is positive
                val = log(c_l) #utility from consumption choice
                for i in 1:zk #find expectation of value function in the next period
                    val += β*Pi[2,i]*res.val_func[i_kp, i]
                end
                if val>candidate_max_l #check for new max value
                    candidate_max_l = val
                    res.pol_func[i_k,2] = kp #update policy function
                end
            end
        end
        v_next[i_k,1] = candidate_max_h #update next guess of high value function
        v_next[i_k,2] = candidate_max_l #update next guess of low value function
    end
    v_next
end

#############
