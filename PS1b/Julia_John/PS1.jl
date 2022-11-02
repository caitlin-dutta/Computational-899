using StatFiles, DataFrames, ForwardDiff, Optim, LatexPrint, BenchmarkTools

df = DataFrame(load("Mortgage_performance_data.dta"))
#select the relevant columns from the data
X = select(df, [:i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :score_1, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5])
#convert into array - I was having trouble with data types, but taking the identity seemed to clear these up
X = identity.(Array(X))
#add a column of ones to handle the intercept term
X = hcat(ones(size(X,1)), X)
#extract the column of Y variables
Y = select(df,[:i_close_first_year])
#convert into a nice array
Y = identity.(Array(Y))

#closed-form logit probability 
function Λ(x)
    val = exp(x)/(1 + exp(x))
    val
end

#function which finds the log likelihood
function log_like(Y, X,β)
    val = sum(log.(Λ.(X*β).^(Y) .* (1 .-Λ.(X*β)).^(1 .-Y)))
    val
end

#function which finds the score (derivative of log likelihood with respect to beta)
function score(Y, X, β)
    g = sum((Y .- Λ.(X*β)) .*X, dims=1) #need to take the sum each column of the resulting matrix (i.e. sum across observations) in order to get a vector (otherwise we'd get a scalar)
    g
end

#this function computes the Hessian matrix for a given beta given observations X
function hessian(X, β)
    H = zeros(size(X,2), size(X,2)) #initialize empty 17x17 array 
    for i=1:size(X, 1) #iterate over observations
        H .-= Λ(X[i,:]'*β)*(1 .- Λ(X[i,:]'*β))*(X[i,:]*X[i,:]') #iteratively subtract each observation's contribution to the Hessian
    end
    H
end

#this function numerically finds the score, which is just the gradient of the log-likelihood. We use the ForwardDiff library
function numerical_score(Y, X, β)
    val = ForwardDiff.gradient(b -> log_like(Y, X, b), β)
    val
end
#as above, we use the ForwardDiff library to numerically compute the Hessian matrix
function numerical_hessian(Y, X, β)
    val = ForwardDiff.hessian(b -> log_like(Y, X, b), β)
    val
end

#this function finds the optimal coefficient vector using Newton's method starting from initial guess β_0
function max_like_newton(Y, X, β_0)
    n = 0
    #find the maximum difference between the initial guess and the first step
    β_next = vec(inv(hessian(X, β_0))*score(Y, X, β_0)')
    error = maximum(abs.(β_0 -β_next))
    tol = 1e-12 #set tolerance 
    while error > tol #continue iterating until the error is greater than the specified tolerance parameter
        β_k = β_next #set previous value as the new value computed in previous iteration
        β_next= β_k - vec( inv(hessian(X, β_k))*score(Y, X, β_k)') #find updated coefficient
        error = maximum(abs.(β_next - β_k)) #find max difference between old vector and new vector
        n +=1 
        println("Iteration: $(n), error: $(error)") #little progress indicator! Optional, I just like knowing things are running
    end
    β_next
end
#construct initial coefficient estimate
β_init = vcat([-1], zeros(16))
#find likelihood evaluated at initial coefficient guess
init_ll = log_like(Y, X, β_init)
#find score at initial coefficient
init_score = score(Y, X, β_init)
#find hessian at initial coefficient
init_hessian = hessian(X, β_init)
#turn the hessian into a latex table! Also round the values because it is just so massive
lap(map.(x -> round(x), hessian(X, β_init)))

#find the numerically computed score
num_score = numerical_score(Y, X, β_init)
#find the numerically computed Hessian
num_hess = numerical_hessian(Y, X, β_init)

#compare the numerical results with the analytical results
maximum(abs.(num_score - init_score'))
maximum(abs.(num_hess - init_hessian))

#find the optimal coefficient using our implementation of Newton's method
#time it using belapsed, which runs the code a bunch of times and records the lowest amount of time it takes
@belapsed β_opt = max_like_newton(Y, X, β_init)
#not necessary, find the score at optimal point. This allows us to check that it is zero
score_opt = numerical_score(Y, X, β_opt)
#find the Hessian - again, not necessary. I was just curious
hessian_opt = numerical_hessian(Y,X, β_opt)

#print the (rounded) coefficient vector as a latex table
lap(round.(β_opt, digits=4))

#find optimal coefficient vector using Nelder-Mead and determine the minimum runtime
#note: starting from the initial guess, it does not converge in 1000 iterations. I increased it to 50000 and works
@belapsed opt_NM = optimize(β -> -log_like(Y, X, β), β_init, Optim.Options(iterations=50000, f_abstol=1e-12))
#display optimal coefficient vector
β_NM = opt_NM.minimizer

#find optimal coefficient vector using BFGS (without gradient or Hessian) and determine how long it takes
@belapsed optim_BFGS = optimize(β -> -log_like(Y, X, β), β_init, BFGS(), Optim.Options(f_abstol=1e-12))
β_BFGS = optim_BFGS.minimizer

#combine all results in one array and turn it into a latex-compatible table. 
res = hcat(β_opt, β_BFGS, β_NM)
lap(map.(x -> round(x, digits=4), res))

#create wrapper functions for the likelihood, score, and hessian which take β as inputs. This is solely to play nicely with the way BFGS works when you want to pass a gradient and Hessian. I don't know of a different way to do it, but maybe there is one!
function f(β)
    -log_like(Y, X, β)
end
function g!(G, β)
    G[:] = -score(Y, X, β)
end
function h!(H, β)
    H[:] = -hessian(X,β)
end

#find optimal coefficient using BGFS with the gradient and Hessian supplied, and determine the minimum runtime 
@belapsed opt_bfgs_gh = optimize(f, g!, h!, β_init, BFGS(), Optim.Options(f_abstol=1e-12))
#display coefficient estimate
opt_bfgs_gh.minimizer