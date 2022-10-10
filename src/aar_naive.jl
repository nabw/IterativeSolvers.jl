import Base: iterate
using Printf
using LinearAlgebra:norm
#using TimerOutputs
export aar_naive, aar_naive!, AARNIterable, aar_iterator!, AARNStateVariables

mutable struct AARNIterable{matT, preclT, precrT, solT, vecT, numT <: Real}
    A::matT
    Q::Matrix # Dense, so we hard-code the type
    R::Matrix # Dense, so we hard-code the type
    x::solT
    b::vecT
    Pl::preclT
    Pr::precrT
    r::vecT
    u::vecT
    work::vecT
    work2::vecT
    weights::vecT
    r_prev::vecT
    u_prev::solT
    F::Matrix
    X::Matrix
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
    depth::Int
    p::Int
    omega::Real
    beta::Real
end

@inline converged(it::AARNIterable) = it.residual ≤ it.tol

@inline start(it::AARNIterable) = 0

@inline done(it::AARNIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary AAR #
###############

function iterate(it::AARNIterable, iteration::Int=start(it))
    # Compute current residual
    ldiv!(it.work, it.Pr, it.u)
    mul!(it.work2, it.A, it.work)
    it.r .= it.b - it.work2

    # Check for termination first
    maxit = it.maxiter
    res = it.residual
    tole = it.tol
    if done(it, iteration)
        return nothing
    end

    # Note: Using preconditioned residuals for mixing, 
    # but original one for computing errors
    ldiv!(it.Pl, it.r) 

    # Update F matrix
    mk = min(iteration, it.depth)
    if iteration > 0
	it.work .= it.r
	axpy!(-1.0, it.r_prev, it.work) # work = r - r_prev
        #push!(it.F, it.r - it.r_prev)

	if iteration > it.depth # Reduce matrix
            #deleteat!(it.F, 1)
	    it.F[:,1:(it.depth-1)] .= it.F[:,2:it.depth]
	    it.F[:,it.depth] .= it.work
	else 
	    it.F[:,iteration] .= it.work
	end
    end

    if (iteration+1) % it.p != 0 || iteration == 0
	print("R ")
	axpy!(it.omega, it.r, it.u)
    else
	print("A ")
	F = it.F[:, 1:mk]
	X = it.X[:, 1:mk]
	weights = (F'F) \ F'it.r
	it.u .= it.u + it.beta * it.r - (X + it.beta * F) * weights
    end

    #push!(it.X, it.u - it.u_prev)
    # Update QR, first remove, then add.
    #if size(it.X, 1) > it.depth # Reduce matrix
    #    deleteat!(it.X, 1)
    #end
    if iteration >= it.depth
        it.X[:,1:(it.depth-1)] .= it.X[:,2:it.depth]
        it.X[:,it.depth] .= it.u - it.u_prev
    else 
        it.X[:,iteration+1] .= it.u - it.u_prev
    end


    it.r_prev .= it.r
    it.u_prev .= it.u
    it.prev_residual = it.residual
    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

#####################
# Preconditioned AAR #
#####################

# Utility functions

"""
Intermediate AAR state variables to be used inside aar and aar!. `u`, `r` and `c` should be of the same type as the solution of `aar` or `aar!`.
```
struct AARNStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end
```
"""
struct AARNStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end

function aarn_iterator!(x, A, b, Pl = Identity(), Pr = Identity();
                      abstol::Real = zero(real(eltype(b))),
                      reltol::Real = sqrt(eps(real(eltype(b)))),
                      maxiter::Int = size(A, 2),
                      statevars::AARNStateVariables = AARNStateVariables(zero(x), similar(x), similar(x)), depth::Int=1, p::Int=1, omega::Real=1.0, beta::Real=1.0)
    u = statevars.u
    r = statevars.r
    c = statevars.c
    u .= x
    copyto!(r, b)

    mul!(c, A, x)
    mv_products=1
    residual = norm(r)
    tolerance = max(reltol * residual, abstol)
    u_prev = copy(u)
    r_prev = copy(r)

    # Return the iterable
    Q = zeros(size(A,1),depth)
    R = zeros(depth,depth)
    work = similar(x)
    work2 = similar(x)
    weights = zeros(depth)
    F = zeros(size(A, 1), depth)
    X = zeros(size(A, 1), depth)
    prev_residual = 1.0
    AARNIterable(A, Q, R, x, b, Pl, Pr, r, u, work, work2, weights, r_prev, u_prev, F, X, tolerance, residual, prev_residual, maxiter, mv_products, depth, p, omega, beta)

end

"""
    aar(A, b; kwargs...) -> x, [history]

Same as [`aar!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
aar_naive(A, b; kwargs...) = aar_naive!(zerox(A, b), A, b; kwargs...)

"""
    aar!(x, A, b; kwargs...) -> x, [history]

# Arguments

- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `statevars::AARStateVariables`: Has 3 arrays similar to `x` to hold intermediate results;
- `Pl = Identity()`: left preconditioner of the method. Should be symmetric,
  positive-definite like `A`;
- `abstol::Real = zero(real(eltype(b)))`,
  `reltol::Real = sqrt(eps(real(eltype(b))))`: absolute and relative
  tolerance for the stopping condition
  `|r_k| ≤ max(reltol * |r_0|, abstol)`, where `r_k ≈ A * x_k - b`
  is approximately the residual in the `k`th iteration.
  !!! note
      The true residual norm is never explicitly computed during the iterations
      for performance reasons; it may accumulate rounding errors.
- `maxiter::Int = size(A,2)`: maximum number of iterations;
- `verbose::Bool = false`: print method information;
- `log::Bool = false`: keep track of the residual norm in each iteration.

# Output

**if `log` is `false`**

- `x`: approximated solution.

**if `log` is `true`**

- `x`: approximated solution.
- `ch`: convergence history.

**ConvergenceHistory keys**

- `:tol` => `::Real`: stopping tolerance.
- `:resnom` => `::Vector`: residual norm at each iteration.
"""
function aar_naive!(x, A, b;
             abstol::Real = zero(real(eltype(b))),
             reltol::Real = sqrt(eps(real(eltype(b)))),
             maxiter::Int = size(A, 2),
             log::Bool = false,
             statevars::AARNStateVariables = AARNStateVariables(zero(x), similar(x), similar(x)),
             verbose::Bool = false,
             Pl = Identity(),
             Pr = Identity(),
	     depth::Int = 5,
	     p::Int = 1,
	     omega::Real = 1.0,
	     beta::Real = 1.0,
             kwargs...)
    history = ConvergenceHistory(partial = !log)
    history[:abstol] = abstol
    history[:reltol] = reltol
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform AAR
    iterable = aarn_iterator!(x, A, b, Pl, Pr; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            statevars = statevars, depth = depth, p = p, omega = omega, beta = beta, kwargs...)
    if log
        history.mvps = iterable.mv_products
    end
    for (iteration, item) = enumerate(iterable)
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, iterable.residual)
        end
        verbose && @printf("%3d\t%1.2e\n", iteration, iterable.residual)
    end

    # Add final correction for right preconditioning
    ldiv!(Pr, iterable.u)
    x .= iterable.u

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end
