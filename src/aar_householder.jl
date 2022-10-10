import Base: iterate
using Printf
using LinearAlgebra:norm
#using TimerOutputs
export aar, aar!, AARIterable, aar_iterator!, AARStateVariables

struct HouseholderMatrix{vecT}
    v::vecT
    level::Int
end

function applyHouseholder!(y::vecT, H::HouseholderMatrix, x::vecT)
    for i in 1:(H.level-1)
        y[i] = x[i]
    end
    xx = x[H.level:size(x,1)] # Truncated vector
    scale = -2 * dot(xx, H.v)
    y[H.level:size(y,1)] .= xx + scale * H.v
end

mutable struct AARIterable{matT, preclT, precrT, solT, vecT, numT <: Real}
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
    work_depth::vecT
    weights::vecT
    r_prev::vecT
    u_prev::solT
    FbX::Matrix # Dense matrix containing X + beta F
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
    depth::Int
    p::Int
    omega::Real
    beta::Real
    first_FX_index::Int
end

# Note: x does not change
function append_column!(Q::Matrix, R::Matrix, x::Vector, work::Vector, work_depth::Vector, position::Int)
end

function solve_R!(R::Matrix, b::Vector, sol::Vector, realSize::Int4)
    # Note: R is upper triangular
    @inbounds sol[realSize] = b[realSize] / R[realSize, realSize]
    for i in realSize-1:-1:1
	@inbounds sol[i] = b[i]
	for j in realSize:-1:(i+1)
	    @inbounds sol[i] = sol[i] - R[i,j] * sol[j]
	end
	@inbounds sol[i] = sol[i] / R[i,i]
    end
end

function remove_first_column!(Q::Matrix, R::Matrix)
    # Shift R
    for j in 2:size(R,2)
        @inbounds R[:,j-1] .= R[:,j]
    end
    @inbounds R[:,size(R,2)] .= 0.0
    # Compute rotations and use then
    for i in 1:(size(R,2)-1) # Last is already null
        g, r = givens(R, i, i+1, i)
	lmul!(g, R)
	rmul!(Q, r')
    end
    triu!(R) # triu helps remove some 1e-17 numbers
end 

@inline converged(it::AARIterable) = it.residual ≤ it.tol

@inline start(it::AARIterable) = 0

@inline done(it::AARIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary AAR #
###############

function iterate(it::AARIterable, iteration::Int=start(it))
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

	# Update QR, first remove, then add.
	#if iteration > it.depth # Reduce matrix
        #    #deleteat!(it.F, 1)
	#    remove_first_column!(it.Q, it.R)
        #end
	iteration > it.depth && remove_first_column!(it.Q, it.R)

	append_column!(it.Q, it.R, it.work, it.work2, it.work_depth, mk) # Work (arg 3) does not change

	# Note that X updates first, so now we add contributions.
	# Note also that updating F (here) is done AFTER X, and iterations get shifted.
	rmul!(it.work, it.beta)
	idx = ((iteration-1) % it.depth) + 1 # On second iteration (it = 1) we need first element, thus the -1. This gives something in [0,depth-1], so we add 1. 
	it.FbX[:, idx] .= it.FbX[:, idx] + it.work
        #println("DEBUG F IDX: $idx")

	# If the matrix has already been filled, then change the starting index
	if iteration > it.depth
	    it.first_FX_index += 1
	    it.first_FX_index = it.first_FX_index % it.depth # Stay in {0,depth-1}
	end
    end

    if (iteration+1) % it.p != 0 || iteration == 0
	print("R ")
	axpy!(it.omega, it.r, it.u)
    else
	print("A ")
	mul!(it.work_depth, it.Q', it.r) 
	solve_R!(it.R, it.work_depth, it.weights, mk)

	axpy!(it.beta, it.r, it.u)
        #print("DEBUG WEIGHTS IDX: ")
        for i in 1:mk
	    #axpy!(-it.weights[i], it.X[i], it.u)
	    #axpy!(-it.beta * it.weights[i], it.F[i], it.u)
	    # First index is started from 0, so we consider shift i to {0,mk-1}, take residue, then add 1 again.
	    idx = ((it.first_FX_index + i - 1) % it.depth) + 1
	    #println("DEBUG ERROR FbX ", norm(it.X[i] + it.F[i] - it.FbX[:,idx]))
	    @inbounds axpy!(-it.weights[i], it.FbX[:, idx], it.u)
        end
    end

    # This is X, indexing follows the same previous logic.
    idx = (iteration % it.depth) + 1
    it.FbX[:, idx] .= it.u - it.u_prev
    #println("\nDEBUG X IDX: $idx")
   
    #push!(it.X, it.u - it.u_prev)
    # Update QR, first remove, then add.
    #if size(it.X, 1) > it.depth # Reduce matrix
    #    deleteat!(it.X, 1)
    #end


    it.r_prev .= it.r
    it.u_prev .= it.u
    it.prev_residual = it.residual
    it.residual = norm(it.r)

    # Return the residual at item and iteration number as state
    #if iteration == 10
    #     stop("WA")
    #end
    it.residual, iteration + 1
end

#####################
# Preconditioned AAR #
#####################

# Utility functions

"""
Intermediate AAR state variables to be used inside aar and aar!. `u`, `r` and `c` should be of the same type as the solution of `aar` or `aar!`.
```
struct AARStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end
```
"""
struct AARStateVariables{T,Tx<:AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end

function aar_iterator!(x, A, b, Pl = Identity(), Pr = Identity();
                      abstol::Real = zero(real(eltype(b))),
                      reltol::Real = sqrt(eps(real(eltype(b)))),
                      maxiter::Int = size(A, 2),
                      statevars::AARStateVariables = AARStateVariables(zero(x), similar(x), similar(x)), depth::Int=1, p::Int=1, omega::Real=1.0, beta::Real=1.0)
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
    work_depth = zeros(depth)
    weights = zeros(depth)
    F = []
    X = []
    FbX = zeros(size(A, 1), depth)
    prev_residual = 1.0
    first_FX_index = 0 # We start with 0 to be consistent with % operator
    AARIterable(A, Q, R, x, b, Pl, Pr, r, u, work, work2, work_depth, weights, r_prev, u_prev, FbX, tolerance, residual, prev_residual, maxiter, mv_products, depth, p, omega, beta, first_FX_index)

end

"""
    aar(A, b; kwargs...) -> x, [history]

Same as [`aar!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
aar(A, b; kwargs...) = aar!(zerox(A, b), A, b; kwargs...)

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
function aar!(x, A, b;
             abstol::Real = zero(real(eltype(b))),
             reltol::Real = sqrt(eps(real(eltype(b)))),
             maxiter::Int = size(A, 2),
             log::Bool = false,
             statevars::AARStateVariables = AARStateVariables(zero(x), similar(x), similar(x)),
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
    iterable = aar_iterator!(x, A, b, Pl, Pr; abstol = abstol, reltol = reltol, maxiter = maxiter,
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
