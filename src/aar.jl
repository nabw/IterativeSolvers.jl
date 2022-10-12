import Base: terate
using Printf
using LinearAlgebra:norm
#using TimerOutputs
export aar, aar!, AARIterable, aar_terator!

mutable struct AARIterable{matT, preclT, precrT, solT, vecT, numT <: Real}
    A::matT
    Q::Matrix # Dense, so we hard-code the type
    R::Matrix # Dense, so we hard-code the type
    x::solT
    b::vecT
    Pl::preclT
    Pr::precrT
    r::vecT
    dr::vecT
    work::vecT
    work2::vecT
    work_depth::vecT
    work_depth2::vecT
    weights::vecT
    r_prev::vecT
    x_prev::solT
    XF::Matrix 
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
    depth::Int
    p::Int
    omega::Real
    beta::Real
    reorthogonalization_factor::Real
end


function appendColumnToMatrix!(A::Matrix, v::Vector, iteration::Int, depth::Int)
    if iteration > depth
        for i in 2:size(A, 2)
	    copy!(view(A,:,i-1), view(A,:,i))
	end
	copy!(view(A,:,size(A,2)), v)
    else
	copy!(view(A,:,iteration), v)
    end
end



# work=(I - Q Q')x, work_depth=Q'x
function computeProjectionStep!(work::Vector, work_depth::Vector, x::Vector, Q::Matrix)
    mul!(work_depth, Q', x)
    mul!(work, Q, work_depth) 
    axpby!(1.0, x, -1.0, work) 
end


# projection is stored in work, new column of R is stored in work_depth
# See Daniel, Gragg, Kaufman, Stewart. Mathematics of Computation (1976).
function computeProjection!(work::Vector, work2::Vector, work_depth::Vector, work_depth2::Vector, x::Vector, Q::Matrix, reorthogonalization_factor::Real)
    norm_prev = norm(x)
    computeProjectionStep!(work, work_depth, x, Q) 
    norm_current = norm(work)
    while norm_current < reorthogonalization_factor * norm_prev 
        norm_prev = norm(work)
        computeProjectionStep!(work2, work_depth2, work, Q) # work is previous solution, we project that one into work2
        axpy!(1.0, work_depth2, work_depth) # We add increment of projection
        norm_current = norm(work2)
        copy!(work, work2)
    end
    normalize!(work)
    norm_current
end

# Note: x does not change
function append_column!(Q::Matrix, R::Matrix, x::Vector, work::Vector, work2::Vector, work_depth::Vector, work_depth2::Vector, position::Int)
    rho = computeProjection!(work, work2, work_depth, work_depth2, x, Q)    
    
    copy!(view(Q,:,position), work)
    copy!(view(R,:,position), work_depth)
    @inbounds R[position, position] = rho
end

function solve_R!(R::Matrix, b::Vector, sol::Vector, realSize::Int64)
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
    for j in 2:size(R, 2)
	copy!(view(R,:,j-1), view(R,:,j))
    end
    @inbounds R[:,size(R,2)] .= 0.0
    # Compute rotations and use then
    for i in 1:(size(R, 2) - 1) # Last is already null
        g, r = givens(R, i, i+1, i)
	lmul!(g, R)
	rmul!(Q, g')
    end
    triu!(R) # triu helps remove some 1e-17 numbers
    @inbounds Q[:,size(Q, 2)] .= 0.0 # Last column disappears
end 

@inline converged(it::AARIterable) = it.residual ≤ it.tol

@inline start(it::AARIterable) = 0

@inline done(it::AARIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)

# Apply matrix A to x. AARIt is used to obtain A, Pr. 
function applyA!(it::AARIterable, x::vecT, work::vecT, result::vecT) where vecT
    ldiv!(work, it.Pr, x)
    mul!(result, it.A, work)
end

# Compute residual b - Ax and return unpreconditioned norm. AARIt is used to obtain A, Pr and b. 
function computeResidual!(it::AARIterable, x::vecT, result::vecT, work::vecT, work2::vecT) where vecT
    applyA!(it, x, work, work2) # Result in work2
    copy!(result, it.b)
    axpy!(-1.0, work2, result) # result = b - Ax
    res_norm = norm(result) 
    ldiv!(it.Pl, result) # result = Pl(b-Ax)
    res_norm
end


###############
# Ordinary AAR #
###############

function iterate(it::AARIterable, iteration::Int=start(it))
    # Compute current residual
    it.residual = computeResidual!(it, it.x, it.r, it.work, it.work2)

    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Update QR factorization
    mk = min(iteration, it.depth)
    if iteration > 0
	iteration > it.depth && remove_first_column!(it.Q, it.R)

	# Add r - r_prev to QR
	copy!(it.dr, it.r)
	axpy!(-1.0, it.r_prev, it.dr) # work = r - r_prev
        axpy!(it.beta, it.dr, view(it.XF,:,mk)) # Set X <- X + beta F
	append_column!(it.Q, it.R, it.dr, it.work, it.work2, it.work_depth, it.work_depth2, mk) # Work (arg 3) does not change
    end

    if (iteration+1) % it.p != 0 || iteration == 0
        # Richardson step
	axpy!(it.omega, it.r, it.x)
    else
        # Anderson step
	mul!(it.work_depth, it.Q', it.r) 
	solve_R!(it.R, it.work_depth, it.weights, mk)

        # Now we do x <- x + beta * r - (X + beta F) weights
	axpy!(it.beta, it.r, it.x) 
	mul!(it.work, it.XF, it.weights)
	axpy!(-1.0, it.work, it.x) 
    end

    # This is X, indexing follows the same previous logic.
    idx = (iteration % it.depth) + 1
    copy!(it.work, it.x)
    axpy!(-1.0, it.x_prev, it.work)
    appendColumnToMatrix!(it.XF, it.work, iteration+1, it.depth)

    copy!(it.x_prev, it.x)
    copy!(it.r_prev, it.r) 
    it.prev_residual = it.residual

    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

# Utility functions

function aar_iterator!(x, A, b, Pl = Identity(), Pr = Identity();
                      abstol::Real = zero(real(eltype(b))),
                      reltol::Real = sqrt(eps(real(eltype(b)))),
                      maxiter::Int = size(A, 2), 
                      depth::Int=1,  
                      p::Int=1,  
                      omega::Real=1.0, 
                      beta::Real=1.0,
                      reorthogonalization_factor::0.0)
    r = copy(b)
    dr = copy(r)

    mv_products=0
    residual = norm(r)
    tolerance = max(reltol * residual, abstol)
    x_prev = copy(x)
    r_prev = copy(r)

    # Return the iterable
    Q = zeros(size(A,1),depth)
    R = zeros(depth,depth)
    work = similar(x)
    work2 = similar(x)
    work_depth = zeros(depth)
    work_depth2 = similar(work_depth)
    weights = zeros(size(work_depth))
    XF = zeros(size(Q))
    prev_residual = 1.0

    AARIterable(A, Q, R, x, b, Pl, Pr, r, dr, work, work2, work_depth, work_depth2, weights, r_prev, x_prev, XF, tolerance, residual, prev_residual, maxiter, mv_products, depth, p, omega, beta, reorthogonalization_factor)

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
             verbose::Bool = false,
             Pl = Identity(),
             Pr = Identity(),
	     depth::Int = 10,
	     p::Int = 5,
	     omega::Real = 1.0,
	     beta::Real = 1.0,
             kwargs...)
    history = ConvergenceHistory(partial = !log)
    history[:abstol] = abstol
    history[:reltol] = reltol
    verbose && @printf("=== aar ===\n%4s\t%4s\t%7s\n","rest","iter","resnorm")
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform AAR
    iterable = aar_iterator!(x, A, b, Pl, Pr; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            depth = depth, p = p, omega = omega, beta = beta, kwargs...)
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
    ldiv!(Pr, iterable.x)
    copy!(x, iterable.x)

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end
