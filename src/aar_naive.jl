import Base: iterate
using Printf
using LinearAlgebra:norm
import QRupdate: solveR!
#using TimerOutputs
export aar_naive, aar_naive!, AARNIterable, aar_iterator!

mutable struct AARNIterable{matT, preclT, precrT, solT, vecT, numT <: Real}
    A::matT
    x::solT
    b::vecT
    Pl::preclT
    Pr::precrT
    r::vecT
    work::vecT
    work_depth::vecT
    weights::vecT
    r_prev::vecT
    x_prev::solT
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
    state::State
end

@inline converged(it::AARNIterable) = it.residual ≤ it.tol

@inline start(it::AARNIterable) = 0

@inline done(it::AARNIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary AAR #
###############

function iterate(it::AARNIterable, iteration::Int=start(it))

    mk = min(iteration, it.depth)

    # Compute current residual
    it.prev_residual = it.residual
    it.r_prev .= it.r
    ldiv!(it.work, it.Pr, it.x)
    copy!(it.r, it.b)
    mul!(it.r, it.A, it.work, -1, 1)
    it.residual = norm(it.r)

    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Note: Using preconditioned residuals for mixing, 
    # but original one for computing errors
    ldiv!(it.Pl, it.r) 

    if iteration > 0
        # Update F matrix
        it.work .= it.r
        axpy!(-1.0, it.r_prev, it.work) # work = r - r_prev
        appendColumn!(it.F, it.work, iteration, it.depth)

        # Update X matrix
        it.work .= it.x
        axpy!(-1.0, it.x_prev, it.work)
        appendColumn!(it.X, it.work, iteration, it.depth)
    end

    it.x_prev .= it.x
    if (iteration+1) % it.p != 0 || iteration == 0
        axpy!(it.omega, it.r, it.x)
        it.state.ITER_TYPE = "R"
    else
        #qr_F = qr(it.F)
        #mul!(it.work_depth, qr_F.Q.factors', it.r)
        #solveR!(qr_F.R, it.work_depth, it.weights, mk)
        ldiv!(it.weights, svd(it.F), it.r)
        #it.weights = qr_F \ it.r
        #ldiv!(it.weights, qr(it.F), it.r)
        axpy!(1.0, it.r, it.x)
        mul!(it.x, it.X, it.weights, -1, 1)
        mul!(it.x, it.F, it.weights, -1, 1)
        it.state.ITER_TYPE = "A"
    end


    # Return the residual at item and iteration number as state
    it.residual, iteration + 1
end

#####################
# Preconditioned AAR #
#####################

# Utility functions

function aarn_iterator!(x, A, b, Pl = Identity(), Pr = Identity();
                      abstol::Real = zero(real(eltype(b))),
                      reltol::Real = sqrt(eps(real(eltype(b)))),
                      maxiter::Int = size(A, 2),
                      depth::Int=1, p::Int=1, omega::Real=1.0, beta::Real=1.0)
    r = - b + A*x  # - b + A x0
    axpy!(1, r, x) # x1 <- x0 + r0

    #mul!(c, A, x)
    mv_products=1
    residual = norm(r)
    tolerance = max(reltol * residual, abstol)
    x_prev = copy(x)
    r_prev = copy(r)

    # Return the iterable
    work = similar(x)
    weights = zeros(depth)
    work_depth = zeros(depth)
    F = zeros(size(A, 1), depth)
    X = zeros(size(A, 1), depth)
    prev_residual = 1.0
    state = State("", "R")

    AARNIterable(A, x, b, Pl, Pr, r, work, work_depth, weights, r_prev, x_prev, F, X, tolerance, residual, prev_residual, maxiter, mv_products, depth, p, omega, state)

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
             verbose::Bool = false,
             Pl = Identity(),
             Pr = Identity(),
             depth::Int = 5,
             p::Int = 1,
             omega::Real = 1.0,
             kwargs...)
    history = ConvergenceHistory(partial = !log)
    history[:abstol] = abstol
    history[:reltol] = reltol
    verbose && @printf("=== aar-naive ===\n%4s\t%4s\t%7s\n","rest","iter","resnorm")
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform AAR
    iterable = aarn_iterator!(x, A, b, Pl, Pr; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            depth = depth, p = p, omega = omega, kwargs...)
    if log
        history.mvps = iterable.mv_products
    end
    for (iteration, item) = enumerate(iterable)
        if log
            nextiter!(history, mvps = 1)
            push!(history, :resnorm, iterable.residual)
        end
        verbose && @printf("%s %3d\t%1.2e\n", iterable.state.ITER_TYPE, iteration, iterable.residual)
    end

    # Add final correction for right preconditioning
    ldiv!(Pr, iterable.x)
    x .= iterable.x

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end

