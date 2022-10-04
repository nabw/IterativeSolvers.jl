import Base: iterate
using Printf
using LinearAlgebra:norm
export aar, aar!, AARIterable, aar_iterator!, AARStateVariables

mutable struct AARIterable{matT, solT, vecT, numT <: Real}
    A::matT
    x::solT
    b::vecT
    r::vecT
    u::vecT
    work::vecT
    r_prev::vecT
    u_prev::solT
    F::Vector{vecT}
    X::Vector{vecT}
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

@inline converged(it::AARIterable) = it.residual ≤ it.tol

@inline start(it::AARIterable) = 0

@inline done(it::AARIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)


###############
# Ordinary AAR #
###############

function iterate(it::AARIterable, iteration::Int=start(it))
    # Compute current residual
    #mul!(it.work, it.A, it.u)
    #it.r .= it.b - it.work
    it.r .= it.b - it.A * it.u

    # Update F matrix
    if iteration > 0
        push!(it.F, it.r - it.r_prev)
        if size(it.F, 2) > it.depth # Reduce matrix
            deleteat!(it.F, 1)
        end
    end

    # Check for termination first
    maxit = it.maxiter
    res = it.residual
    tole = it.tol
    if done(it, iteration)
        return nothing
    end


    if (iteration+1) % it.p != 0 || iteration == 0
	print("J ")
        it.u .= it.u + it.omega * it.r
    else
	print("A ")
	F = reduce(hcat, it.F)
	X = reduce(hcat, it.X)
	FT = transpose(F)
	QR = qr(F) 
	RT = transpose(QR.R)
	Q = Matrix(QR.Q)
	#alphas = QR \ it.r
	B = (X + it.beta * F) * inv(RT * QR.R) * FT
	it.u .= it.u + it.beta * it.r - B * it.r
    end

    # Update X matrix
    push!(it.X, it.u - it.u_prev)
    if size(it.X, 2) > it.depth # Reduce matrix
        deleteat!(it.X, 1)
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

function aar_iterator!(x, A, b, Pl = Identity();
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
    AARIterable(A, x, b, r, u, similar(x), r_prev, u_prev, [], [], tolerance, residual, one(residual), maxiter, mv_products, depth, p, omega, beta)
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
    iterable = aar_iterator!(x, A, b, Pl; abstol = abstol, reltol = reltol, maxiter = maxiter,
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

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end
