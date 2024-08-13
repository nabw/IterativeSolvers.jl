import Base: iterate
using Printf
using LinearAlgebra:norm
using QRupdate
import QRupdate: solveR!, solveRT!
using TimerOutputs
export aar, aar!, AARIterable, aar_iterator!

CORRECT = false # Highly experimental

mutable struct State
    ORTH::String  # "" or "O", used for output
    ITER_TYPE::String # R or A, used for output
end

mutable struct QRaar{matT, Int}
    R::matT
    size::Int
end

mutable struct AARIterable{matT, preclT, precrT, solT, vecT, FPiT,  T, vecI, numT <: Real}
    A::matT
    qr::QRaar
    x::solT
    b::vecT
    Pl::preclT
    Pr::precrT
    r::vecT
    work::vecT
    work_depth::vecT
    work_depth2::vecT
    work_depth3::vecT
    work_depth4::vecT
    weights::vecT
    dr::vecT
    dx::solT
    X::Matrix{T} 
    F::Matrix{T} 
    FPi::FPiT
    tol::numT
    residual::numT
    prev_residual::numT
    maxiter::Int
    mv_products::Int
    depth::Int
    p::Int
    omega::Real
    mask::vecI
    state::State
end



function appendColumn!(A::Matrix{T}, v::Vector{T}, iteration::Int, depth::Int) where {T}
    if iteration > depth
        n = size(A,2)
        for i in 2:n
            A[:,i-1] .= @view A[:,i]
        end
        A[:,n] .= v
    else
        A[:,iteration] .= v
    end
end


@inline converged(it::AARIterable) = it.residual ≤ it.tol

@inline start(it::AARIterable) = 0

@inline done(it::AARIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)

# Compute residual b - Ax and return unpreconditioned norm. AARIt is used to obtain A, Pr and b. 
function computeResidual!(it::AARIterable, x::vecT, result::vecT, work::vecT) where vecT
    # Apply A
    ldiv!(work, it.Pr, x) # work <- Pr x
    copy!(result, it.b) # result <- b
    mul!(result, it.A, work, -1, 1) # result = b - Ax

    res_norm = norm(result) 
    ldiv!(it.Pl, result) # result = Pl(b-Ax)
    res_norm
end

function andersonStep!(it::AARIterable, mk::Int)
    # Anderson step, weights are updated here.
    if isnothing(it.mask)
        csne!(it.qr.R, it.F, it.r, it.weights, it.work_depth, it.work_depth2, it.work_depth3, it.work, it.qr.size)
    else    
        csne!(it.qr.R, view(it.FPi, it.mask, :), view(it.r, it.mask), it.weights, it.work_depth, it.work_depth2, it.work_depth3, view(it.work, it.mask), it.qr.size)
    end

    # Now we do x <- x + r - (X + F) weights
    if CORRECT
        # Build complement mask
        Is = zeros(Int, size(it.r,1) - size(it.mask,1))
        i = 1
        for m in 1:size(it.r,1)
            if !(m in it.mask)
                Is[i] = m
                i += 1
            end
        end
        # Update only the part from the mask
        axpy!(1, view(it.r,it.mask), view(it.x, it.mask))
        mul!(view(it.x,it.mask), view(it.X, it.mask, :), it.weights, -1, 1)
        mul!(view(it.x,it.mask), view(it.F, it.mask, :), it.weights, -1, 1)
        # Then impose the consistency correction in the complement
        #dxv = view(it.X+it.F,it.mask, :) * it.weights
        Av = view(it.A, Is, Is)
        BT = view(it.A, Is, it.mask)
        
        #dxv = view(it.X+it.F,it.mask, :) * it.weights
        pnew = zeros(length(it.mask))
        pnew .= view(it.x, it.mask)
        dxv = pnew
        rhs = view(it.b,Is) - BT * dxv # works only with LEFT preconditioning. TODO: Fix to include Pr.
        AA = zeros(size(Av))
        AA .= Av
        unew = AA \ rhs
        view(it.x, Is) .= unew
        #axpy!(1.0, du, view(it.x,Is))
    else
        axpy!(1, it.r, it.x)
        mul!(it.x, it.X, it.weights, -1, 1)
        mul!(it.x, it.F, it.weights, -1, 1)
    end # CORRECT 
end

###############
# Ordinary AAR #
###############

function iterate(it::AARIterable, iteration::Int=start(it))

    mk = min(iteration, it.depth)

    # Compute current residual
    copy!(it.dr, it.r)
    it.residual = computeResidual!(it, it.x, it.r, it.work)
    axpby!(1, it.r, -1, it.dr) # dr = r - r_prev

    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    # Update X and Q-less QR factorization of F
    if iteration > 0
        if isnothing(it.mask)
            if iteration > it.depth  
                qrdelcol!(it.F, it.qr.R, 1) 
                it.qr.size -= 1
            end
            
            appendColumn!(it.X, it.dx, iteration, it.depth)

            qraddcol!(it.F, it.qr.R, it.dr, it.qr.size, it.work_depth, it.work_depth2, it.work_depth3, it.work_depth4, it.work)
            it.qr.size += 1
        else # it.mask is not nothing
            # Note that in this case, R represents the QR of Pi*F.
            if iteration > it.depth  
                qrdelcol!(it.FPi, it.qr.R, 1) 
                it.qr.size -= 1
            end
            
            appendColumn!(it.X, it.dx, iteration, it.depth)
            appendColumn!(it.F, it.dr, iteration, it.depth)

            #it.work .=0 
            #view(it.work, it.mask) = view(it.dr,it.mask)
            qraddcol!(view(it.FPi,it.mask,:), it.qr.R, view(it.dr,it.mask), it.qr.size, it.work_depth, it.work_depth2, it.work_depth3, it.work_depth4, view(it.work,it.mask))
            it.qr.size += 1
           
        end # if isnothing(it.mask)
    end # if iteration > 0

    copy!(it.dx, it.x)
    # Iterate for new x
    if (iteration+1) % it.p != 0 || iteration == 0
            axpy!(it.omega, it.r, it.x)
            it.state.ITER_TYPE = "R"
    else
            andersonStep!(it, mk)
            it.state.ITER_TYPE = "A"
    end
    axpby!(1.0, it.x, -1, it.dx) # dx = x - x_previous

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
                      mask = nothing)
    
    r = - b + A*x  # - b + A x0
    mv_products = 0
    residual = norm(r)
    tolerance = max(reltol * residual, abstol)
    axpy!(1, r, x) # x1 = x0 + r0
    dx = copy(r) # x1 - x0 = r0
    dr = copy(r) # random

    # Return the iterable
    R = zeros(depth,depth)
    qr = QRaar(R, 0)
    work = similar(x)
    weights = zeros(depth)
    work_depth = zeros(depth)
    work_depth2 = similar(work_depth)
    work_depth3 = similar(work_depth)
    work_depth4 = similar(work_depth)
    X = zeros(size(A,1),depth)
    F = zeros(size(A,1),depth)
    FPi = nothing
    if !isnothing(mask)
        FPi = zeros(size(A,1),depth)
    end
    prev_residual = 1.0
    state = State("", "R")

    AARIterable(A, qr, x, b, Pl, Pr, r, work, work_depth, work_depth2, work_depth3, work_depth4, weights, dr, dx, X, F, FPi, tolerance, residual, prev_residual, maxiter, mv_products, depth, p, omega, mask, state)
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
             mask = nothing,
             kwargs...)
    history = ConvergenceHistory(partial = !log)
    history[:abstol] = abstol
    history[:reltol] = reltol
    verbose && @printf("=== aar ===\n%4s\t%4s\t%7s\n","rest","iter","resnorm")
    log && reserve!(history, :resnorm, maxiter + 1)

    # Actually perform AAR
    iterable = aar_iterator!(x, A, b, Pl, Pr; abstol = abstol, reltol = reltol, maxiter = maxiter,
                            depth = depth, p = p, omega = omega, mask = mask, kwargs...)
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
    copy!(x, iterable.x)

    verbose && println()
    log && setconv(history, converged(iterable))
    log && shrink!(history)

    log ? (iterable.x, history) : iterable.x
end
