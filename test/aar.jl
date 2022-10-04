module TestAAR

using IterativeSolvers
using LinearMaps
using Test
using LinearAlgebra
using SparseArrays
using Random

import LinearAlgebra.ldiv!

include("laplace_matrix.jl")


@testset "Alternating Anderson Richardson" begin

Random.seed!(1234321)

@testset "Small full system" begin
    n = 10

    @testset "Matrix{$T}" for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand(T, n, n)
        A = A' * A + I
        b = rand(T, n)
        reltol = √eps(real(T))

        x,ch = aar(A, b; reltol=reltol, omega=0.1, maxiter=2n, log=true)
        @test norm(A*x - b) / norm(b) ≤ reltol
        @test ch.isconverged

        # If you start from the exact solution, you should converge immediately
        x,ch = aar!(A \ b, A, b; abstol=2n*eps(real(T)), reltol=zero(real(T)), log=true)


        # All-zeros rhs should give all-zeros lhs
        x0 = aar(A, zeros(T, n))
        @test x0 == zeros(T, n)
    end
end

@testset "Sparse Laplacian" begin
    A = laplace_matrix(Float64, 10, 2)

    rhs = randn(size(A, 2))
    rmul!(rhs, inv(norm(rhs)))
    abstol = 1e-5
    reltol = 1e-5

    @testset "SparseMatrixCSC{$T, $Ti}" for T in (Float64, Float32), Ti in (Int64, Int32)
        xCG = aar(A, rhs; reltol=reltol, omega=0.1, maxiter=1000)
        @test norm(A * xCG - rhs) ≤ reltol
    end

    Af = LinearMap(A)
    @testset "Function" begin
        xCG = aar(Af, rhs; reltol=reltol, omega=0.1, maxiter=1000)
        @test norm(A * xCG - rhs) ≤ reltol
    end

    @testset "Function with specified starting guess" begin
        x0 = randn(size(rhs))
        xCG, hCG = aar!(copy(x0), Af, rhs; abstol=abstol, reltol=0.0, omega=0.1, maxiter=1000, log=true)
        @test norm(A * xCG - rhs) ≤ reltol
    end
end

@testset "CG with a view" begin
    A = rand(10, 10)
    A = A + A' + 100I
    x = view(rand(10, 2), :, 1)
    b = rand(10)
    x, hist = aar!(x, A, b, log = true)
    @test hist.isconverged
end

end

end # module
