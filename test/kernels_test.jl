using Test
using Nystrom
using StaticArrays
using LinearAlgebra

@testset "Kernels" begin
    pde  = Helmholtz(;dim=3,k=1)
    G    = SingleLayerKernel(pde)
    dG    = DoubleLayerKernel(pde)
    @test Nystrom.kernel_type(G) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(dG) == Nystrom.DoubleLayer()
    @test Nystrom.return_type(G) == ComplexF64
    @test Nystrom.return_type(dG) == ComplexF64
    @test Nystrom.combined_field_coefficients(G) == (0,-1)
    @test Nystrom.combined_field_coefficients(dG) == (1,0)

    pde  = Laplace(;dim=3)
    G    = SingleLayerKernel(pde)
    dG    = DoubleLayerKernel(pde)
    @test Nystrom.kernel_type(G) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(dG) == Nystrom.DoubleLayer()
    @test Nystrom.return_type(G) == Float64
    @test Nystrom.return_type(dG) == Float64
    @test Nystrom.combined_field_coefficients(G) == (0,-1)
    @test Nystrom.combined_field_coefficients(dG) == (1,0)

    pde  = Elastostatic(;μ=1,λ=2.0,dim=3)
    G    = SingleLayerKernel(pde)
    dG   = DoubleLayerKernel(pde)
    @test Nystrom.kernel_type(G) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(dG) == Nystrom.DoubleLayer()
    @test Nystrom.return_type(G) == SMatrix{3,3,Float64,9}
    @test Nystrom.return_type(dG) == SMatrix{3,3,Float64,9}
    @test Nystrom.combined_field_coefficients(G) == (0,-1)
    @test Nystrom.combined_field_coefficients(dG) == (1,0)
end
