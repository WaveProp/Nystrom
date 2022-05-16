using Test
using StaticArrays
using Random
using Nystrom
using LinearAlgebra
Random.seed!(1)

@testset "Density test" begin
    # Scalar case
    Geometry.clear_entities!()
    pde   = Helmholtz(;dim=3,k=1)
    Ω     = ParametricSurfaces.Sphere()
    Γ     = boundary(Ω) |> Geometry.Domain
    M     = ParametricSurfaces.meshgen(Γ,(10,10))
    mesh  = NystromMesh(view(M,Γ),order=1)
    σ     = Density(target->norm(coords(target)),mesh)
    @test eltype(σ) == Float64
    σ     = Density(mesh) do target
        x = coords(target)
        exp(im*2*norm(x))
    end
    @test eltype(σ) == ComplexF64

    n = 10
    Amat = rand(ComplexF64,n,length(Nystrom.dofs(mesh)))
    μ = Amat*σ
    @test μ isa Density
    @test eltype(μ) == ComplexF64
    @test length(μ) == n

    # Tensor case
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere(;radius=1) |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = ParametricSurfaces.meshgen(Γ,(2,2))
    mesh = NystromMesh(view(M,Γ),order=2)
    pde = Elastostatic(;dim=3,μ=2,λ=3)
    T = Nystrom.default_density_eltype(pde)
    xout = SVector(3,3,3)
    c = rand(T)
    σ = Density(mesh) do target  # density defined with a tangential field
        x = coords(target)
        cross(normal(target), SingleLayerKernel(pde)(xout,target)*c)
    end
    @test eltype(σ) == SVector{3,Float64}

    n = 10
    V = Nystrom.default_kernel_eltype(pde)
    Amat = rand(eltype(T),(n,length(Nystrom.dofs(mesh))).*size(V))
    μ = Amat*σ
    @test μ isa Density
    @test eltype(μ) == T
    @test length(μ) == n

    @testset "TangentialDensity test" begin
        tan_σ = TangentialDensity(σ)
        @test Density(tan_σ) ≈ σ
        ncross_σ = ncross(σ)
        ncross_tan_σ = ncross(tan_σ)
        @test TangentialDensity(ncross_σ) ≈ ncross_tan_σ
        @test Density(ncross_tan_σ) ≈ ncross_σ
    end
end
