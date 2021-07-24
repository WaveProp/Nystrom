using Test
using StaticArrays
using Random
using Nystrom
using LinearAlgebra
Random.seed!(1)

@testset "Density test" begin
    Geometry.clear_entities!()
    pde   = Helmholtz(;dim=3,k=1)
    Ω,M   = GmshSDK.sphere(dim=2)
    Γ     = boundary(Ω)
    mesh  = NystromMesh(view(M,Γ),order=1)
    σ     = Density(target->norm(coords(target)),mesh)
    @test eltype(σ) == Float64
    σ     = Density(mesh) do target
        x = coords(target)
        exp(im*2*norm(x))
    end
    @test eltype(σ) == ComplexF64
end

@testset "TangentialDensity test" begin
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
    tan_σ = TangentialDensity(σ)
    @test Density(tan_σ) ≈ σ

    ncross_σ = ncross(σ)
    ncross_tan_σ = ncross(tan_σ)
    @test TangentialDensity(ncross_σ) ≈ ncross_tan_σ
    @test Density(ncross_tan_σ) ≈ ncross_σ
end
