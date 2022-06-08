using Test
using Nystrom
using StaticArrays
using Random
Random.seed!(1)

@testset "Interior representation" begin
    # test the interior representation formula u(r) = 𝒮[γ₁u](r) - 𝒟[γ₀u](r)
    @testset "2d" begin
        clear_entities!()
        Ω    = ParametricSurfaces.Disk() |> Domain
        Γ    = boundary(Ω)
        M    = ParametricSurfaces.meshgen(Γ,(10,))
        mesh = NystromMesh(view(M,Γ),order=5)
        x₀   = SVector(0.1,-0.2)
        xout = SVector(3,3)
        ops = (
            Laplace(;dim=2),
            Helmholtz(;dim=2,k=1.2),
            Elastostatic(;dim=2,μ=2,λ=3)
        )
        for pde in ops
            S    = SingleLayerPotential(pde,mesh)
            D    = DoubleLayerPotential(pde,mesh)
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xout,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xout,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            uₐ(x) = S[γ₁u](x) - D[γ₀u](x)
            @test isapprox(u(x₀),uₐ(x₀),rtol=1e-3)
        end
    end
    @testset "3d" begin
        clear_entities!()
        Ω  = ParametricSurfaces.Sphere() |> Domain
        Γ    = boundary(Ω)
        M    = ParametricSurfaces.meshgen(Γ,(4,4))
        mesh = NystromMesh(view(M,Γ),order=5)
        x₀   = SVector(0.1,-0.2,0.1)
        xout = SVector(3,3,3)
        ops = (
            Laplace(;dim=3),
            Helmholtz(;dim=3,k=1.2),
            Elastostatic(;dim=3,μ=2,λ=3)
        )
        for pde in ops
            S    = SingleLayerPotential(pde,mesh)
            D    = DoubleLayerPotential(pde,mesh)
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xout,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xout,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            uₐ(x) = S[γ₁u](x) - D[γ₀u](x)
            @test isapprox(u(x₀),uₐ(x₀),rtol=1e-3)
        end
    end
end

@testset "Exterior representation" begin
    # test the exterior representation formula -u(r) = 𝒮[γ₁u](r) - 𝒟[γ₀u](r)
    @testset "2d" begin
        clear_entities!()
        Ω  = ParametricSurfaces.Disk() |> Domain
        Γ    = boundary(Ω)
        M    = ParametricSurfaces.meshgen(Γ,(10,))
        mesh = NystromMesh(view(M,Γ),order=5)
        x₀   = SVector(3,3)
        xin = SVector(0.1,0.2)
        ops = (
            Laplace(;dim=2),
            Helmholtz(;dim=2,k=1.2),
            Elastostatic(;dim=2,μ=2,λ=3)
        )
        for pde in ops
            S    = SingleLayerPotential(pde,mesh)
            D    = DoubleLayerPotential(pde,mesh)
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xin,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xin,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            uₐ(x) = -S[γ₁u](x) + D[γ₀u](x)
            @test isapprox(u(x₀),uₐ(x₀),rtol=1e-3)
        end
    end
    @testset "3d" begin
        clear_entities!()
        Ω  = ParametricSurfaces.Sphere() |> Domain
        Γ    = boundary(Ω)
        M    = ParametricSurfaces.meshgen(Γ,(4,4))
        mesh = NystromMesh(view(M,Γ),order=5)
        x₀   = SVector(3,3,3)
        xin = SVector(0.1,-0.2,0)
        ops = (
            Laplace(;dim=3),
            Helmholtz(;dim=3,k=1.2),
            Elastostatic(;dim=3,μ=2,λ=3)
        )
        for pde in ops
            S    = SingleLayerPotential(pde,mesh)
            D    = DoubleLayerPotential(pde,mesh)
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xin,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xin,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            uₐ(x) = -S[γ₁u](x) + D[γ₀u](x)
            @test isapprox(u(x₀),uₐ(x₀),rtol=1e-3)
        end
    end
end
