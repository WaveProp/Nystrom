using Test
using LinearAlgebra
using Nystrom
using StaticArrays
using Random
Random.seed!(1)

@testset "DIM" begin
    # construct interior solution
    rtol=5e-2
    @testset "Greens identity (interior) 2d" begin
        Geometry.clear_entities!()
        Ω   = ParametricSurfaces.Disk() |> Geometry.Domain
        Γ   = boundary(Ω)
        M   = ParametricSurfaces.meshgen(Γ,(5,))
        mesh = NystromMesh(view(M,Γ),order=3)
        xout = SVector(3,3)
        ops = (
            Laplace(;dim=2),
            Helmholtz(;dim=2,k=1.2),
            Elastostatic(;dim=2,μ=2,λ=3)
        )
        for pde in ops
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(qnode,xout)*c
            dudn = (qnode) -> AdjointDoubleLayerKernel(pde)(qnode,xout)*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)
            γ₁u_norm = norm(norm.(γ₁u,Inf),Inf)
            # single and double layer
            S     = SingleLayerOperator(pde,mesh)
            Smat  = S |> Matrix
            D     = DoubleLayerOperator(pde,mesh)
            Dmat = D |> Matrix
            e0   = Nystrom.error_interior_green_identity(Smat,Dmat,γ₀u,γ₁u)/γ₀u_norm
            Sdim  = Nystrom.assemble_dim(S)
            Ddim  = Nystrom.assemble_dim(D)
            e1    = Nystrom.error_interior_green_identity(Sdim,Ddim,γ₀u,γ₁u)/γ₀u_norm
            @testset "Single/double layer $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
            # adjoint double-layer and hypersingular
            K     = AdjointDoubleLayerOperator(pde,mesh)
            Kmat     = K |> Matrix
            H     = HyperSingularOperator(pde,mesh)
            Hmat = H |> Matrix
            e0    = Nystrom.error_interior_derivative_green_identity(Kmat,Hmat,γ₀u,γ₁u)/γ₁u_norm
            Kdim  = Nystrom.assemble_dim(K)
            Hdim  = Nystrom.assemble_dim(H)
            e1   = Nystrom.error_interior_derivative_green_identity(Kdim,Hdim,γ₀u,γ₁u)/γ₁u_norm
            @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                @test norm(e0,Inf) > rtol
                @test norm(e1,Inf) < rtol
            end
        end
    end

    @testset "Greens identity (interior) 3d" begin
        Geometry.clear_entities!()
        Ω   = ParametricSurfaces.Sphere(;radius=1) |> Geometry.Domain
        Γ   = boundary(Ω)
        M   = ParametricSurfaces.meshgen(Γ,(2,2))
        mesh = NystromMesh(view(M,Γ),order=3)
        xout = SVector(3,3,3)
        ops = (
            Laplace(;dim=3),
            Helmholtz(;dim=3,k=1.2),
            Elastostatic(;dim=3,μ=2,λ=3),
            Maxwell(;k=2.)
        )
        for pde in ops
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xout,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xout,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)
            γ₁u_norm = norm(norm.(γ₁u,Inf),Inf)
            # single and double layer
            S     = SingleLayerOperator(pde,mesh)
            Smat  = S |> Matrix
            D     = DoubleLayerOperator(pde,mesh)
            Dmat = D |> Matrix
            e0   = Nystrom.error_interior_green_identity(Smat,Dmat,γ₀u,γ₁u)/γ₀u_norm
            Sdim  = Nystrom.assemble_dim(S)
            Ddim  = Nystrom.assemble_dim(D)
            e1    = Nystrom.error_interior_green_identity(Sdim,Ddim,γ₀u,γ₁u)/γ₀u_norm
            @testset "Single/double layer $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
            # adjoint double-layer and hypersingular
            pde isa Maxwell && continue
            K     = AdjointDoubleLayerOperator(pde,mesh)
            Kmat     = K |> Matrix
            H     = HyperSingularOperator(pde,mesh)
            Hmat = H |> Matrix
            e0   = Nystrom.error_interior_derivative_green_identity(Kmat,Hmat,γ₀u,γ₁u)/γ₁u_norm
            Kdim  = Nystrom.assemble_dim(K)
            Hdim  = Nystrom.assemble_dim(H)
            e1   = Nystrom.error_interior_derivative_green_identity(Kdim,Hdim,γ₀u,γ₁u)/γ₁u_norm
            @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
        end
    end

    @testset "Greens identity (exterior) 2d" begin
        Geometry.clear_entities!()
        Ω   = ParametricSurfaces.Disk() |> Geometry.Domain
        Γ   = boundary(Ω)
        M   = ParametricSurfaces.meshgen(Γ,(7,))
        mesh = NystromMesh(view(M,Γ),order=3)
        xin = SVector(0.1,0.2)
        ops = (
            Laplace(;dim=2),
            Helmholtz(;dim=2,k=1.2),
            Elastostatic(;dim=2,μ=2,λ=3)
        )
        for pde in ops
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xin,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xin,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)
            γ₁u_norm = norm(norm.(γ₁u,Inf),Inf)
            # single and double layer
            S     = SingleLayerOperator(pde,mesh)
            Smat  = S |> Matrix
            D     = DoubleLayerOperator(pde,mesh)
            Dmat = D |> Matrix
            e0   = Nystrom.error_exterior_green_identity(Smat,Dmat,γ₀u,γ₁u)/γ₀u_norm
            Sdim  = Nystrom.assemble_dim(S)
            Ddim  = Nystrom.assemble_dim(D)
            e1    = Nystrom.error_exterior_green_identity(Sdim,Ddim,γ₀u,γ₁u)/γ₀u_norm
            @testset "Single/double layer $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
            # adjoint double-layer and hypersingular
            K     = AdjointDoubleLayerOperator(pde,mesh)
            Kmat     = K |> Matrix
            H     = HyperSingularOperator(pde,mesh)
            Hmat = H |> Matrix
            e0   = Nystrom.error_exterior_derivative_green_identity(Kmat,Hmat,γ₀u,γ₁u)/γ₁u_norm
            Kdim  = Nystrom.assemble_dim(K)
            Hdim  = Nystrom.assemble_dim(H)
            e1   = Nystrom.error_exterior_derivative_green_identity(Kdim,Hdim,γ₀u,γ₁u)/γ₁u_norm
            @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                @test norm(e0,Inf) > rtol
                @test norm(e1,Inf) < rtol
            end
        end
    end

    @testset "Greens identity (exterior) 3d" begin
        Geometry.clear_entities!()
        Ω   = ParametricSurfaces.Sphere(;radius=3) |> Geometry.Domain
        Γ   = boundary(Ω)
        M   = ParametricSurfaces.meshgen(Γ,(4,4))
        mesh = NystromMesh(view(M,Γ),order=3)
        xs = SVector(0.1,-0.1,0.2)
        ops = (
            Laplace(;dim=3),
            Helmholtz(;dim=3,k=1.2),
            Elastostatic(;dim=3,μ=2,λ=3),
            Maxwell(;k=1)
        )
        for pde in ops
            T    = Nystrom.default_density_eltype(pde)
            c    = rand(T)
            u    = (qnode) -> SingleLayerKernel(pde)(xs,qnode)*c
            dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xs,qnode))*c
            γ₀u   = Density(u,mesh)
            γ₁u   = Density(dudn,mesh)
            γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)
            γ₁u_norm = norm(norm.(γ₁u,Inf),Inf)
            # single and double layer
            S     = SingleLayerOperator(pde,mesh)
            Smat  = S |> Matrix
            D     = DoubleLayerOperator(pde,mesh)
            Dmat = D |> Matrix
            e0   = Nystrom.error_exterior_green_identity(Smat,Dmat,γ₀u,γ₁u)/γ₀u_norm
            Sdim  = Nystrom.assemble_dim(S)
            Ddim  = Nystrom.assemble_dim(D)
            e1    = Nystrom.error_exterior_green_identity(Sdim,Ddim,γ₀u,γ₁u)/γ₀u_norm
            @testset "Single/double layer $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
            # adjoint double-layer and hypersingular
            pde isa Maxwell && continue
            K     = AdjointDoubleLayerOperator(pde,mesh)
            Kmat     = K |> Matrix
            H     = HyperSingularOperator(pde,mesh)
            Hmat = H |> Matrix
            e0   = Nystrom.error_exterior_derivative_green_identity(Kmat,Hmat,γ₀u,γ₁u)/γ₁u_norm
            Kdim  = Nystrom.assemble_dim(K)
            Hdim  = Nystrom.assemble_dim(H)
            e1   = Nystrom.error_exterior_derivative_green_identity(Kdim,Hdim,γ₀u,γ₁u)/γ₁u_norm
            @testset "Adjoint double-layer/hypersingular $(string(pde))" begin
                @test norm(e0,Inf) > norm(e1,Inf)
                @test norm(e1,Inf) < rtol
            end
        end
    end
end
