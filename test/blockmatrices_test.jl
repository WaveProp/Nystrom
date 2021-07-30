using Test
using LinearAlgebra
using Nystrom
using StaticArrays

@testset "Basic tests" begin
    @testset "Block matrix generation" begin
        Tlist = [SMatrix{1,1,ComplexF64,1}, SMatrix{3,3,ComplexF64,9}]
        n, m = 3, 5
        for T in Tlist
            pm = Nystrom.pseudoblockmatrix(T, n, m)
            pm[Nystrom.Block(1,1)] = zero(T)
            @test zero(T) == pm[Nystrom.Block(1,1)]
            @test Nystrom.blocksize(pm) == (n,m)
            @test size(pm) == (n,m).*size(T)
        end
    end

    @testset "Block matrix wrapping" begin
        n = 9
        m = 3
        @test n%m == 0
        T = SMatrix{m,m,Float64,m*m}
        A = rand(n,n)
        Ap = Nystrom.pseudoblockmatrix(A, T)
        @test A[1,1] == Ap[1,1]
        @test A[1:m,1:m] == Ap[Nystrom.Block(1,1)]
        @test A === Nystrom.to_matrix(Ap)
    end

    @testset "Integral operators to PseudoBlockMatrix" begin
        ops = (
            Helmholtz(;dim=3,k=1.2),
            Elastostatic(;dim=3,μ=2,λ=3)
        )
        for pde in ops
            Geometry.clear_entities!()
            Ω   = ParametricSurfaces.Sphere(;radius=1) |> Geometry.Domain
            Γ   = boundary(Ω)
            M   = ParametricSurfaces.meshgen(Γ,(2,2))
            mesh = NystromMesh(view(M,Γ),order=5)
            S = SingleLayerOperator(pde,mesh)
            Sm = Matrix(S)
            Sp = Nystrom.pseudoblockmatrix(S)
            @test Sp[Nystrom.Block(1,1)] == S[1,1]
            @test Sp[Nystrom.Block(7,1)] == S[7,1]
            @test Nystrom.to_matrix(Sp) == Nystrom.to_matrix(S) == Nystrom.blockmatrix_to_matrix(Sm)
        end
    end
end