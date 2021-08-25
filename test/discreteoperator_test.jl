using Test
using StaticArrays
using Nystrom
using LinearAlgebra
using Random
Random.seed!(1)

@testset "Discrete Operator test" begin
    T = SVector{3,ComplexF64}
    T1 = SMatrix{3,3,ComplexF64,9}
    T2 = SMatrix{2,3,ComplexF64,6}

    A = rand(T1, 5, 5)
    B = rand(T1, 5, 5)
    C = rand(T2, 5, 5)
    x = rand(T, 5)
    α = rand(ComplexF64)
    Aop = Nystrom.DiscreteOp(A)
    Bop = Nystrom.DiscreteOp(B)
    Cop = Nystrom.DiscreteOp(C)

    @testset "DiscreteOp test" begin
        @test size(A) == size(Aop)
        @test A*x == Aop*x
        @test A == Nystrom.materialize(Aop)
    end

    @testset "CompositeDiscreteOp test" begin
        @test size(C*A) == size(Cop*Aop)
        @test C*(A*x) == Cop*Aop*x == Cop*(Aop*x)
        @test C*A == Nystrom.materialize(Cop*Aop)

        @test size(C*B*A) == size(Cop*Bop*Aop)
        @test length((Cop*Bop*Aop).maps) == 3
        @test C*(B*(A*x)) == Cop*Bop*Aop*x == Cop*Bop*(Aop*x) == Cop*(Bop*Aop)*x
        @test C*(B*A) == Nystrom.materialize(Cop*Bop*Aop)
    end

    @testset "LinearCombinationDiscreteOp test" begin
        @test size(α*A - B) == size(α*Aop - Bop)
        @test (α*(A*x) - B*x) == (α*Aop*x - Bop*x) == (α*Aop - Bop)*x
        @test (α*A - B) == Nystrom.materialize(α*Aop - Bop)

        @test size(α*A - B + α*B) == size(α*Aop - Bop + α*Bop)
        @test (α*(A*x) - B*x + α*(B*x)) == (α*Aop - Bop + α*Bop)*x
        @test (α*A - B + α*B) == Nystrom.materialize(α*Aop - Bop + α*Bop)
    end

    @testset "Mixed tests" begin
        @test size(C*α*(B-α*A)+(1-α)*C) == size(Cop*α*(Bop-α*Aop)+(1-α)*Cop)
        @test C*(α*(B*x-α*(A*x)))+(1-α)*(C*x) == (Cop*α*(Bop-α*Aop)+(1-α)*Cop)*x
        @test C*(α*(B-α*A))+(1-α)*C == Nystrom.materialize(Cop*α*(Bop-α*Aop)+(1-α)*Cop)
    end

    @testset "GMRES and LU tests" begin
        # mesh
        Geometry.clear_entities!()
        Ω = ParametricSurfaces.Sphere(;radius=1) |> Geometry.Domain
        Γ = boundary(Ω)
        M = ParametricSurfaces.meshgen(Γ,(2,2))
        mesh = NystromMesh(view(M,Γ),order=1)

        mat = α*A-B+α*B
        op = α*Aop-Bop+α*Bop
        Amat = Nystrom.blockmatrix_to_matrix(mat)
        xvec = reinterpret(eltype(eltype(x)), x)
        b = reinterpret(eltype(x), Amat\xvec) |> collect
        σx = Density(x, mesh)
        σb = Density(b, mesh)

        ll = Nystrom.DiscreteOpGMRES(op, σx)
        yy = similar(xvec)
        mul!(yy, ll, xvec)

        @testset "Vector case" begin
            @test σb == (op\σx)
            @test σb ≈ Nystrom.gmres(op, σx)
        end

        op_scalar = Nystrom.DiscreteOp(Amat)
        σx_scalar = Density(collect(xvec), mesh)
        σb_scalar = Density(Amat\xvec, mesh)
        @testset "Scalar case" begin
            @test σb_scalar == (op_scalar\σx_scalar)
            @test σb_scalar ≈ Nystrom.gmres(op_scalar, σx_scalar)
        end
    end
end