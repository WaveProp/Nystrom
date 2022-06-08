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

    a,b = randn(2)
    n = 5
    A = randn(T1,n,n)
    B = randn(T1,n,n)
    C = randn(T2,n,n)
    x = randn(T,n)
    α = randn(ComplexF64)
    Aop = Nystrom.DiscreteOp(A)
    Bop = Nystrom.DiscreteOp(B)
    Cop = Nystrom.DiscreteOp(C)

    @testset "UniformScalingDiscreteOp test" begin
        λ = -3.5
        Uop = Nystrom.UniformScalingDiscreteOp(λ)
        @test UniformScaling(λ)*x == λ*x == Uop*x
        @test UniformScaling(λ) == Nystrom.materialize(Uop)
        y = Uop*x
        @test a*λ*x+b*y == mul!(y,Uop,x,a,b) == y
    end

    @testset "DiscreteOp test" begin
        @test A*x == Aop*x
        @test A == Nystrom.materialize(Aop)
        y = A*x
        @test a*A*x+b*y ≈ mul!(y,Aop,x,a,b) == y
    end

    @testset "CompositeDiscreteOp test" begin
        @test C*(A*x) == Cop*Aop*x == Cop*(Aop*x)
        @test C*A == Nystrom.materialize(Cop*Aop)

        @test length((Cop*Bop*Aop).maps) == 3
        @test C*(B*(A*x)) == Cop*Bop*Aop*x == Cop*Bop*(Aop*x) == Cop*(Bop*Aop)*x
        @test C*(B*A) == Nystrom.materialize(Cop*Bop*Aop)
        op = Bop*Aop
        y  = op*x
        @test a*op*x+b*y ≈ mul!(y,op,x,a,b) == y
    end

    @testset "LinearCombinationDiscreteOp test" begin
        @test (α*(A*x) - B*x) == (α*Aop*x - Bop*x) == (α*Aop - Bop)*x
        @test (α*A - B) == Nystrom.materialize(α*Aop - Bop)

        @test (α*(A*x) - B*x + α*(B*x)) == (α*Aop - Bop + α*Bop)*x
        @test (α*A - B + α*B) == Nystrom.materialize(α*Aop - Bop + α*Bop)
        op = α*Aop-Bop+α*Bop
        y  = op*x
        @test a*op*x+b*y ≈ mul!(y,op,x,a,b) == y
    end

    @testset "Mixed tests" begin
        @test C*(α*(α*x+B*x-α*(A*x)))+(1-α)*(C*x) ≈ (Cop*α*(α*I+Bop-α*Aop)+(1-α)*Cop)*x
        @test C*(α*(α*I+B-α*A))+(1-α)*C == Nystrom.materialize(Cop*α*(α*I+Bop-α*Aop)+(1-α)*Cop)
        op = (Bop*α*(α*I+Bop-α*Aop)+(1-α)*Bop)
        y  = op*x
        @test a*op*x+b*y ≈ mul!(y,op,x,a,b) == y
    end

    @testset "GMRES and LU tests" begin
        # mesh
        clear_entities!()
        Ω = ParametricSurfaces.Sphere(;radius=1) |> Domain
        Γ = boundary(Ω)
        M = ParametricSurfaces.meshgen(Γ,(2,2))
        mesh = NystromMesh(view(M,Γ),order=1)

        Amat = Nystrom.blockmatrix_to_matrix(A)
        Bmat = Nystrom.blockmatrix_to_matrix(B)
        Amat_op = Nystrom.DiscreteOp(Amat)
        Bmat_op = Nystrom.DiscreteOp(Bmat)

        mat = α*Amat-Bmat+α*Bmat
        op = α*Amat_op-Bmat_op+α*Bmat_op
        @assert Nystrom.materialize(op) ≈ mat
        xvec = reinterpret(eltype(eltype(x)), x)
        b = reinterpret(eltype(x), mat\xvec) |> collect
        σx = Density(x, mesh)
        σb = Density(b, mesh)

        ll = Nystrom.DiscreteOpGMRES(op, σx)
        yy = similar(xvec)
        mul!(yy, ll, xvec)
        @assert yy ≈ mat*xvec

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
