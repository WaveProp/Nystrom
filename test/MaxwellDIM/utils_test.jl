using Test
using Nystrom
using StaticArrays
using LinearAlgebra
using Nystrom.SparseArrays
using Nystrom.MaxwellDIM

@testset "Utils" begin
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere(;radius=3) |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = ParametricSurfaces.meshgen(Γ,(2,2))
    mesh = NystromMesh(view(M,Γ),order=2)
    k = 1
    pde = Maxwell(;k)

    @testset "Block diagonal preconditioner" begin
        _,K = MaxwellDIM.maxwell_dim(pde,mesh)  # MFIE
        Kmat = Nystrom.materialize(K)
        # check that precondicioner is equal to operator
        # on the block diagonal
        Kprecon = MaxwellDIM.blockdiag(K)
        I,J,V = findnz(Kprecon)
        passtest = true
        for (i,j,v) in zip(I,J,V)
            passtest &= (Kmat[i,j] == v)
            passtest || break
        end
        @test passtest
    end

    @testset "Scalar2VectorOp test" begin
        dofs_per_qnode = 4
        n = 11*dofs_per_qnode
        T = ComplexF64
    
        A = rand(T,n,n)
        x = rand(SVector{dofs_per_qnode,T},n)
        xr = reinterpret(T,x)
        y = deepcopy(x)
        b = A*x
    
        As = MaxwellDIM.Scalar2VectorOp(A;dofs_per_qnode)
        @test As*xr ≈ reinterpret(T,b)
        @test As*x ≈ b
        @test mul!(y,As,x) ≈ y ≈ b
    end

    @testset "Helmholtz regularizer" begin
        δ=1/2
        R = MaxwellDIM.helmholtz_regularizer(pde, mesh; δ)
        k_helmholtz = im*k*δ
        pdeHelmholtz = Helmholtz(;dim=3,k=k_helmholtz)
        S,_ = Nystrom.single_doublelayer_dim(pdeHelmholtz,mesh)
        σ = rand(SVector{3,ComplexF64},length(Nystrom.dofs(mesh)))
        y = S*σ
        w = deepcopy(y)

        σd = Nystrom.Density(σ,mesh)
        @test R*σd ≈ R*σ ≈ mul!(w,R,σ) ≈ y
    end
end