using Test
using LinearAlgebra
using Nystrom
using Nystrom.MaxwellDIM
using StaticArrays
using Nystrom.SparseArrays

@testset "Stratton-Chu" begin
    rtol=5e-2
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere(;radius=3) |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = ParametricSurfaces.meshgen(Γ,(4,4))
    mesh = NystromMesh(view(M,Γ),order=3)
    xs = SVector(0.1,-0.1,0.2)
    pde = Maxwell(;k=1)
        
    V    = Nystrom.default_density_eltype(pde)
    c    = rand(V)
    γ₀    = (qnode) -> MaxwellDIM.EFIEKernel(pde)(qnode,xs)*c
    γ₁    = (qnode) -> MaxwellDIM.MFIEKernel(pde)(qnode,xs)*c
    γ₀E   = Density(γ₀,mesh)
    γ₁E   = Density(γ₁,mesh)
    γ₀E_norm = norm(norm.(γ₀E,Inf),Inf)
    Tmat  = EFIEOperator(pde,mesh)|> Matrix
    Kmat  = MFIEOperator(pde,mesh)|> Matrix
    # error in exterior Stratton-Chu identity
    error_strattonchu(EFIE,MFIE) = (EFIE*γ₁E+MFIE*γ₀E-γ₀E/2)/γ₀E_norm
    e0 = error_strattonchu(Tmat,Kmat)
    Tdim, Kdim = MaxwellDIM.maxwell_dim(pde,mesh)
    e1 = error_strattonchu(Tdim,Kdim)
    @test norm(e0,Inf) > norm(e1,Inf)
    @test norm(e1,Inf) < rtol
end

@testset "Block diagonal preconditioner" begin
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere(;radius=3) |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = ParametricSurfaces.meshgen(Γ,(2,2))
    mesh = NystromMesh(view(M,Γ),order=2)
    pde = Maxwell(;k=1)

    _,K = MaxwellDIM.maxwell_dim(pde,mesh)  # MFIE
    Kmat = Nystrom.materialize(K)

    # check that precondicioner is equal to operator
    # on the block diagonal
    Kprecon = MaxwellDIM.blockdiag_preconditioner(K)
    I,J,V = findnz(Kprecon)
    passtest = true
    for (i,j,v) in zip(I,J,V)
        passtest &= (Kmat[i,j] == v)
        passtest || break
    end
    @test passtest
end

