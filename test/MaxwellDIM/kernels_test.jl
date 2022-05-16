using Test
using Nystrom
using StaticArrays
using LinearAlgebra
using Nystrom.MaxwellDIM

k = π
pde   = TrueMaxwell(;k)
Nystrom.clear_entities!()
r     = 0.5
n     = 1
Ω     = ParametricSurfaces.Sphere(;radius=r) |> Nystrom.Domain
Γ     = boundary(Ω)
M     = ParametricSurfaces.meshgen(Γ,(n,n))
mesh  = NystromMesh(M,order=4)

@testset "Basic tests" begin
    T = EFIEOperator(pde,mesh)
    K = MFIEOperator(pde,mesh)
    Ttype = SMatrix{3,3,ComplexF64,9}
    @test Nystrom.parameters(pde) == k
    @test Nystrom.default_kernel_eltype(pde) == Ttype
    @test Nystrom.kernel(T) isa MaxwellDIM.EFIEKernel
    @test Nystrom.kernel(K) isa MaxwellDIM.MFIEKernel
    @test T[1,2] isa Ttype
    @test K[1,2] isa Ttype

    x = @SVector rand(3)
    x = 5*x/norm(x)
    Vtype = SVector{3,ComplexF64}
    ϕ = rand(Vtype,length(Nystrom.dofs(mesh)))
    @test Vtype == Nystrom.default_density_eltype(pde)
    Tpot = EFIEPotential(pde,mesh)
    Kpot = MFIEPotential(pde,mesh)
    @test Nystrom.kernel(Tpot) isa MaxwellDIM.EFIEPotentialKernel
    @test Nystrom.kernel(Kpot) isa MaxwellDIM.MFIEPotentialKernel
    @test Tpot[ϕ](x) isa Vtype
    @test Kpot[ϕ](x) isa Vtype
end

