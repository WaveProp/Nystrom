using Test
using LinearAlgebra
using Nystrom
using StaticArrays

@testset "Basic tests" begin
    pde   = Helmholtz(;dim=3,k=1)
    Nystrom.clear_entities!()
    r = 0.5
    n     = 1
    Ω     = ParametricSurfaces.Sphere(;radius=r) |> Nystrom.Domain
    Γ     = boundary(Ω)
    M     = ParametricSurfaces.meshgen(Γ,(n,n))
    mesh   = NystromMesh(M,order=4)
    𝐒     = SingleLayerOperator(pde,mesh)
    𝐃     = DoubleLayerOperator(pde,mesh)
    @test Nystrom.kernel_type(𝐒) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(𝐃) == Nystrom.DoubleLayer()
end
