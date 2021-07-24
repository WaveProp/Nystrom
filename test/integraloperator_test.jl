using Test
using LinearAlgebra
using Nystrom
using StaticArrays

@testset "Basic tests" begin
    pde   = Helmholtz(;dim=3,k=1)
    Geometry.clear_entities!()
    Ω,M   = GmshSDK.sphere(dim=2)
    Γ     = boundary(Ω)
    mesh  = NystromMesh(view(M,Γ),order=1)
    𝐒     = SingleLayerOperator(pde,mesh)
    𝐃     = DoubleLayerOperator(pde,mesh)
    @test Nystrom.kernel_type(𝐒) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(𝐃) == Nystrom.DoubleLayer()
end
