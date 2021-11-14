using Test
using LinearAlgebra
using Nystrom
using StaticArrays

@testset "Basic tests" begin
    pde   = Helmholtz(;dim=3,k=1)
    @gmsh begin
        Geometry.clear_entities!()
        gmsh.initialize()
        GmshSDK.set_meshsize(0.1)
        Geometry.clear_entities!()
        gmsh.model.occ.addSphere(0,0,0,1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Ω = GmshSDK.domain(dim=3)
        M = GmshSDK.meshgen(Ω,dim=3)
        return Ω,M
    end
    Γ     = boundary(Ω)
    mesh  = NystromMesh(view(M,Γ),order=1)
    𝐒     = SingleLayerOperator(pde,mesh)
    𝐃     = DoubleLayerOperator(pde,mesh)
    @test Nystrom.kernel_type(𝐒) == Nystrom.SingleLayer()
    @test Nystrom.kernel_type(𝐃) == Nystrom.DoubleLayer()
end
