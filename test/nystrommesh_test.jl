using Test
using LinearAlgebra
using Nystrom

@testset "Area/volume" begin
    @testset "Cube" begin
        # generate a mesh
        (lx,ly,lz) = widths = (1.,1.,2.)
        Ω,M = @gmsh begin
            Geometry.clear_entities!()
            gmsh.model.occ.addBox(0,0,0,lx,ly,lz)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            Ω = GmshSDK.domain(dim=3)
            M = GmshSDK.meshgen(Ω,dim=3)
            Ω,M
        end
        ∂Ω = boundary(Ω)
        mesh  = NystromMesh(view(M,∂Ω),order=1)
        A     = 2*(lx*ly + lx*lz + ly*lz)
        @test A ≈ sum(qweights(mesh))
        # generate a Nystrom mesh for volume
        mesh  = NystromMesh(view(M,Ω),order=1)
        V     = prod(widths)
        # sum only weights corresponding to tetras
        @test V ≈ sum(qweights(mesh))
    end
    @testset "Sphere" begin
        # create a mesh using GmshSDK package
        r = 0.5
        Geometry.clear_entities!()
        Ω,M   = @gmsh begin
            GmshSDK.set_meshsize(0.1)
            Geometry.clear_entities!()
            gmsh.model.occ.addSphere(0,0,0,r)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
            Ω = GmshSDK.domain(dim=3)
            M = GmshSDK.meshgen(Ω,dim=3)
            Ω,M
        end
        Γ     = boundary(Ω)
        nmesh = NystromMesh(M,Γ,order=4) # NystromMesh of surface Γ
        area  = sum(qweights(nmesh))
        @test isapprox(area,4*π*r^2,atol=5e-2)
        nmesh = NystromMesh(M,Ω,order=4) # Nystrom mesh of volume Ω
        volume  = sum(qweights(nmesh))
        @test isapprox(volume,4/3*π*r^3,atol=1e-2)
        # create a mesh using ParametricSurfaces package
        Geometry.clear_entities!()
        ent   = ParametricSurfaces.Sphere(radius=r)
        Ω     = Geometry.Domain(ent)
        Γ     = boundary(Ω)
        M     = ParametricSurfaces.meshgen(Γ,(4,4))
        nmesh = NystromMesh(view(M,Γ),order=5)
        area   = sum(qweights(nmesh))
        @test isapprox(area,4*π*r^2,atol=1e-5)
    end
    @testset "Circle" begin
        r = rx = ry = 0.5
        Geometry.clear_entities!()
        Ω,M = @gmsh begin
            Geometry.clear_entities!()
            gmsh.model.occ.addDisk(0,0,0,rx,ry)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            Ω = GmshSDK.domain(dim=2)
            M = GmshSDK.meshgen(Ω,dim=2)
            Ω,M
        end
        Γ    = boundary(Ω)
        mesh = NystromMesh(view(M,Ω),order=2)
        A    = π*r^2
        # test area
        @test isapprox(A,sum(qweights(mesh));atol=1e-2)
        # test perimeter
        mesh = NystromMesh(view(M,Γ),order=2)
        P    = 2π*r
        @test isapprox(P,sum(qweights(mesh));atol=1e-2)
        Geometry.clear_entities!()
        geo   = ParametricSurfaces.Circle(radius=r)
        Ω     = Geometry.Domain(geo)
        Γ     = boundary(Ω)
        M     = ParametricSurfaces.meshgen(Γ,(100,))
        nmesh = NystromMesh(M,Γ,order=5)
        @test isapprox(sum(qweights(nmesh)),π,atol=1e-10)
    end
end
