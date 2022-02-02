using Test
using Nystrom.ParametricSurfaces
using StaticArrays

@testset "Intersection and composite surfaces" begin
    clear_entities!()
    l1   = line((0,0),(1,0));
    l2   = line((1,0),(1,1));
    l3   = line((1,1),(0,1));
    l4   = line((0,1),(0,0));
    l5   = line((1,1),(1,2))
    l6   = line((1,2),(0,2))
    l7   = line((0,2),(0,1))
    ent1 = ElementaryEntity(;boundary=[l1,l2,l3,l4])
    ent2 = ElementaryEntity(;boundary=[l5,l6,l7,ParametricSurfaces.flip_normal(l3)])
    ent3 = ElementaryEntity(;boundary=[l1,l2,l5,l6,l7,l4])
    Ω1 = Domain(ent1)
    Ω2 = Domain(ent2)
    Ω  = union(Ω1,Ω2)
    # intersect. When entities intersect, return them. When they don't, recurse
    # on the boundaries
    @test intersect(Ω,Ω1) == Ω1
    @test intersect(Ω,Ω2) == Ω2
    Γ12 = intersect(Ω1,Ω2)
    Γ21 = intersect(Ω2,Ω1)
    @test Γ12 == Domain(l3)
    @test Γ12 == Γ21
    # test that the orientation is different for the entity in Γ12 and Γ21
    @test normal(Γ12[1],0.5) == SVector(0,1)
    @test normal(Γ21[1],0.5) == SVector(0,-1)
    # check that skeleton includes the interior boundary l3
    Σ  = skeleton(Ω)
    @test l3 ∈ Σ
    @test l3 ∈   internal_boundary(Ω)
    @test !(l3 ∈ external_boundary(Ω))
end

@testset "curves" begin
    f   = (x) -> SVector(cos(x[1]),sin(x[1]))
    d   = HyperRectangle(0,2π)
    ent = ParametricEntity(f,d)
    s  =  SVector(rand())
    @test ent(s) == f(s)
    jac = jacobian(ent,s)
    @test jac[1] ≈ -sin(s[1]) && jac[2] ≈ cos(s[1])
    @test ent(s) ≈ normal(ent,s)
end

@testset "surfaces" begin
    f      = (x) -> SVector(x[1],x[2],sin(x[1]))
    d      = HyperRectangle((0,0),(1,1))
    ent    = ParametricEntity(f,d)
    s      = SVector{2}(rand(2))
    @test ent(s) == f(s)
    @test jacobian(ent,s) ≈ [1 0; 0 1; cos(s[1]) 0]
end

@testset "Parametric entity tests" begin
    @testset "curves" begin
        f   = (x) -> SVector(cos(x[1]),sin(x[1]))
        d   = ReferenceLine()
        ent = ParametricEntity(f,d)
        s  =  SVector(rand())
        @test ent(s) == f(s)
        jac = jacobian(ent,s)
        @test jac[1] ≈ -sin(s[1]) && jac[2] ≈ cos(s[1])
        @test ent(s) ≈ normal(ent,s)
    end
    @testset "surfaces" begin
        f      = (x) -> SVector(x[1],x[2],sin(x[1]))
        d      = ReferenceSquare()
        ent    = ParametricEntity(f,d)
        s      =  SVector{2}(rand(2))
        @test ent(s) == f(s)
        @test jacobian(ent,s) ≈ [1 0; 0 1; cos(s[1]) 0]
    end
end
