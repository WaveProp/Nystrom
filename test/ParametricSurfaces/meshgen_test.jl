using Test
using Nystrom.ParametricSurfaces

@testset "Disk" begin
    disk = ParametricSurfaces.Circle() # abstract entity
    Γ = boundary(disk) |> Domain
    M = meshgen(Γ,(10,))
    # plot(M,Γ)
    @test ParametricSurfaces.entities(Γ) == boundary(disk)
    @test geometric_dimension(disk) == 2
end

@testset "Ball" begin
    ball = ParametricSurfaces.Sphere() # abstract entity
    Γ = boundary(ball) |> Domain
    M = meshgen(Γ,(1,1))
    # plot(M,Γ)
    @test ParametricSurfaces.entities(Γ) == boundary(ball)
    @test geometric_dimension(ball) == 3
end
