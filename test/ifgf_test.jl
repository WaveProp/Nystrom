using Test
using StaticArrays
using Random
using Nystrom
using ParametricSurfaces
using LinearAlgebra
Random.seed!(1)

@testset "IFGF test" begin
    k    = 2π       # wavenumber
    λ    = 2π/k     # wavelength
    ppw  = 4        # points per wavelength
    dx   = λ/ppw    # distance between points
    nmax = 100      # max points per leaf box
    p    = (3,5,5)  # interpolation points per dimension
    pde  = Helmholtz(dim=3,k=k)

    # geometry
    clear_entities!()
    geo = ParametricSurfaces.Sphere(;radius=1)
    Ω   = Domain(geo)
    Γ   = boundary(Ω)
    np  = ceil(2/dx)
    M   = meshgen(Γ,(np,np))
    msh = NystromMesh(M,Γ;order=1)
    nx = length(msh.dofs)
    Xpts = msh.dofs
    B = rand(ComplexF64, nx)

    for op in [SingleLayerOperator, DoubleLayerOperator]
        iop = op(pde,msh)
        compress = IFGFCompressor(;p,nmax)
        A = compress(iop)
        exa = iop*B
        @test A*B ≈ iop*B
    end
end