using Test
using StaticArrays
using Random
using Nystrom
using LinearAlgebra
Random.seed!(1)

@testset "IFGF test" begin
    k    = π       # wavenumber
    λ    = 2π/k     # wavelength
    ppw  = 4        # points per wavelength
    dx   = λ/ppw    # distance between points
    order = (3,5,5)  # interpolation order per dimension
    pde  = Helmholtz(dim=3,k=k)
    nsample = 20    # number of points to measure the error

    # geometry
    Geometry.clear_entities!()
    geo = ParametricSurfaces.Sphere(;radius=1)
    Ω   = ParametricSurfaces.Domain(geo)
    Γ   = boundary(Ω)
    np  = ceil(Int,2/dx)
    M   = ParametricSurfaces.meshgen(Γ,(np,np))
    nmesh = NystromMesh(M,Γ;order=2)
    Xpts = Ypts = nmesh |> Nystrom.qcoords |> collect
    npts = length(Nystrom.dofs(nmesh))
    T    = ComplexF64
    B    = rand(T,npts)

    # TODO: adapt IFGF to more operators
    for op in (SingleLayerOperator,)
        iop = op(pde,nmesh)
        K = Nystrom.kernel(iop)
        compress = ifgf_compressor(;order)
        L = compress(iop)

        J  = rand(1:npts,nsample)
        B  = randn(T,npts)
        yw = Nystrom.qweights(nmesh) |> collect
        exa = [sum(K(Xpts[i],Ypts[j])*yw[j]*B[j] for j in 1:npts) for i in J]

        y = similar(B)
        mul!(y,L,B)
        er = norm(y[J]-exa) / norm(exa) # relative error
        @test er < 1e-6
    end
end
