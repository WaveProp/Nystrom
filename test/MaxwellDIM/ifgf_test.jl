using Test
using Nystrom
using Nystrom.MaxwellDIM
using Nystrom.IFGF
using LinearAlgebra
using StaticArrays
using Random
Random.seed!(1)

@testset "Maxwell IFGF test" begin
    k = π
    pde = Maxwell(k)
    nsample = 20
    order = (3,3,3)

    Geometry.clear_entities!()
    geo = ParametricSurfaces.Sphere(;radius=1)
    Ω = Geometry.Domain(geo)
    Γ = boundary(Ω)
    M = Nystrom.meshgen(Γ,(4,4))

    nmesh = NystromMesh(view(M,Γ);order=1)
    Xpts = Ypts = nmesh |> Nystrom.qcoords |> collect
    npts = length(Nystrom.dofs(nmesh))
    xn = Nystrom.qnormals(nmesh) |> collect
    yw = Nystrom.qweights(nmesh) |> collect
    J  = rand(1:npts,nsample)
    T  = Nystrom.default_density_eltype(pde)
    B  = randn(T,npts)

    @testset "IFGF operator" begin
        for op in (EFIEOperator,MFIEOperator)
            iop = op(pde,nmesh)
            K = Nystrom.kernel(iop)
            Kpot = MaxwellDIM.potential_kernel(K)
            compress = ifgf_compressor(;order)
            L = compress(iop)
    
            exa = [sum(cross(xn[i],Kpot(Xpts[i],Ypts[j])*(yw[j]*B[j])) for j in 1:npts) for i in J]
    
            y = similar(B)
            mul!(y,L,B)
            er = norm(y[J]-exa) / norm(exa) # relative error
            @test er < 1e-4
        end
    end

    @testset "DIM & IFGF" begin
        ifgf_compress = ifgf_compressor(;order)
        T1,K1 = MaxwellDIM.maxwell_dim(pde,nmesh;compress=Matrix)
        T2,K2 = MaxwellDIM.maxwell_dim(pde,nmesh;compress=ifgf_compress)

        σ = Density(B,nmesh)
        @test norm(T1*σ-T2*σ,Inf) < 1e-10
        @test norm(K1*σ-K2*σ,Inf) < 1e-10
    end
end
