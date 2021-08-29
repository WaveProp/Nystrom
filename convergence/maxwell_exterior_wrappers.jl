using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots

##
k = 3.3
pde = Maxwell(;dim=3,k)

# exact solution
G    = SingleLayerKernel(pde)
xs   = SVector(0.1,0.2,0.3)
x0   = SVector(5,5,5)
pol    = SVector(1+im,-2.,3.)
E    = (dof) -> G(dof,xs)*pol
exa  = E(x0)

# geometry and parameters
sph_radius = 2
geo = ParametricSurfaces.Sphere(;radius=sph_radius)
Ω = Geometry.Domain(geo)
Γ = boundary(Ω)

## CFIE
α = 1
β = im*k
order = 5         # quadrature order 1D
ndofs = Float64[]
errs = Float64[]
iterative = true;
##
for n in [2,4,8]
    M     = meshgen(Γ,(n,n))
    nmesh = NystromMesh(view(M,Γ);order)
    J     = Nystrom.diagonal_jacobian_matrix(nmesh)
    dualJ = Nystrom.diagonal_dualjacobian_matrix(nmesh)
    γ₀E   = γ₀(E,nmesh)
    rhs   = dualJ*ncross(γ₀E)

    @info "Computing operators..."
    T,K = Nystrom.maxwell_dim(pde,nmesh)   # EFIE, MFIE
    @info "Assembling system matrix..."
    L   = Nystrom.assemble_maxwell_indirect_cfie(nmesh, α, β, K, T)
    @info "Solving..."
    if iterative
        ϕ_coeff = gmres(L,rhs; verbose=true,maxiter=600,restart=600,abstol=1e-6)
    else
        ϕ_coeff = L\rhs
    end
    ϕ     = J*ϕ_coeff
    Spot  = Nystrom.MaxwellEFIEPotential(pde,nmesh)
    Dpot  = Nystrom.MaxwellMFIEPotential(pde,nmesh)
    Eₐ    = (x) -> α*Dpot[ϕ](x) + β*Spot[ncross(ϕ)](x)
    er    = (Eₐ(x0) - exa)/norm(exa)

    ndof = length(Nystrom.dofs(nmesh))
    err = norm(er,Inf)

    push!(ndofs, ndof)
    push!(errs, err)
    @info "" ndof err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errs,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black)
title = "k=$k, α=$α, β=$β, qorder=$order"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errs[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)