using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots
BLAS.set_num_threads(Threads.nthreads())
@info "Threads: $(Threads.nthreads())"

# Convergence test using Carlos'
# operators and definitions

##
k = 3.3
pde = Nystrom.MaxwellCFIE(k)

# geometry and parameters
sph_radius = 2
geo = ParametricSurfaces.Sphere(;radius=sph_radius)
Ω = Geometry.Domain(geo)
Γ = boundary(Ω)

# evaluation mesh
eval_geo = ParametricSurfaces.Sphere(;radius=5)
eval_Ω = Geometry.Domain(eval_geo)
eval_Γ = boundary(eval_Ω)
eval_M = meshgen(eval_Γ,(2,2))
eval_nystrom_mesh = NystromMesh(view(eval_M,eval_Γ);order=4)
eval_mesh = qcoords(eval_nystrom_mesh) |> collect

# exact solution
#= electric dipole
G    = (x,y) -> Nystrom.maxwell_green_tensor(x, y, k)
xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
E    = (dof) -> G(dof,xs)*c
exa  = E.(eval_mesh);
=#
# plane wave
E = (dof) -> -Nystrom.incident_planewave(dof; k)  
exa = Nystrom.mieseries(eval_mesh; k, a=sph_radius, n_terms=80)

## Indirect formulation
n_src = 50         # number of interpolant sources
α = 1              # MFIE constant
β = im*k           # EFIE constant
qorder = 5         # quadrature order 1D
ndofs = Float64[]
errs = Float64[]
iterative = false;
##
for n in [20]
    M     = meshgen(Γ,(n,n))
    nmesh  = NystromMesh(view(M,Γ);order=qorder)
    γ₀E   = ncross(γ₀(E,nmesh))
    S,D   = Nystrom.single_doublelayer_dim(pde,nmesh;n_src=n_src)
    N,J,dualJ = Nystrom.ncross_and_jacobian_matrices(nmesh)

    @info "Assembling matrix..."
    L     = Nystrom.assemble_dim_nystrom_matrix(nmesh, α, β, D, S)
    @info "Solving..."
    rhs   = dualJ*γ₀E
    if iterative
        Pl = Nystrom.blockdiag_preconditioner(nmesh, L)  # left preconditioner
        ϕ_coeff = Density(Nystrom.solve_GMRES(L, rhs;Pl,verbose=true,maxiter=600,restart=600,abstol=1e-6), nmesh)
    else
        ϕ_coeff = Density(Nystrom.solve_LU(L, rhs), nmesh)
    end
    ϕ     = J*ϕ_coeff
    Spot  = Nystrom.maxwellCFIE_SingleLayerPotencial(pde, nmesh)
    Dpot  = Nystrom.maxwellCFIE_DoubleLayerPotencial(pde, nmesh)
    Eₐ    = (x) -> α*Dpot(ϕ,x) + β*Spot(ncross(ϕ),x)
    er    = (Eₐ.(eval_mesh) - exa)/norm(exa,Inf)
    ndof = length(Nystrom.dofs(nmesh))
    err = norm(er,Inf)

    push!(ndofs, ndof)
    push!(errs, err)
    @info "" ndof err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errs,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black)
title = "k=$k, α=$α, β=$β, qorder=$qorder"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errs[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)