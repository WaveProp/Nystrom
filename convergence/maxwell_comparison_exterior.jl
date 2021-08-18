using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots
BLAS.set_num_threads(Threads.nthreads())
@info "Threads: $(Threads.nthreads())"

##
k = 1.3
pdeRodrigo = Nystrom.MaxwellCFIE(k)
pdeLuiz = Maxwell(;dim=3,k)

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
G    = (x,y) -> Nystrom.maxwell_green_tensor(x, y, k)
xs   = SVector(0.1,0.2,0.3) 
pol    = SVector(1+im,-2.,3.)
E    = (dof) -> G(dof,xs)*pol
exa  = E.(eval_mesh);

## Indirect formulation
n_src = 50         # number of interpolant sources (Rodrigo)
α = 1              # MFIE constant
β = im*k           # EFIE constant
qorder = 5         # quadrature order 1D
ndofs = Float64[]
errsRodrigo = Float64[]
errsLuiz = Float64[]
iterative = false;
##
for n in [20]
    # Mesh
    M     = meshgen(Γ,(n,n))
    mesh  = NystromMesh(view(M,Γ);order=qorder)
    ndof = length(Nystrom.dofs(mesh))
    push!(ndofs, ndof)
    @info "\nNdofs: $ndof"

    # Rodrigo's version
    γ₀E   = ncross(γ₀(E,mesh))
    S,D   = Nystrom.single_doublelayer_dim(pdeRodrigo,mesh;n_src=n_src)
    N,J,dualJ = Nystrom.ncross_and_jacobian_matrices(mesh)
    @info "Assembling matrix..."
    L     = Nystrom.assemble_dim_nystrom_matrix(mesh, α, β, D, S)
    @info "Solving..."
    rhs   = dualJ*γ₀E
    if iterative
        ϕ_coeff = Density(Nystrom.solve_GMRES(L, rhs;verbose=true,maxiter=600,restart=600,abstol=1e-6), mesh)
    else
        ϕ_coeff = Density(Nystrom.solve_LU(L, rhs), mesh)
    end
    ϕ     = J*ϕ_coeff
    Spot  = Nystrom.maxwellCFIE_SingleLayerPotencial(pdeRodrigo, mesh)
    Dpot  = Nystrom.maxwellCFIE_DoubleLayerPotencial(pdeRodrigo, mesh)
    Eₐ    = (x) -> α*Dpot(ϕ,x) + β*Spot(ncross(ϕ),x)
    er    = (Eₐ.(eval_mesh) - exa)/norm(exa,Inf)
    err = norm(er,Inf)
    push!(errsRodrigo, err)
    @info "Rodrigo:" err

    # Luiz's version
    γ₀E   = γ₀(E,mesh)
    S,D   = Nystrom.single_doublelayer_dim(pdeLuiz,mesh)
    rhs   = dualJ * ncross(γ₀E)
    @info "Assembling matrix..."
    L     = Nystrom.assemble_dim_nystrom_matrix_Luiz(mesh, α, β, D, S)
    @info "Solving..."
    if iterative
        ϕ_coeff = Density(Nystrom.solve_GMRES(L, rhs;verbose=true,maxiter=600,restart=600,abstol=1e-6), mesh)
    else
        ϕ_coeff = Density(Nystrom.solve_LU(L, rhs), mesh)
    end
    ϕ     = J*ϕ_coeff
    Spot  = SingleLayerPotential(pdeLuiz,mesh)
    Dpot  = DoubleLayerPotential(pdeLuiz,mesh)
    Eₐ    = (x) -> -α*Dpot[ncross(ϕ)](x) + β*Spot[ncross(ϕ)](x)
    er    = (Eₐ.(eval_mesh) - exa)/norm(exa,Inf)
    err = norm(er,Inf)
    push!(errsLuiz, err)
    @info "Luiz:" err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errsRodrigo,xscale=:log10,yscale=:log10,m=:o,label="errRodrigo")
plot!(sqrt_ndofs,errsLuiz,xscale=:log10,yscale=:log10,m=:o,label="errLuiz")
title = "k=$k, α=$α, β=$β, qorder=$qorder, R=$sph_radius"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errsRodrigo[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)