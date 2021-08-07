using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots
BLAS.set_num_threads(Threads.nthreads())
@info "Threads: $(Threads.nthreads())"

##
k = 3.3
pde = Nystrom.MaxwellCFIE(;k)

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
Gh   = (x,y) -> 1/(im*k)*Nystrom.maxwell_curl_green_tensor(x, y, k)
xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
E    = (dof) -> G(dof,xs)*c
H    = (dof) -> Gh(dof,xs)*c
exa  = E.(eval_mesh);

## Indirect formulation
n_src = 50         # number of interpolant sources
η = 0
qorder = 5         # quadrature order 
ndofs = Float64[]
errs = Float64[]
iterative = true;
##
# Load a mesh with quadratic elements
for n in [4]
    M     = meshgen(Γ,(n,n))
    mesh  = NystromMesh(view(M,Γ);order=qorder)
    γ₀E   = ncross(γ₀(E,mesh))    # n × E
    γ₀H   = ncross(γ₀(H,mesh))    # n × H
    S,D   = Nystrom.single_doublelayer_dim(pde,mesh;n_src)
    N,J,dualJ = Nystrom.ncross_and_jacobian_matrices(mesh)
    rhs = dualJ*((1-η)*γ₀H + η*N*γ₀E)

    @info "Assembling matrix..."
    L = Nystrom.assemble_direct_nystrom(pde, mesh, η, D, S)
    @info "Solving..."
    if iterative
        ϕ_coeff = Density(Nystrom.solve_GMRES(L,rhs;verbose=true,maxiter=600,restart=600,abstol=1e-6), mesh)
    else
        ϕ_coeff = Density(Nystrom.solve_LU(L,rhs), mesh)
    end
    ϕ     = J*ϕ_coeff
    Spot  = Nystrom.maxwellCFIE_SingleLayerPotencial(pde, mesh)
    Eₐ    = (x) -> im*k*Spot(ϕ,x)
    er    = (Eₐ.(eval_mesh) - exa)/norm(exa,Inf)
    ndof = length(Nystrom.dofs(mesh))
    err = norm(er,Inf)

    push!(ndofs, ndof)   
    push!(errs, err)     
    @info "" ndof err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errs,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black)
title = "CFIE direct, k=$k, η=$η, qorder=$qorder"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errs[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)