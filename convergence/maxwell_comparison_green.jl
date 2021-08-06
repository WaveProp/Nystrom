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

# exact solution
xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
γₒG  = (qnode) -> SingleLayerKernel(pdeRodrigo)(qnode,xs)*c  # Rodrigo
γ₁G  = (qnode) -> DoubleLayerKernel(pdeRodrigo)(qnode,xs)*c  # Rodrigo
u    = (qnode) -> SingleLayerKernel(pdeLuiz)(xs,qnode)*c             # Luiz
dudn = (qnode) -> transpose(DoubleLayerKernel(pdeLuiz)(xs,qnode))*c  # Luiz

## Indirect formulation
n_src = 50         # number of interpolant sources
qorder = 5         # quadrature order 
ndofs = Float64[]
errsRodrigo = Float64[]
errsLuiz = Float64[];
##
for n in [4,8,12,16]
    # Mesh
    M     = meshgen(Γ,(n,n))
    mesh  = NystromMesh(view(M,Γ);order=qorder)
    ndof = length(Nystrom.dofs(mesh))
    push!(ndofs, ndof)
    @info "\nNdofs: $ndof"

    # Rodrigo
    γ₀E   = Density(γₒG,mesh)
    γ₁E   = Density(γ₁G,mesh) 
    @info "Computing matrices..."
    T,K   = Nystrom.single_doublelayer_dim(pdeRodrigo,mesh;n_src=n_src)
    @info "Computing forward map..."
    γ₀Eₐ = 2*(T*γ₁E + K*γ₀E)
    errR = norm(γ₀Eₐ - γ₀E, Inf) / norm(γ₀E, Inf)
    push!(errsRodrigo, errR)    
    @info "" errR 

    # Luiz
    S,D = Nystrom.single_doublelayer_dim(pdeLuiz,mesh)
    γ₀u = Density(u,mesh)
    γ₁u = Density(dudn,mesh)
    N,_,_ = Nystrom.ncross_and_jacobian_matrices(mesh)
    γ₀Eₐ = -2*N*(S*γ₁u  - D*γ₀u)
    errL = norm(γ₀Eₐ - γ₀E, Inf) / norm(γ₀E, Inf)
    push!(errsLuiz, errL)    
    @info "" errL 
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errsRodrigo,xscale=:log10,yscale=:log10,m=:o,label="errRodrigo")
plot!(sqrt_ndofs,errsLuiz,xscale=:log10,yscale=:log10,m=:o,label="errLuiz")
title = "k=$k, qorder=$qorder, R=$sph_radius"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errsRodrigo[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)