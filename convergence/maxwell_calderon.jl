using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots
BLAS.set_num_threads(Threads.nthreads())
@info "Threads: $(Threads.nthreads())"

## geometry and parameters
k = 3.3
sph_radius = 2
qorder = 5     # quadrature order
n_src = 50    # number of sources 
pde = Nystrom.MaxwellCFIE(k)

geo = ParametricSurfaces.Sphere(;radius=sph_radius)
Ω = Nystrom.Domain(geo)
Γ = boundary(Ω)

xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
γₒG  = (qnode) -> SingleLayerKernel(pde)(qnode,xs)*c;  

##
ndofs = Float64[]
errs1 = Float64[]
errs2 = Float64[];
##
for n in [2,4,8,12,16]
    M     = meshgen(Γ,(n,n))
    mesh  = NystromMesh(view(M,Γ);order=qorder)
    γ₀E   = reinterpret(ComplexF64, Density(γₒG,mesh).vals)  # n × Eⁱ
    T, K = Nystrom.single_doublelayer_dim(pde,mesh;n_src)
    T = im*k*Nystrom.to_matrix(T)
    K = Nystrom.to_matrix(K)
    
    # Calderon identities
    e1 = norm(T*(T*γ₀E) - K*(K*γ₀E) + γ₀E/4, Inf)  # Calderon 1
    e2 = norm(T*(K*γ₀E) + K*(T*γ₀E), Inf)          # Calderon 2
    ndof = length(Nystrom.dofs(mesh))
    push!(ndofs, ndof)
    push!(errs1, e1)
    push!(errs2, e2)
    @info "" ndof e1 e2
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errs1,xscale=:log10,yscale=:log10,m=:o,label="err Calderon 1",lc=:black)
plot!(sqrt_ndofs,errs2,xscale=:log10,yscale=:log10,m=:o,label="err Calderon 2",lc=:black)
title = "k=$k, qorder=$qorder, R=$sph_radius"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errs2[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)