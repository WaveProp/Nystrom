using LinearAlgebra
using Nystrom
using Nystrom.MaxwellDIM
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

## general definitions
xs = SVector(3,3,3)
pde = Maxwell(;k=2)
V = Nystrom.default_density_eltype(pde)
c = rand(V)

## place to store dofs and error
ndofs = []
ee = []

npatches = [2,4,8]
qorder   =  3
for n in npatches
    ## generate mesh
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere() |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = Nystrom.meshgen(Γ,(n,n))
    mesh = NystromMesh(view(M,Γ),order=qorder)
    push!(ndofs,length(Nystrom.dofs(mesh)))
    ## create exact solution
    γ₀    = (qnode) -> MaxwellDIM.EFIEKernel(pde)(qnode,xs)*c
    γ₁    = (qnode) -> MaxwellDIM.MFIEKernel(pde)(qnode,xs)*c
    γ₀E   = Density(γ₀,mesh)
    γ₁E   = Density(γ₁,mesh)
    γ₀E_norm = norm(norm.(γ₀E,Inf),Inf)

    ## compute error
    Tdim, Kdim = MaxwellDIM.maxwell_dim(pde,mesh)
    e1 = (Tdim*γ₁E+Kdim*γ₀E+γ₀E/2)/γ₀E_norm  # interior Stratton-Chu
    push!(ee,norm(e1,Inf))
    @info ndofs[end],ee[end]
end

xx = sqrt.(ndofs)
fig = plot(xx,ee,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black)
plot!(xlabel="√n",ylabel="error",title="$pde")
for p in 1:5
    c = ee[end]*xx[end]^p
    plot!(fig,xx,c ./ xx.^(p),xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)
