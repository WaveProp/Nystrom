using LinearAlgebra
using Nystrom
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

## general definitions
xs = SVector(3,3,3)
ops = (
    Laplace(;dim=3),
    Helmholtz(;dim=3,k=1.2),
    Elastostatic(;dim=3,μ=2,λ=3)
)
pde = ops[1] # chose your pde here
T    = Nystrom.default_density_eltype(pde)
c    = rand(T)

## place to store dofs and error
ndofs = []
ee = []

npatches = [2,4,8]
qorder   =  6
for n in npatches
    ## generate mesh
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Sphere() |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = meshgen(Γ,(n,n))
    mesh = NystromMesh(view(M,Γ),order=qorder)
    push!(ndofs,length(Nystrom.dofs(mesh)))
    ## create exact solution
    u    = (qnode) -> SingleLayerKernel(pde)(xs,qnode)*c
    dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xs,qnode))*c
    γ₀u   = Density(u,mesh)
    γ₁u   = Density(dudn,mesh)
    γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)

    ## compute error
    S,D  = Nystrom.single_doublelayer_dim(pde,mesh)
    e1    = Nystrom.error_interior_green_identity(S,D,γ₀u,γ₁u)/γ₀u_norm
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
