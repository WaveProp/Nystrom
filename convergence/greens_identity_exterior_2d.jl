using LinearAlgebra
using Nystrom
using StaticArrays
using Random
using Plots
plotlyjs()
Random.seed!(1)

## general definitions
xout = SVector(3,3)
ops = (
    Laplace(;dim=2),
    Helmholtz(;dim=2,k=1.2),
    Elastostatic(;dim=2,μ=2,λ=3)
)
methods = (
    Nystrom.assemble_dim,
    Nystrom.assemble_gk
)

pde      = ops[1]     # choose the pde
assemble = methods[1] # choose the singular integration method

T    = Nystrom.default_density_eltype(pde)
c    = rand(T)

## place to store dofs and error
ndofs = []
ee = []

npatches = [2,4,8,16,32,64,128,256,512]
qorder   =  5

for n in npatches
    ## generate mesh
    Geometry.clear_entities!()
    Ω   = ParametricSurfaces.Circle() |> Geometry.Domain
    Γ   = boundary(Ω)
    M   = meshgen(Γ,(n,))
    mesh = NystromMesh(view(M,Γ),order=qorder)
    push!(ndofs,length(Nystrom.dofs(mesh)))
    ## create exact solution
    u    = (qnode) -> SingleLayerKernel(pde)(xout,qnode)*c
    dudn = (qnode) -> transpose(DoubleLayerKernel(pde)(xout,qnode))*c
    γ₀u   = Density(u,mesh)
    γ₁u   = Density(dudn,mesh)
    γ₀u_norm = norm(norm.(γ₀u,Inf),Inf)
    # define integral operators
    Sop = SingleLayerOperator(pde,mesh)
    Dop = DoubleLayerOperator(pde,mesh)
    ## compute error
    S = assemble(Sop)
    D = assemble(Dop)
    e1    = Nystrom.error_interior_green_identity(S,D,γ₀u,γ₁u)/γ₀u_norm
    push!(ee,norm(e1,Inf))
end

fig = plot(ndofs,ee,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black);
plot!(fig,xlabel="n",ylabel="error");
for p in 1:5
    c = ee[end]*ndofs[end]^p
    plot!(fig,ndofs,c ./ ndofs.^(p),xscale=:log10,yscale=:log10,label="h^$p",ls=:dash);
end
display(fig)
