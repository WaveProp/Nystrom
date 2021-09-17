using LinearAlgebra
using StaticArrays
using ParametricSurfaces
using Random
using Nystrom
Random.seed!(1)

# parameters
k   = 4π
λ   = 2π/k
ppw = 16
dx  = λ/ppw

# select pde
pdes = [Laplace(dim=3),
        Helmholtz(dim=3,k=k),
        Elastostatic(dim=3,μ=1,λ=2),
        Maxwell(dim=3,k=k)]
pde = pdes[3]

# select operator
ops = [SingleLayerOperator,
       DoubleLayerOperator]
op = ops[1]

# geometry
clear_entities!()
geo = ParametricSurfaces.Sphere(;radius=1)
Ω   = Domain(geo)
Γ   = boundary(Ω)
np  = ceil(2/dx)
M   = meshgen(Γ,(np,np))
msh = NystromMesh(M,Γ;order=1)
nx = length(msh.dofs);
iop = op(pde,msh) 

# full product estimation
T = Nystrom.default_density_eltype(pde)
I = rand(1:nx,1000)
B = rand(T,nx) # CHECK: randn or rand?
Xpts = msh.dofs
Ypts = msh.dofs
K = (x,y) -> iop.kernel(x,y)*Nystrom.weight(y)
tfull = @elapsed exa = [sum(K(Xpts[i],Ypts[j])*B[j] for j in 1:nx) for i in I]
@info "Estimated time for full product: $(tfull*nx/1000)"

# IFGF
nmax = 100   # max points per leaf box
p = (3,5,5)  # interpolation points per dimension
compress = Nystrom.IFGFCompressor(;p,nmax,_profile=true);
A = compress(iop);
C = zeros(T,nx)
Nystrom.IFGF.@hprofile C = A*B
er = norm(C[I]-exa,2) / norm(exa,2)
@info "" typeof(pde) op (er,nx)
