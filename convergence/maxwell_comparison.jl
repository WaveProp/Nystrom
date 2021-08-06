using Nystrom
using StaticArrays
using LinearAlgebra
using IterativeSolvers
using Plots
BLAS.set_num_threads(Threads.nthreads())
@info "Threads: $(Threads.nthreads())"

## geometry and parameters
k = 1
sph_radius = 2
n = 16          # number of patches
qorder = 5     # quadrature order
geo = ParametricSurfaces.Sphere(;radius=sph_radius)
Ω = Nystrom.Domain(geo)
Γ = boundary(Ω)
M     = meshgen(Γ,(n,n))
mesh  = NystromMesh(view(M,Γ);order=qorder)
n_src = 50;    # number of sources (must be equal in both formulation)

##
# Carlos formulation
pdeCarlos = Nystrom.MaxwellCFIE(k)
T, K = Nystrom.single_doublelayer_dim(pdeCarlos,mesh;n_src)
T = Nystrom.to_matrix(T)
K = Nystrom.to_matrix(K)

# Luiz formulation
pdeLuiz = Maxwell(;dim=3,k)
N,_,_ = Nystrom.ncross_and_jacobian_matrices(mesh)
S, D = Nystrom.single_doublelayer_dim(pdeLuiz,mesh)
S = Nystrom.to_matrix(S)
D = Nystrom.to_matrix(D)
N = Nystrom.diagonalblockmatrix_to_matrix(N.diag);

## Comparison
nS = N*S
e1 = norm(T-nS)
@info "single layer:" (T ≈ nS)

nD = N*D
Kn = K*N
e2 = norm(nD-Kn)
@info "double layer:" (nD ≈ Kn)
