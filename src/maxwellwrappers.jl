############
# Potentials
############

"""
    abstract type AbstractMaxwellPotential{T<:AbstractKernel,S}

Type that wraps around `IntegralOperator`. It is used to define
the Maxwell potentials `MaxwellEFIEPotential` and `MaxwellMFIEPotential`.
"""
abstract type AbstractMaxwellPotential{T<:AbstractKernel,S} end

# same interface as IntegralPotential
kernel(p::AbstractMaxwellPotential) = kernel(p.ip)
surface(p::AbstractMaxwellPotential) = surface(p.ip)
kernel_type(p::AbstractMaxwellPotential) = kernel_type(p.ip)
Base.getindex(p::AbstractMaxwellPotential,σ::AbstractVector) = Base.getindex(p.ip,σ)

# EFIE
struct MaxwellEFIEPotential{T,S} <: AbstractMaxwellPotential{T,S}
    ip::IntegralPotential{T,S}
end
MaxwellEFIEPotential(op::Maxwell,surf) = MaxwellEFIEPotential(SingleLayerPotential(op,surf))

# MFIE
struct MaxwellMFIEPotential{T,S} <: AbstractMaxwellPotential{T,S}
    ip::IntegralPotential{T,S}
end
MaxwellMFIEPotential(op::Maxwell,surf) = MaxwellMFIEPotential(DoubleLayerPotential(op,surf))
# overload MFIE potential evaluation
Base.getindex(p::MaxwellMFIEPotential,σ::AbstractVector) = Base.getindex(p.ip,-ncross(σ))


############
# Operators
############

function maxwell_dim(pde,X,args...;kwargs...)
    S, D = single_doublelayer_dim(pde,X,args...;kwargs...)
    N = diagonal_ncross_matrix(X) |> DiscreteOp
    T = N*S     # EFIE
    K = -N*D*N  # MFIE
    return T, K
end
maxwell_efie_dim(pde,X,args...;kwargs...) = maxwell_dim(pde,X,args...;kwargs...)[1]
maxwell_mfie_dim(pde,X,args...;kwargs...) = maxwell_dim(pde,X,args...;kwargs...)[2]

function assemble_maxwell_indirect_cfie(nmesh, α, β, K, T; exterior=true)
    σ = exterior ? 0.5 : -0.5
    N = diagonal_ncross_matrix(nmesh)  |> DiscreteOp
    J = diagonal_jacobian_matrix(nmesh) |> DiscreteOp
    dualJ = diagonal_dualjacobian_matrix(nmesh) |> DiscreteOp
    cfie = dualJ*(σ*α*I + α*K + β*T*N)*J
    return cfie
end

# TODO: delete 
function assemble_dim_nystrom_matrix_Luiz(mesh, α, β, D, S) 
    N, J, dualJ = diagonal_ncross_and_jacobian_matrices(mesh)
    n_qnodes = length(dofs(mesh))
    M = Matrix{SMatrix{2,2,ComplexF64,4}}(undef, n_qnodes, n_qnodes)
    M .= dualJ*(0.5*α*I + N*(-α*D + β*S)*N)*J
    return M
end