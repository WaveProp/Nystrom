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


