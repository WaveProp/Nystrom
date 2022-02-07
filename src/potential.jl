struct IntegralPotential{T,S}
    kernel::T
    surface::S
end

kernel(pot::IntegralPotential) = pot.kernel
surface(pot::IntegralPotential) = pot.surface

kernel_type(pot::IntegralPotential) = kernel_type(pot.kernel)

function (pot::IntegralPotential)(σ::AbstractVector,x)
    f = kernel(pot)
    Γ = surface(pot)
    iter = zip(dofs(Γ),σ)
    out = sum(iter) do (source,σ)
        w = weight(source)
        f(x,source)*σ*w
    end
    return out
end

Base.getindex(pot::IntegralPotential,σ::AbstractVector) = (x) -> pot(σ,x)

SingleLayerPotential(op::AbstractPDE,surf) = IntegralPotential(SingleLayerKernel(op),surf)
DoubleLayerPotential(op::AbstractPDE,surf) = IntegralPotential(DoubleLayerKernel(op),surf)
AdjointDoubleLayerPotential(op::AbstractPDE,surf) = IntegralPotential(AdjointDoubleLayerKernel(op),surf)
HyperSingularPotential(op::AbstractPDE,surf)      = IntegralPotential(HyperSingularKernel(op),surf)
CombinedFieldPotential(op::AbstractPDE,surf;α,β)  = IntegralPotential(CombinedFieldKernel(op,α,β),surf)
AdjointCombinedFieldPotential(op::AbstractPDE,surf;α,β)  = IntegralPotential(AdjointCombinedFieldKernel(op,α,β),surf)