"""
    struct IntegralOperator{T,K,S,V} <: AbstractMatrix{T}

Representation of an integral operator which takes a density ϕ defined on `Y`
and integrates it with `kernel` for all elements `x ∈ X`.
"""
struct IntegralOperator{T,K,S,V} <: AbstractMatrix{T}
    kernel::K
    target_surface::S
    source_surface::V
end

function Base.Matrix(iop::IntegralOperator{T}) where {T<:SMatrix}
    # convert the `IntegralOperator` (matrix of SMatrix)
    # into a plain matrix (matrix of scalars)
    N,M = size(iop)
    A,Ablock = MatrixAndBlockIndexer(T,N,M)
    @threads for i in 1:N
        for j in 1:M 
            Ablock[i,j] = iop[i,j]
        end
    end
    return A
end

kernel(iop::IntegralOperator) = iop.kernel
target_surface(iop::IntegralOperator) = iop.target_surface
source_surface(iop::IntegralOperator) = iop.source_surface

kernel_type(iop::IntegralOperator) = iop |> kernel |> kernel_type
combined_field_coefficients(iop::IntegralOperator) = iop |> kernel |> combined_field_coefficients

IntegralOperator{T}(k,X,Y) where {T} = IntegralOperator{T,typeof(k),typeof(X),typeof(Y)}(k, X, Y)

function IntegralOperator(k, X, Y=X)
    T = return_type(k)
    assert_concrete_type(T)
    IntegralOperator{T}(k, X, Y)
end

function Base.size(iop::IntegralOperator)
    X = target_surface(iop)
    Y = source_surface(iop)
    (length(dofs(X)), length(dofs(Y)))
end

# kernel_type(iop::IntegralOperator) = kernel_type(iop.kernel)
function Base.getindex(iop::IntegralOperator,i::Integer,j::Integer)
    k = kernel(iop)
    targets = target_surface(iop) |> dofs
    sources = source_surface(iop) |> dofs
    return k(targets[i],sources[j])*weight(sources[j])
end

# convenience constructors
SingleLayerOperator(op::AbstractPDE,X,Y=X)        = IntegralOperator(SingleLayerKernel(op), X, Y)
DoubleLayerOperator(op::AbstractPDE,X,Y=X)        = IntegralOperator(DoubleLayerKernel(op), X, Y)
AdjointDoubleLayerOperator(op::AbstractPDE,X,Y=X) = IntegralOperator(AdjointDoubleLayerKernel(op), X, Y)
HyperSingularOperator(op::AbstractPDE,X,Y=X)      = IntegralOperator(HyperSingularKernel(op), X, Y)

ambient_dimension(iop::IntegralOperator) = ambient_dimension(iop.kernel)

# Applying Laplace's double-layer to a constant will yield either 1 or -1,
# depending on whether the target point is inside or outside the obstacle
function isinside(x::SVector, mesh::NystromMesh)
    N   = ambient_dimension(mesh)
    pde = Laplace(dim=N)
    K   = DoubleLayerKernel(pde)
    u   = sum(dofs(mesh)) do source
         K(x, source) * weight(source)
    end
    u + 0.5 < 0
    # u < 0
end
isinside(x::Tuple,mesh::NystromMesh) = isinside(SVector(x), mesh)
