
############
# AbstractDiscreteOp
############

abstract type AbstractDiscreteOp end
const DiscreteOpTuple = Tuple{Vararg{AbstractDiscreteOp}}

Base.:+(d::AbstractDiscreteOp) = d
Base.:+(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:+(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.:-(d::AbstractDiscreteOp) = abstractmethod(d)
Base.:-(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:-(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.:*(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:*(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.size(d::AbstractDiscreteOp) = abstractmethod(d)
materialize(d::AbstractDiscreteOp) = abstractmethod(d)
_materialize(d::AbstractDiscreteOp,x) = materialize(d)*x

_check_dim_mul(A,B) = @assert(size(A,2) == size(B,1))
_check_dim_sum(A,B) = @assert(size(A) == size(B))

############
# UniformScalingDiscreteOp
############
struct UniformScalingDiscreteOp{T<:Number} <: AbstractDiscreteOperator
    λ::T
    s::Int64  # size
end
Base.size(d::UniformScalingDiscreteOp) = (d.s,d.s)

function Base.:*(d::UniformScalingDiscreteOp,x::AbstractVector) 
    _check_dim_mul(d,x)
    return d.λ*x
end
function Base.:*(d1::U,d2::U) where {U<:UniformScalingDiscreteOp}
    _check_dim_mul(d1,d2)
    return UniformScalingDiscreteOp(d1.λ*d2.λ, size(d1,1))
end
function Base.:*(d::UniformScalingDiscreteOp,α::Number)
    return UniformScalingDiscreteOp(d.λ*α, size(d,2))
end
function Base.:*(α::Number,d::UniformScalingDiscreteOp)
    return UniformScalingDiscreteOp(α*d.λ, size(d,1))
end

materialize(d::UniformScalingDiscreteOp) = d.λ

############
# DiscreteOp
############
struct DiscreteOp{D} <: AbstractDiscreteOperator
    op::D
end
Base.:*(d::DiscreteOp,x::AbstractVector) = d.op*x
Base.size(d::DiscreteOp) = size(d.op)
materialize(d::DiscreteOp) = d.op

############
# CompositeDiscreteOp
############
struct CompositeDiscreteOp{D<:DiscreteOpTuple} <: AbstractDiscreteOperator
    maps::D
    function CompositeDiscreteOp(maps::D) where D<:DiscreteOpTuple
        @assert length(maps) > 1
        return new{D}(maps)
    end
end
CompositeDiscreteOp(d1::D, d2::D) where {D<:AbstractDiscreteOp} = CompositeDiscreteOp((d1,d2))
function CompositeDiscreteOp(d1::D, d2::D) where {D<:CompositeDiscreteOp}
    return CompositeDiscreteOp((d1.maps...,d2.maps...)) 
end

function Base.:*(d::CompositeDiscreteOp,x::AbstractVector) 
    _check_dim_mul(d,x)
    # evaluate product from right to left
    y = last(d.maps)*x
    length(d.maps) == 2 && return first(d.maps)*y
    return CompositeDiscreteOp(d.maps[1:end-1])*y
end
function Base.:*(d1::D, d2::D) where {D<:AbstractDiscreteOp}
    _check_dim_mul(d1,d2)
    return CompositeDiscreteOp(d1, d2)
end
function Base.:*(α::Number, d::AbstractDiscreteOp)
    return CompositeDiscreteOp(UniformScalingDiscreteOp(α,size(d,1)), d)
end
function Base.:*(d::AbstractDiscreteOp, α::Number)
    return CompositeDiscreteOp(d, UniformScalingDiscreteOp(α,size(d,2)))
end

function materialize(d::CompositeDiscreteOp)
    return _materialize(d, true)
end
function _materialize(d::CompositeDiscreteOp, x)
    # evaluate operators from right to left
    y = materialize(d.maps[end])*x
    length(d.maps) == 2 && return materialize(d.maps[1])*y
    return _materialize(CompositeDiscreteOp(d.maps[1:end-1]), y)
end

############
# LinearCombinationDiscreteOp
############
struct LinearCombinationDiscreteOp{D<:DiscreteOpTuple} <: AbstractDiscreteOperator
    maps::D
    function LinearCombinationDiscreteOp(maps::D) where D<:DiscreteOpTuple
        @assert length(maps) > 1
        return new{D}(maps)
    end
end
LinearCombinationDiscreteOp(d1::D, d2::D) where {D<:AbstractDiscreteOp} = LinearCombinationDiscreteOp((d1,d2))
function LinearCombinationDiscreteOp(d1::D, d2::D) where {D<:LinearCombinationDiscreteOp}
    return LinearCombinationDiscreteOp((d1.maps...,d2.maps...)) 
end

function Base.:*(d::LinearCombinationDiscreteOp,x::AbstractVector) 
    _check_dim_mul(d,x)
    # evaluate product from left to right
    y = first(d.maps)*x
    for n in 2:lenght(d.maps)
        # TODO: use mul! instead
        y += d.maps[n]*x
    end
    return y
end
function Base.:+(d1::D, d2::D) where {D<:AbstractDiscreteOp}
    _check_dim_sum(d1,d2)
    return LinearCombinationDiscreteOp(d1, d2)
end

function materialize(d::LinearCombinationDiscreteOp)
    # evaluate sum from left to right
    y = materialize(first(d.maps))
    for n in 2:lenght(d.maps)
        y += materialize(d.maps[n])
    end
    return y
end

