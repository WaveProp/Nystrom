# Discrete operators types and methods.
# Based on the LinearMaps.jl package.

############
# AbstractDiscreteOp
############

"""
    abstract type AbstractDiscreteOp
    
A linear operator that can be evaluated using its forward map. It also 
supports additions and compositions with others `::AbstractDiscreteOp`.
"""
abstract type AbstractDiscreteOp end

const DiscreteOpTuple = Tuple{Vararg{AbstractDiscreteOp}}

Base.:+(d::AbstractDiscreteOp) = d
Base.:+(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:+(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.:-(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:-(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.:*(d::AbstractDiscreteOp,_) = abstractmethod(d)
Base.:*(_,d::AbstractDiscreteOp) = abstractmethod(d)
Base.size(d::AbstractDiscreteOp) = abstractmethod(d)
Base.size(d::AbstractDiscreteOp,i::Integer) = size(d)[i]

"""
    materialize(d::AbstractDiscreteOp)

Returns the explicit matrix representation of the underlying operator in 
`d::AbstractDiscreteOp`. This method can be expensive due to 
matrix additions and multiplications.
"""
materialize(d::AbstractDiscreteOp) = abstractmethod(d)
_materialize(d::AbstractDiscreteOp,x) = materialize(d)*x

_check_dim_mul(A,B) = @assert(size(A,2) == size(B,1))
_check_dim_mul(A,B::Density{V}) where {V<:SVector} = @assert(size(A,2) == size(B,1)*length(V))
_check_dim_sum(A,B) = @assert(size(A) == size(B))

############
# UniformScalingDiscreteOp
############
struct UniformScalingDiscreteOp{T<:Number} <: AbstractDiscreteOp
    λ::T
    s::Int64  # size
end
Base.size(d::UniformScalingDiscreteOp) = (d.s,d.s)

function Base.:*(d::UniformScalingDiscreteOp,x::AbstractVecOrMat) 
    _check_dim_mul(d,x)
    return d.λ*x
end
function Base.:*(d1::UniformScalingDiscreteOp,d2::UniformScalingDiscreteOp) 
    _check_dim_mul(d1,d2)
    return UniformScalingDiscreteOp(d1.λ*d2.λ, size(d1,1))
end
function Base.:*(d::UniformScalingDiscreteOp,α::Number)
    return UniformScalingDiscreteOp(d.λ*α, size(d,2))
end
function Base.:*(α::Number,d::UniformScalingDiscreteOp)
    return UniformScalingDiscreteOp(α*d.λ, size(d,1))
end

materialize(d::UniformScalingDiscreteOp) = UniformScaling(d.λ)

############
# DiscreteOp
############
struct DiscreteOp{D} <: AbstractDiscreteOp
    op::D
end
DiscreteOp(d::AbstractDiscreteOp) = d
Base.size(d::DiscreteOp) = size(d.op)
Base.:*(d::DiscreteOp,x::AbstractVecOrMat) = d.op*x
materialize(d::DiscreteOp) = d.op

############
# CompositeDiscreteOp
############
struct CompositeDiscreteOp{D<:DiscreteOpTuple} <: AbstractDiscreteOp
    maps::D
    function CompositeDiscreteOp(maps::D) where D<:DiscreteOpTuple
        @assert length(maps) > 1
        return new{D}(maps)
    end
end
CompositeDiscreteOp(d1::AbstractDiscreteOp, d2::AbstractDiscreteOp) = CompositeDiscreteOp((d1,d2))
CompositeDiscreteOp(d1::AbstractDiscreteOp, d2::CompositeDiscreteOp) = CompositeDiscreteOp((d1,d2.maps...)) 
CompositeDiscreteOp(d1::CompositeDiscreteOp, d2::AbstractDiscreteOp) = CompositeDiscreteOp((d1.maps...,d2)) 
CompositeDiscreteOp(d1::CompositeDiscreteOp, d2::CompositeDiscreteOp) = CompositeDiscreteOp((d1.maps...,d2.maps...)) 

Base.size(d::CompositeDiscreteOp) = (size(first(d.maps),1), size(last(d.maps),2))

function Base.:*(d::CompositeDiscreteOp,x::AbstractVecOrMat) 
    _check_dim_mul(d,x)
    # evaluate product from right to left
    y = last(d.maps)*x
    length(d.maps) == 2 && return first(d.maps)*y
    return CompositeDiscreteOp(d.maps[1:end-1])*y
end
function Base.:*(d1::AbstractDiscreteOp, d2::AbstractDiscreteOp)
    _check_dim_mul(d1,d2)
    return CompositeDiscreteOp(d1, d2)
end
function Base.:*(α::Number, d::AbstractDiscreteOp)
    return CompositeDiscreteOp(UniformScalingDiscreteOp(α,size(d,1)), d)
end
function Base.:*(d::AbstractDiscreteOp, α::Number)
    return CompositeDiscreteOp(d, UniformScalingDiscreteOp(α,size(d,2)))
end
Base.:*(u::UniformScaling, d::AbstractDiscreteOp) = u.λ*d
Base.:*(d::AbstractDiscreteOp, u::UniformScaling) = d*u.λ
function Base.:-(d::AbstractDiscreteOp)
    return CompositeDiscreteOp(UniformScalingDiscreteOp(-1,size(d,1)), d)
end

function materialize(d::CompositeDiscreteOp)
    return _materialize(d, true)
end
function _materialize(d::CompositeDiscreteOp, x)
    # evaluate operators from right to left
    m1 = materialize(d.maps[end])
    y = _mymul(m1,x)   # m1*x
    if length(d.maps) == 2
        m2 = materialize(d.maps[1])
        return _mymul(m2,y)  # m2*y
    end
    return _materialize(CompositeDiscreteOp(d.maps[1:end-1]), y)
end
function _mymul(A, B)
    # this function patches a bug that
    # occurs when a Diagonal{<:SMatrix}
    # is multiplied by a Matrix{<:SMatrix}
    # with SMatrices of different sizes
    if A isa Diagonal{<:SMatrix} && B isa Matrix{<:SMatrix}
        return A.diag.*B
    elseif A isa Matrix{<:SMatrix} && B isa Diagonal{<:SMatrix}
        return A.*permutedims(B.diag)
    else
        return A*B
    end
end

############
# LinearCombinationDiscreteOp
############
struct LinearCombinationDiscreteOp{D<:DiscreteOpTuple} <: AbstractDiscreteOp
    maps::D
    function LinearCombinationDiscreteOp(maps::D) where D<:DiscreteOpTuple
        @assert length(maps) > 1
        return new{D}(maps)
    end
end
LinearCombinationDiscreteOp(d1::AbstractDiscreteOp, d2::AbstractDiscreteOp) = LinearCombinationDiscreteOp((d1,d2))
LinearCombinationDiscreteOp(d1::AbstractDiscreteOp, d2::LinearCombinationDiscreteOp) = LinearCombinationDiscreteOp((d1,d2.maps...)) 
LinearCombinationDiscreteOp(d1::LinearCombinationDiscreteOp, d2::AbstractDiscreteOp) = LinearCombinationDiscreteOp((d1.maps...,d2)) 
LinearCombinationDiscreteOp(d1::LinearCombinationDiscreteOp, d2::LinearCombinationDiscreteOp) = LinearCombinationDiscreteOp((d1.maps...,d2.maps...)) 

Base.size(d::LinearCombinationDiscreteOp) = size(first(d.maps))

function Base.:*(d::LinearCombinationDiscreteOp,x::AbstractVecOrMat) 
    _check_dim_mul(d,x)
    # evaluate product from left to right
    y = first(d.maps)*x
    for n in 2:length(d.maps)
        # TODO: use mul! instead
        y += d.maps[n]*x
    end
    return y
end
function Base.:+(d1::AbstractDiscreteOp, d2::AbstractDiscreteOp)
    _check_dim_sum(d1,d2)
    return LinearCombinationDiscreteOp(d1, d2)
end
function Base.:+(u::UniformScaling, d::AbstractDiscreteOp)
    @assert size(d,1) == size(d,2)
    return UniformScalingDiscreteOp(u.λ,size(d,1)) + d
end
function Base.:+(d::AbstractDiscreteOp, u::UniformScaling)
    @assert size(d,1) == size(d,2)
    return d + UniformScalingDiscreteOp(u.λ,size(d,2))
end
Base.:-(u::UniformScaling, d::AbstractDiscreteOp) = u+(-d)
Base.:-(d::AbstractDiscreteOp, u::UniformScaling) = d+(-u)
Base.:-(d1::AbstractDiscreteOp, d2::AbstractDiscreteOp) = d1+(-d2)

function materialize(d::LinearCombinationDiscreteOp)
    # find a map that is not an UniformScalingDiscreteOp
    map_index = 1
    for _ in eachindex(d.maps)
        !isa(d.maps[map_index], UniformScalingDiscreteOp) && break
        map_index += 1
    end
    msg = """Cannot materialize a LinearCombinationDiscreteOp
    whose maps are all UniformScalingDiscreteOp. Not implemented."""
    @assert (map_index ≤ length(d.maps)) msg
    # materialize map and
    # evaluate the sum from left to right
    y = materialize(d.maps[map_index])
    for n in eachindex(d.maps)
        n == map_index && continue
        y += materialize(d.maps[n])
    end
    return y
end

############
# GMRES and solvers
############
"""
    struct DiscreteOpGMRES{D<:AbstractDiscreteOp,T<:Number,V}

A wrapper around 'op::AbstractDiscreteOp' to use in conjunction with
the `IterativeSolvers.gmres!` method. 
"""
struct DiscreteOpGMRES{D<:AbstractDiscreteOp,T<:Number}
    # T: scalar matrix element type
    op::D
    s::Int64  # scalar matrix size
end
function DiscreteOpGMRES(op::AbstractDiscreteOp, σ::AbstractVector{V}) where V
    @assert size(op,1) == size(op,2)
    _check_dim_mul(op,σ)
    if V <: Number
        s = length(σ)  # size
        return DiscreteOpGMRES{typeof(op),V}(op, s)
    elseif V <: SVector
        T = eltype(V)
        s = length(σ) * length(V)  # size
        return DiscreteOpGMRES{typeof(op),T}(op, s)
    else
        notimplemented()
    end
end
Base.size(g::DiscreteOpGMRES) = (g.s, g.s)
Base.size(g::DiscreteOpGMRES, i) = size(g)[i]
Base.eltype(::DiscreteOpGMRES{D,T}) where {D,T} = T
function LinearAlgebra.mul!(yvec::AbstractVector{T}, g::DiscreteOpGMRES{D,T}, xvec::AbstractVector{T}) where {D,T}
    yvec .= g.op*xvec
    return yvec
end

function IterativeSolvers.gmres!(σ::Density{V},A::AbstractDiscreteOp,μ::Density{V},args...;kwargs...) where V
    log = haskey(kwargs,:log) ? kwargs[:log] : false
    Aop = DiscreteOpGMRES(A, μ)
    if V <: Number
        if log
            vals,hist = gmres!(σ.vals,Aop,μ.vals,args...;kwargs...)
            return σ,hist
        else
            vals = gmres!(σ.vals,Aop,μ.vals,args...;kwargs...)
            return σ
        end
    elseif V <: SVector
        σ_vec = reinterpret(eltype(V),σ.vals)
        μ_vec = reinterpret(eltype(V),μ.vals)
        if log
            vals,hist = gmres!(σ_vec,Aop,μ_vec,args...;kwargs...)
            return σ,hist
        else
            vals = gmres!(σ_vec,Aop,μ_vec,args...;kwargs...)
            return σ
        end
    else
        notimplemented()
    end
end

function Base.:\(A::AbstractDiscreteOp,σ::Density{V}) where V
    Amat = materialize(A)  # assemble the full matrix
    @assert size(Amat,1) == size(Amat,2)
    @assert eltype(Amat) <: Number
    if V <: Number
        @assert eltype(Amat) === V
        vals = Amat \ σ.vals
        return Density(vals, σ.mesh)
    elseif V <: SVector
        T = eltype(V)
        σ_vec = reinterpret(T, σ.vals)
        @assert size(Amat,2) == length(σ_vec)
        vals_vec = Amat \ σ_vec
        vals = reinterpret(V, vals_vec) |> collect
        return Density(vals, σ.mesh)
    else
        notimplemented()
    end
end

############
# show
############
function _show_size(D)
    return "$(size(D,1))×$(size(D,2))"
end

function _show_typeof(D) 
    return _show_size(D) * ' ' * string(typeof(D))
end
function _show_typeof(D::Number) 
    return string(typeof(D))
end
function _show_typeof(D::Nystrom.AbstractDiscreteOp) 
    return _show_size(D) * ' ' * split(string(typeof(D)),'{')[1]
end

function _show_content(D::Nystrom.AbstractDiscreteOp, i)
    abstractmethod(D)
end
function _show_content(D::Nystrom.UniformScalingDiscreteOp, i)
    content = _show_typeof(D) * " with scale: $(D.λ)" * '\n' * ' '^i * _show_typeof(D.λ)
    return content
end
function _show_content(D::Nystrom.DiscreteOp, i)
    content = _show_typeof(D) * " of" *'\n' * ' '^i * _show_typeof(D.op)
    return content
end
function _show_content(D::Union{Nystrom.CompositeDiscreteOp,Nystrom.LinearCombinationDiscreteOp}, i)
    content = _show_typeof(D) * " with $(length(D.maps)) maps:"
    for m in D.maps
        content *= '\n' * ' '^i * _show_content(m, i+2)
    end
    return content
end

Base.show(io::IO, D::Nystrom.AbstractDiscreteOp) = print(io, _show_content(D, 2))
