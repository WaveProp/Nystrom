
"""
    struct BlockIndexer{T<:SMatrix,S<:Number} <: AbstractMatrix{T}

Wrapper used to index a matrix `op::Matrix{S}` as a matrix of `SMatrix`.
"""
struct BlockIndexer{T<:SMatrix,S<:Number} <: AbstractMatrix{T}
    op::Matrix{S}
    function BlockIndexer{T}(op) where T
        S = eltype(op)
        @assert (isconcretetype(T)&&isconcretetype(S)) "element types must be concrete"
        @assert (S===eltype(T)) "element type of block `T` and `op::Matrix` are not equal"
        @assert (size(op).%size(T)==(0,0)) "size of block `T` and `op::Matrix` are not compatible"
        return new{T,S}(op)
    end
end
BlockIndexer(op,T) = BlockIndexer{T}(op)
BlockIndexer(op::Matrix{S},::Type{S}) where {S<:Number} = op

Base.Matrix(b::BlockIndexer) = b.op
Base.copy(b::BlockIndexer{T}) where T = BlockIndexer(Matrix(b),T)

function Base.size(b::BlockIndexer{T}) where T
    op = Matrix(b)
    return size(op) .÷ size(T)
end
function Base.size(b::BlockIndexer{T},i::Integer) where T
    op = Matrix(b)
    return size(op,i) ÷ size(T,i)
end

function _blockindex2index(T::Type{<:SMatrix},i::Integer,j::Integer)
    N,M = size(T)
    irange = ((i-1)*N+1):(i*N)
    jrange = ((j-1)*M+1):(j*M)
    return irange,jrange
end
function _blockindex2index(::BlockIndexer{T},i::Integer,j::Integer) where T
    return _blockindex2index(T,i,j)
end

function Base.view(b::BlockIndexer,i::Integer,j::Integer)
    op = Matrix(b)
    irange,jrange = _blockindex2index(b,i,j)
    return view(op,irange,jrange)
end

function Base.getindex(b::BlockIndexer{T},i::Integer,j::Integer)::T where T
    return T(view(b,i,j))
end

function Base.setindex!(b::BlockIndexer,x,i::Integer,j::Integer)
    op = Matrix(b)
    irange,jrange = _blockindex2index(b,i,j)
    return op[irange,jrange] = x
end

"""
    MatrixAndBlockIndexer(T::Type{<:SMatrix},n::Integer,m::Integer)
    MatrixAndBlockIndexer(T::Type{<:Number},n::Integer,m::Integer)

Construct an uninitialized `A::Matrix{eltype(T)}` of size `(size(T,1)*n)×(size(T,2)*m)` and
its respective `Ablock::BlockIndexer` with blocks of type `T`. If `T::Type{<:Number}`, then
`Ablock===A`.
"""
function MatrixAndBlockIndexer(T::Type{<:SMatrix},n::Integer,m::Integer)
    A = Matrix{eltype(T)}(undef,size(T,1)*n,size(T,2)*m)
    Ablock = BlockIndexer(A,T)
    return A,Ablock
end
function MatrixAndBlockIndexer(T::Type{<:Number},n::Integer,m::Integer)
    A = Matrix{T}(undef,n,m)
    return A,A
end

"""
    struct BlockSparseConstructor{T<:Union{SMatrix,Number},S<:Number}

Convenient structure used to incrementally build a `SparseMatrixCSC{S}` from "block"-indices
and blocks of type `T` with `eltype(T)==S`. First, create a `b::BlockSparseConstructor` and add entries 
using the `addentry!(b::BlockSparseConstructor{T},i::Integer,j::Integer,v::T)` or `addentries!` functions.
Then, convert it into a sparse matrix with `sparse(b)`.
"""
struct BlockSparseConstructor{T<:Union{SMatrix,Number},S<:Number}
    size::Tuple{Int,Int}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{S}
    function BlockSparseConstructor{T,S}(n,m) where {T,S}
        @assert (isconcretetype(T)&&isconcretetype(S)) "element types must be concrete"
        @assert (S===eltype(T)) "element type of block `T` and scalar type `S` are not equal"
        I = Int[]
        J = Int[]
        V = S[]
        return new((n,m),I,J,V)
    end
end
BlockSparseConstructor(T,n,m) = BlockSparseConstructor{T,eltype(T)}(n,m)

Base.size(b::BlockSparseConstructor)   = b.size
Base.size(b::BlockSparseConstructor,i) = size(b)[i]

Base.eltype(::BlockSparseConstructor{T}) where T = T

SparseArrays.sparse(b::BlockSparseConstructor{T}) where {T<:SMatrix} = sparse(b.I,b.J,b.V,(size(b).*size(T))...)
SparseArrays.sparse(b::BlockSparseConstructor{T}) where {T<:Number}  = sparse(b.I,b.J,b.V,size(b)...)

addentry!(::BlockSparseConstructor,i,j,v) = error()
function addentry!(b::BlockSparseConstructor{T},i::Integer,j::Integer,v::T) where {T<:SMatrix}
    append!(b.V, v)
    irange,jrange = _blockindex2index(T,i,j)
    cart_ind = CartesianIndices((irange,jrange))
    for index in cart_ind
        i,j = Tuple(index)
        push!(b.I, i)
        push!(b.J, j)
    end
end
function addentry!(b::BlockSparseConstructor,i::Integer,j::Integer,v::Number)
    push!(b.I, i)
    push!(b.J, j)
    push!(b.V, v)
    return nothing
end
function addentries!(b::BlockSparseConstructor,I,J,V::AbstractVector)
    @assert length(I) == length(J) == length(V)
    for (i,j,v) in zip(I,J,V)
        addentry!(b,i,j,v)
    end
end
