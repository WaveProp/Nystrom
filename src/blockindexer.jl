
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
    blocksparse(I, J, V::Vector{T}, n::Integer,m::Integer) where {T<:SMatrix}
    blocksparse(I, J, V::Vector{<:Number}, n::Integer,m::Integer)

Similar to `SparseArrays.sparse`, but `V` is a vector of `SMatrix` blocks and both I` and `J`
are vectors of indices in "block"-coordinates. A `SparseMatrixCSC{<:Scalar}` is returned.
"""
function blocksparse(I, J, V::Vector{T}, n::Integer,m::Integer) where {T<:SMatrix}
    S = eltype(T)
    nmatrix,mmatrix = (n,m) .* size(T)
    Iscalar = Int[]
    Jscalar = Int[]
    Vscalar = S[]
    for (i,j,v) in zip(I,J,V)
        append!(Vscalar,v)
        irange,jrange = _blockindex2index(T,i,j)
        cart_ind = Tuple.(CartesianIndices((irange,jrange)))
        for (i,j) in cart_ind
            push!(Iscalar,i)
            push!(Jscalar,j)
        end
    end
    return sparse(Iscalar,Jscalar,Vscalar,nmatrix,mmatrix)
end
function blocksparse(I, J, V::Vector{<:Number}, n::Integer,m::Integer)
    return sparse(I,J,V,n,m)
end
