
"""
    pseudoblockmatrix(blocktype::Type{<:SMatrix}, n, m)

Returns an uninitialized PseudoBlockMatrix of size (`n` blocks) × (`m` blocks) with
blocks of type `blocktype`.
"""
function pseudoblockmatrix(blocktype::Type{<:SMatrix}, n, m)
    T = eltype(blocktype)
    isize, jsize = size(blocktype)
    blocksize_i = [isize for _ in 1:n]
    blocksize_j = [jsize for _ in 1:m]
    return PseudoBlockMatrix{T}(undef, blocksize_i, blocksize_j)
end

"""
    pseudoblockmatrix(matrix::Matrix{<:Number}, blocktype::Type{<:SMatrix})

Wraps a `matrix::Matrix{<:Number}` into a PseudoBlockMatrix with blocks of type `blocktype`.
The sizes must be compatible. This operation does not allocate a new matrix.
"""
function pseudoblockmatrix(matrix::Matrix{<:Number}, blocktype::Type{<:SMatrix})
    T = eltype(blocktype)
    @assert eltype(matrix) == T
    (n, nrem), (m, mrem) = divrem.(size(matrix), size(blocktype))
    @assert nrem == 0 && mrem == 0  # sizes are compatible
    isize, jsize = size(blocktype)
    blocksize_i = [isize for _ in 1:n]
    blocksize_j = [jsize for _ in 1:m]
    return PseudoBlockMatrix(matrix, blocksize_i, blocksize_j)
end

"""
    pseudoblockmatrix(op::IntegralOperator{T}) where T

Converts `op::IntegralOperator{T}` into `psmatrix::PseudoBlockMatrix` with blocks 
of type `T`. Threads are used to speed up computations.
"""
function pseudoblockmatrix(op::IntegralOperator{T}) where T
    imax, jmax = size(op)
    psmatrix = pseudoblockmatrix(T, imax, jmax)
    for j in 1:jmax
        for i in 1:imax
            @inbounds psmatrix[Block(i, j)] = op[i, j]
        end
    end
    return psmatrix
end

"""
    to_matrix(p::PseudoBlockMatrix)

Returns the underlying matrix of `p::PseudoBlockMatrix`. This operation 
does not allocate a new matrix.
"""
function to_matrix(p::PseudoBlockMatrix)
    return p.blocks
end

"""
    to_matrix(op::IntegralOperator) 

Converts `op::IntegralOperator` into a contiguous `Matrix{<:Number}`
"""
function to_matrix(op::IntegralOperator) 
    p = pseudoblockmatrix(op)
    return to_matrix(p)
end

