function diagonal_ncross_and_jacobian_matrices(nmesh)
    qnodes = Nystrom.dofs(nmesh)
    n_qnodes = length(qnodes)
    # construct diagonal matrices as sparse arrays using BlockSparseConstructor
    Tn = qnodes |> first |> Nystrom.normal |> Nystrom.cross_product_matrix |> typeof
    Tj = qnodes |> first |> Nystrom.jacobian |> typeof
    Td = SMatrix{2,3,Float64,6}  # TODO: remove harcoded type
    nblock = Nystrom.BlockSparseConstructor(Tn,n_qnodes,n_qnodes)
    jblock = Nystrom.BlockSparseConstructor(Tj,n_qnodes,n_qnodes)
    dblock = Nystrom.BlockSparseConstructor(Td,n_qnodes,n_qnodes)
    for i in 1:n_qnodes
        q = qnodes[i]
        n = Nystrom.cross_product_matrix(normal(q))
        j = Nystrom.jacobian(q)
        d = Td(pinv(j))
        Nystrom.addentry!(nblock,i,i,n)
        Nystrom.addentry!(jblock,i,i,j)
        Nystrom.addentry!(dblock,i,i,d)
    end
    return sparse(nblock), sparse(jblock), sparse(dblock)
end
diagonal_ncross_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[1]
diagonal_jacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[2]
diagonal_dualjacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[3]

"""
    blockdiag(d::Nystrom.LinearCombinationDiscreteOp;
              iop::Union{Nothing,MaxwellOperator}=nothing)

Returns the block-diagonal (as a `SparseMatrixCSC{<:Number}`) of
an integral operator `d` obtained with `maxwell_dim`. 
The respective `iop:MaxwellOperator` (`EFIEOperator` or `MFIEOperator`)
is required only if `maxwell_dim` was called with a `compress` argument 
(e.g. with `compress = ifgf_compressor(;order=(3,3,3))`).
"""
function blockdiag(d::Nystrom.LinearCombinationDiscreteOp;
                   iop::Union{Nothing,MaxwellOperator}=nothing)
    # Check that `d` has the following form:
    # d.maps[1]: uncorrected integral operator
    # d.maps[2]: DIM sparse correction
    @assert length(d.maps) == 2
    C₀,C₁ = d.maps
    @assert C₀ isa Nystrom.AbstractDiscreteOp
    @assert C₁ isa Nystrom.DiscreteOp{<:SparseMatrixCSC{<:Number}}
    bd = _blockdiag(C₀,C₁;iop)
    return bd
end
function _blockdiag(C₀,C₁;iop)
    # the DIM sparse correction needs to be corrected
    # with the integral operator `iop`
    @assert !isnothing(iop) "you must provide an `iop::MaxwellOperator`."
    C₁m::SparseMatrixCSC{<:Number} = Nystrom.materialize(C₁)
    nqnodes = size(C₁m,1) ÷ 3            # 3 scalar entries per qnode
    # compute size and number of blocks in the diagonal
    Nsize   = length(nzrange(C₁m,1)) ÷ 3  # 3 scalar entries per qnode
    Nblocks = nqnodes ÷ Nsize
    # construct correction
    T     = SMatrix{3,3,ComplexF64,9}
    pool  = [Nystrom.MatrixAndBlockIndexer(T,Nsize,Nsize) 
            for _ in 1:Threads.nthreads()]  # memory pool for threads
    bd    = deepcopy(C₁m)   # block diagonal
    vals  = nonzeros(bd)
    Threads.@threads for n in 0:(Nblocks-1)
        M,Mblock = pool[Threads.threadid()]
        # fill `Mblock` first, 
        # which is a matrix of `SMatrix{3,3,ComplexF64,9}`
        for j in 1:Nsize
            for i in 1:Nsize
                index_i = n*Nsize+i
                index_j = n*Nsize+j
                Mblock[i,j] = iop[index_i,index_j]
            end
        end
        # update values in `bd` using `M`,
        # which are matrices of `Complexf64`.
        # We take advantage of the sparse block-diagonal
        # structure of `bd`.
        j = 0
        for index_j in (3n*Nsize+1):(3*(n+1)*Nsize)  # 3 scalar entries per qnode
            j += 1
            i = 0
            for index in nzrange(bd,index_j)
                i += 1
                # update value in `bd`
                vals[index] += M[i,j]
            end
        end
    end
    return bd
end
function _blockdiag(C₀::Nystrom.DiscreteOp{<:Matrix{<:Number}},C₁;iop)
    # the DIM sparse correction needs to be corrected
    # with the uncorrected integral operator `C₀`
    C₀m::Matrix{<:Number}          = Nystrom.materialize(C₀)
    C₁m::SparseMatrixCSC{<:Number} = Nystrom.materialize(C₁)
    bd    = deepcopy(C₁m)    # block diagonal
    rows  = rowvals(bd)
    vals  = nonzeros(bd)
    ncols = size(bd,2)   # number of columns
    Threads.@threads for j in 1:ncols
        for index in nzrange(bd, j)
            i = rows[index]  # row index
            # update value in `bd`
            vals[index] += C₀m[i,j]
        end
    end
    return bd
end

###
# Helmholtz CFIE Regularizer
###

"""
    struct Scalar2VectorOp

Convenient structure used to wrap an operator `op`,
which normally multiplies `AbstractVector{<:Number}`, so that 
`d::Scalar2VectorOp(op;dofs_per_qnode::Int64)` can multiply
`x::AbstractVector{SVector{dofs_per_qnode,<:Number}}` by multiplying
`op` with each component of the `SVector`s.

# Example
```julia-repl
julia> n = 2
2

julia> dofs_per_qnode = 3
3

julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> x = [SVector(3.,-2.,9.),SVector(1.,0.,-5.)]
2-element Vector{SVector{3, Float64}}:
 [3.0, -2.0, 9.0]
 [1.0, 0.0, -5.0]

julia> As = MaxwellDIM.Scalar2VectorOp(A;dofs_per_qnode)
Nystrom.MaxwellDIM.Scalar2VectorOp([1 2; 3 4], 3)

julia> As*x ≈ A*x
true
```
"""
struct Scalar2VectorOp
    op
    dofs_per_qnode::Int64
    function Scalar2VectorOp(op;dofs_per_qnode)
        return new(op,dofs_per_qnode)
    end
end

function Base.:*(d::Scalar2VectorOp,x::AbstractVector{<:Number})
    @assert iszero(length(x)%d.dofs_per_qnode)
    xr = reshape(x,d.dofs_per_qnode,:) |> transpose
    yr = Nystrom._mymul(d.op,xr)
    y = reshape(transpose(yr),size(x))
    return y
end
function Base.:*(d::Scalar2VectorOp,x::AbstractVector{T}) where {T<:SVector}
    V = eltype(T)
    xv = reinterpret(V,x)
    yv = d*xv
    y = reinterpret(T,yv)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector{<:Number},
                            d::Scalar2VectorOp,
                            x::AbstractVector{<:Number},
                            a,b)
    @assert iszero(length(x)%d.dofs_per_qnode)
    xr = reshape(x,d.dofs_per_qnode,:) |> transpose
    yr = reshape(y,d.dofs_per_qnode,:) |> transpose
    Nystrom._mymul!(yr,d.op,xr,a,b)
    return y
end
function LinearAlgebra.mul!(y::AbstractVector{T},
                            d::Scalar2VectorOp,
                            x::AbstractVector{T},
                            a,b) where {T<:SVector}
    V = eltype(T)
    xv = reinterpret(V,x)
    yv = reinterpret(V,y)
    mul!(yv,d,xv,a,b)
    return y
end

function helmholtz_regularizer(pde::Maxwell, nmesh; δ=1/2, kwargs...)
    k = Nystrom.parameters(pde)
    k_helmholtz = im*k*δ
    pdeHelmholtz = Nystrom.Helmholtz(;dim=3,k=k_helmholtz)
    S,_ = Nystrom.single_doublelayer_dim(pdeHelmholtz,nmesh;kwargs...)
    # the regularizer should act on vector quantities instead of 
    # scalar quantities
    R = Scalar2VectorOp(S;dofs_per_qnode=3) # currents have 3 components (Jx,Jy,Jz) 
    return R
end
