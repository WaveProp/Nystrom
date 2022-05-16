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

function helmholtz_regularizer(pde::Maxwell, nmesh; δ=1/2, kwargs...)
    k = Nystrom.parameters(pde)
    k_helmholtz = im*k*δ
    pdeHelmholtz = Nystrom.Helmholtz(;dim=3,k=k_helmholtz)
    S,_ = Nystrom.single_doublelayer_dim(pdeHelmholtz,nmesh;kwargs...)
    # TODO: implement efficient version without
    # assembling the full matrix
    Smat = Nystrom.materialize(S)
    T = Nystrom.default_kernel_eltype(pde)
    diagI = spdiagm([one(T) for _ in 1:size(Smat,1)])
    R = (Smat*diagI) |> Nystrom.blockmatrix_to_matrix |> Nystrom.DiscreteOp 
    return R
end
