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
    # d.maps[2]: sparse correction
    @assert length(d.maps) == 2
    C₀ = Nystrom.materialize(d.maps[1])  # uncorrected integral operator
    C₁ = Nystrom.materialize(d.maps[2])  # sparse correction
    @assert C₁ isa SparseMatrixCSC{<:Number}
    bd = _blockdiag(C₀,C₁;iop)
    return bd
end
function _blockdiag(C₀::Matrix{<:Number},C₁;iop)
    # the sparse correction needs to be corrected
    # with the uncorrected integral operator `C₀`
    bd = deepcopy(C₁)    # block diagonal preconditioner
    rows = rowvals(bd)
    vals = nonzeros(bd)
    ncols = size(bd,2)   # number of columns
    Threads.@threads for j in 1:ncols
        for index in nzrange(bd, j)
            i = rows[index]  # row index
            # update value in blockdiag
            vals[index] += C₀[i,j]
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
