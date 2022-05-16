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

function blockdiag_preconditioner(dop::Nystrom.LinearCombinationDiscreteOp)
    # FIXME: too hacky
    # dop.maps[1]: uncorrected integral operator
    # dop.maps[2]: sparse correction
    @assert length(dop.maps) == 2
    C₀ = Nystrom.materialize(dop.maps[1])  # uncorrected integral operator
    C₁ = Nystrom.materialize(dop.maps[2])  # sparse correction
    # TODO: handle the case when C₀ is not a matrix
    @assert C₀ isa Matrix{<:Number}
    @assert C₁ isa SparseMatrixCSC{<:Number}

    # the sparse correction needs to be corrected
    # with the integral operator
    blockdiag = deepcopy(C₁)    # block diagonal preconditioner
    rows = rowvals(blockdiag)
    vals = nonzeros(blockdiag)
    ncols = size(blockdiag,2)   # number of columns
    Threads.@threads for j in 1:ncols
        for index in nzrange(blockdiag, j)
            i = rows[index]  # row index
            # update value in blockdiag
            vals[index] += C₀[i,j]
        end
    end
    return blockdiag
end