function single_doublelayer_dim(pde::MaxwellCFIE,X,Y=X;n_src,compress=Matrix)
    Sop = SingleLayerOperator(pde,X,Y)
    Dop = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    S = compress(Sop)
    D = compress(Dop)
    dict_near = near_interaction_list(dofs(X),Y;atol=0)
    # precompute dim quantities    
    basis,γ₁_basis = _basis_dim_maxwellcfie(Sop, n_src)  # list of functions γ₀ and γ₁ for each source
    γ₀B, γ₁B, R = _auxiliary_quantities_dim_maxwellcfie(Sop,S,D,basis,γ₁_basis)
    # compute corrections
    δS, δD = _singular_weights_dim_maxwellCFIE(Sop,γ₀B,γ₁B,R,dict_near)
    # add corrections to the dense part
    axpy!(true,δS,S)  # S = S + δS
    axpy!(true,δD,D)  # D = D + δD
    return S, D
end

function _basis_dim_maxwellcfie(iop, n_src)
    op = pde(kernel(iop))
    src_list = _source_gen(iop,n_src;kfactor=5) # list of Lebedev sources
    basis = [(qnode) -> SingleLayerKernel(op)(qnode,src) for src in src_list]
    γ₁_basis = [(qnode) -> DoubleLayerKernel(op)(qnode,src) for src in src_list]
    return basis, γ₁_basis
end

function _auxiliary_quantities_dim_maxwellcfie(iop,S,D,basis,γ₁_basis)
    T = eltype(iop)
    X,Y = target_surface(iop), source_surface(iop)
    n_src = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes = dofs(Y)
    γ₀B = Matrix{T}(undef, length(ynodes), n_src)
    γ₁B = Matrix{T}(undef, length(ynodes), n_src)
    for k in 1:n_src
        for i in 1:length(ynodes)
            γ₀B[i,k] = basis[k](ynodes[i])
            γ₁B[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    R = -0.5*γ₀B - D*γ₀B - S*γ₁B
    return γ₀B, γ₁B, R
end

function _singular_weights_dim_maxwellCFIE(iop::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    X,Y = target_surface(iop), source_surface(iop)
    T   = eltype(iop)
    n_src = size(γ₀B,2)
    Is = Int[]
    Js = Int[]
    Ss = T[]   # for single layer
    Ds = T[]   # for double layer
    for (E,list_near) in dict_near
        el2qnodes = elt2dof(Y,E)
        num_qnodes, num_els   = size(el2qnodes)
        M                     = Matrix{T}(undef,2*num_qnodes,n_src)
        @assert length(list_near) == num_els
        for n in 1:num_els
            j_glob                = @view el2qnodes[:,n]
            M[1:num_qnodes,:]     = @view γ₀B[j_glob,:]
            M[num_qnodes+1:end,:] = @view γ₁B[j_glob,:]
            F = pinv(blockmatrix_to_matrix(M))  # FIXME: use LQ decomp. instead (M must be full rank)
            for (i,_) in list_near[n]
                tmp_scalar  = blockmatrix_to_matrix(R[i:i,:]) * F
                tmp = matrix_to_blockmatrix(tmp_scalar,T)
                Dw = view(tmp,1:num_qnodes)
                Sw = view(tmp,(num_qnodes+1):(2*num_qnodes))
                #w = axpby!(a,view(tmp,1:n_qnodes),b,view(tmp,(n_qnodes+1):(2*n_qnodes)))
                append!(Is,fill(i,num_qnodes))
                append!(Js,j_glob)
                append!(Ss,Sw)
                append!(Ds,Dw)
            end
        end
    end
    n_qnodes = size(γ₀B,1)
    Sp = sparse(Is,Js,Ss,n_qnodes,n_qnodes)
    Dp = sparse(Is,Js,Ds,n_qnodes,n_qnodes)
    return Sp, Dp
end

function ncross_and_jacobian_matrices(mesh)
    qnodes = dofs(mesh)
    nmatrix = Diagonal([cross_product_matrix(normal(q)) for q in qnodes])
    jmatrix = Diagonal([jacobian(q) for q in qnodes])
    dual_jmatrix = Diagonal([SMatrix{2,3,Float64,6}(pinv(jacobian(q))) for q in qnodes])
    return nmatrix, jmatrix, dual_jmatrix
end

function assemble_dim_nystrom_matrix(mesh, α, β, D, S; exterior=true)
    σ = exterior ? 0.5 : -0.5
    N, J, dualJ = ncross_and_jacobian_matrices(mesh)
    Jm = diagonalblockmatrix_to_matrix(J.diag)
    dualJm = diagonalblockmatrix_to_matrix(dualJ.diag)
    n_qnodes = length(dofs(mesh))
    M = Matrix{ComplexF64}(undef, 2*n_qnodes, 2*n_qnodes)
    M .= dualJm*blockmatrix_to_matrix(σ*α*I + α*D + β*S*N)*Jm
    return M
end