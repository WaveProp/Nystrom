function maxwell_dim(pde::Maxwell,X,Y=X;compress=Matrix,location=:onsurface)
    msg = "unrecognized value for kw `location`: received $location.
           Valid options are `:onsurface`, `:inside` and `:outside`."
    σ = location === :onsurface ? -0.5 : location === :inside ? 0 : location === :outside ? -1 : error(msg)
    Top = EFIEOperator(pde,X,Y)
    Kop = MFIEOperator(pde,X,Y)
    # convert to a possibly more efficient format
    T = compress(Top)
    K = compress(Kop)
    # compute near interaction list
    dict_near = Nystrom._near_interaction_list_dim(X,Y)
    # precompute dim quantities
    basis,γ₁_basis = _maxwell_basis_dim(Top)  # list of functions γ₀ and γ₁ for each source
    γ₀B,γ₁B,R      = _maxwell_auxiliary_quantities_dim(Top,T,K,basis,γ₁_basis,σ)
    # compute corrections
    δT,δK = _maxwell_singular_weights_dim(Top,γ₀B,γ₁B,R,dict_near)
    # convert to DiscreteOp
    T_discrete = Nystrom.DiscreteOp(T) + Nystrom.DiscreteOp(δT)
    K_discrete = Nystrom.DiscreteOp(K) + Nystrom.DiscreteOp(δK)
    return T_discrete,K_discrete
end
efie_dim(args...;kwargs...) = maxwell_dim(args...;kwargs...)[1]
mfie_dim(args...;kwargs...) = maxwell_dim(args...;kwargs...)[2]

function _maxwell_basis_dim(iop)
    op = Nystrom.pde(Nystrom.kernel(iop))
    xs = Nystrom._source_gen(iop) # list of Lebedev sources
    basis     = [(qnode) -> EFIEKernel(op)(qnode,src) for src in xs]
    γ₁_basis  = [(qnode) -> MFIEKernel(op)(qnode,src) for src in xs]
    return basis,γ₁_basis
end

function _maxwell_auxiliary_quantities_dim(iop,T,K,basis,γ₁_basis,σ)
    V   = eltype(iop)
    X,Y = Nystrom.target_surface(iop), Nystrom.source_surface(iop)
    num_basis = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes = Nystrom.dofs(Y)
    γ₀B,γ₀B_block = Nystrom.MatrixAndBlockIndexer(V,length(ynodes),num_basis)
    γ₁B,γ₁B_block = Nystrom.MatrixAndBlockIndexer(V,length(ynodes),num_basis)
    Threads.@threads for i in 1:length(ynodes)
        for k in 1:num_basis
            γ₀B_block[i,k] = basis[k](ynodes[i])
            γ₁B_block[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    xnodes      = Nystrom.dofs(X)
    num_targets = length(xnodes)
    R,R_block = Nystrom.MatrixAndBlockIndexer(V,num_targets,num_basis)
    # R .= -T*γ₁B - K*γ₀B
    mul!(R,T,γ₁B,-1,false)
    mul!(R,K,γ₀B,-1,1)
    if X === Y
        R += σ*γ₀B
    else
        Threads.@threads for k in 1:num_basis
            for i in 1:num_targets
                # analytic correction for on-surface evaluation of Stratton-Chu identity
                R_block[i,k] += σ*basis[k](xnodes[i])
            end
        end
    end
    return γ₀B, γ₁B, R
end

function _maxwell_singular_weights_dim(Top,γ₀B,γ₁B,R,dict_near)
    # initialize vectors for the sparse matrix, then dispatch to type-stable
    # method for each element type
    Y  = Nystrom.source_surface(Top)
    V  = eltype(Top)
    sizeop = size(Top)
    # for EFIE
    δTblock = Nystrom.BlockSparseConstructor(V,sizeop...)
    # for MFIE
    δKblock = Nystrom.BlockSparseConstructor(V,sizeop...)
    for (E,list_near) in dict_near
        _maxwell_singular_weights_dim!(δTblock,δKblock,Y,γ₀B,γ₁B,R,E,list_near)
    end
    # convert to SparseMatrixCSC{<:Number}
    δT = sparse(δTblock)
    δK = sparse(δKblock)
    return δT, δK
end

@noinline function _maxwell_singular_weights_dim!(δTblock,δKblock,Y,γ₀B,γ₁B,R,E,list_near)
    qnodes = Nystrom.dofs(Y)
    V = eltype(δTblock)  # block type
    V2 = SMatrix{2,3,ComplexF64,6}
    V3 = SMatrix{3,2,ComplexF64,6}
    el2qnodes = Nystrom.elt2dof(Y,E)
    num_qnodes, num_els = size(el2qnodes)
    γ₀B_block = Nystrom.BlockIndexer(γ₀B,V)
    γ₁B_block = Nystrom.BlockIndexer(γ₁B,V)
    R_block   = Nystrom.BlockIndexer(R,V)
    num_basis = size(γ₀B_block,2)
    M,Mblock  = Nystrom.MatrixAndBlockIndexer(V2,2*num_qnodes,num_basis)
    H,Hblock  = Nystrom.MatrixAndBlockIndexer(V,1,num_basis)
    G,Gblock  = Nystrom.MatrixAndBlockIndexer(V3,1,2*num_qnodes)
    Ts = Vector{V}(undef,num_qnodes)  # for EFIE
    Ks = Vector{V}(undef,num_qnodes)  # for MFIE
    @assert length(list_near) == num_els
    for n in 1:num_els
        j_glob = @view el2qnodes[:,n]
        for p in 1:num_basis
            for k in 1:num_qnodes
                J = Nystrom.jacobian(qnodes[j_glob[k]])
                Mblock[k,p]            = transpose(J)*γ₀B_block[j_glob[k],p]
                Mblock[num_qnodes+k,p] = transpose(J)*γ₁B_block[j_glob[k],p]
            end
        end
        F = qr!(M)
        for i in list_near[n]
            Hblock[:,:] = @view R_block[i:i,:]
            G[:,:] = (H/F.R)*adjoint(F.Q)
            for k in 1:num_qnodes
                J     = Nystrom.jacobian(qnodes[j_glob[k]])
                Ks[k] = Gblock[k]*transpose(J)
                Ts[k] = Gblock[k+num_qnodes]*transpose(J)
            end
            # add entries to BlockSparseConstructor
            Is = fill(i,num_qnodes)
            Js = j_glob
            Nystrom.addentries!(δTblock,Is,Js,Ts)
            Nystrom.addentries!(δKblock,Is,Js,Ks)
        end
    end
end