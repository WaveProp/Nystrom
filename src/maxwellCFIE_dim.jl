function single_doublelayer_dim(pde::MaxwellCFIE,X,Y=X;n_src,compress=pseudoblockmatrix)
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
    corrections = _singular_weights_dim_maxwellCFIE(Sop,γ₀B,γ₁B,R,dict_near)
    # add corrections to the dense part
    _add_corrections_maxwellcfie!(S, D, corrections)
    return S, D
end
function _add_corrections_maxwellcfie!(S, D, corrections)
    Is, Js, Ss, Ds = corrections
    Threads.@threads for n in 1:length(Is)
        i = Is[n]
        j = Js[n] 
        vs = Ss[n] 
        vd = Ds[n]
        S[Block(i,j)] += vs
        D[Block(i,j)] += vd
    end
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
    γ₀B = pseudoblockmatrix(T, length(ynodes), n_src)
    γ₁B = pseudoblockmatrix(T, length(ynodes), n_src)
    for k in 1:n_src
        for i in 1:length(ynodes)
            γ₀B[Block(i,k)] = basis[k](ynodes[i])
            γ₁B[Block(i,k)] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    R = -0.5*γ₀B - D*γ₀B - S*γ₁B
    return γ₀B, γ₁B, R
end

function _singular_weights_dim_maxwellCFIE(iop::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    X,Y = target_surface(iop), source_surface(iop)
    T   = eltype(iop)
    @assert all(m isa PseudoBlockArray for m in (γ₀B,γ₁B,R))
    n_src = blocksize(γ₀B,2)
    Is = Int[]
    Js = Int[]
    Ss = T[]   # for single layer
    Ds = T[]   # for double layer
    for (E,list_near) in dict_near
        el2qnodes = elt2dof(Y,E)
        num_qnodes, num_els   = size(el2qnodes)
        M                     = pseudoblockmatrix(T, 2*num_qnodes,n_src)
        @assert length(list_near) == num_els
        for n in 1:num_els
            j_glob                = @view el2qnodes[:,n]
            _assemble_interpolant_matrix!(M, γ₀B, γ₁B, j_glob, num_qnodes, n_src)  # assemble M
            F = pinv(to_matrix(M))  # FIXME: use LQ decomp. instead (M must be full rank)
            for (i,_) in list_near[n]
                tmp_scalar = to_matrix(R[Block(i),Block.(1:n_src)]) * F
                tmp = pseudoblockmatrix(tmp_scalar, T)  # blocksize(tmp) = (1,2*num_qnodes)
                Dw = view(tmp,Block(1),Block.(1:num_qnodes))
                Sw = view(tmp,Block(1),Block.((num_qnodes+1):(2*num_qnodes)))
                #@info "" blocksize(Sw) size(Sw) blocksize(tmp) size(tmp) 
                append!(Is,fill(i,num_qnodes))
                append!(Js,j_glob)
                for l in 1:num_qnodes
                    #@info "" blocksize(view(Sw, Block(1,l))) size(view(Sw, Block(1,l)))
                    push!(Ss,view(Sw, Block(1,l)))
                    push!(Ds,view(Dw, Block(1,l)))
                end
            end
        end
    end
    return Is, Js, Ss, Ds
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
function assemble_dim_nystrom_matrix(mesh, α, β, D::T, S::T) where T<:PseudoBlockMatrix
    N, J, dualJ = ncross_and_jacobian_matrices(mesh)
    Jm = diagonalblockmatrix_to_matrix(J.diag)
    dualJm = diagonalblockmatrix_to_matrix(dualJ.diag)
    Nm = diagonalblockmatrix_to_matrix(N.diag)
    Sm = to_matrix(S)
    Dm = to_matrix(D)
    n_qnodes = length(dofs(mesh))
    M = Matrix{ComplexF64}(undef, 2*n_qnodes, 2*n_qnodes)
    M .= dualJm*(0.5*α*I + α*Dm + β*Sm*Nm)*Jm
    return M
end

function assemble_dim_nystrom_matrix_Luiz(mesh, α, β, D::T, S::T) where T<:PseudoBlockMatrix
    N, J, dualJ = ncross_and_jacobian_matrices(mesh)
    Jm = diagonalblockmatrix_to_matrix(J.diag)
    dualJm = diagonalblockmatrix_to_matrix(dualJ.diag)
    Nm = diagonalblockmatrix_to_matrix(N.diag)
    Sm = to_matrix(S)
    Dm = to_matrix(D)
    n_qnodes = length(dofs(mesh))
    M = Matrix{ComplexF64}(undef, 2*n_qnodes, 2*n_qnodes)
    M .= dualJm*(0.5*α*I + Nm*(-α*Dm + β*Sm)*Nm)*Jm
    return M
end

function solve_LU(A::Matrix{ComplexF64}, σ::AbstractVector{V}) where {V}
    Amat    = A
    σ_vec   = reinterpret(eltype(V),σ)
    vals_vec = Amat\σ_vec
    vals    = reinterpret(V,vals_vec) |> collect
    return vals
end

function solve_GMRES(A::Matrix{ComplexF64}, σ::AbstractVector{V}, args...; kwargs...) where {V}
    σ_vec   = reinterpret(eltype(V),σ)
    vals_vec = copy(σ_vec)
    gmres!(vals_vec, A, σ_vec, args...; kwargs...)
    vals = reinterpret(V,vals_vec) |> collect
    return vals
end