function assemble_dim(iop::IntegralOperator;compress=Matrix,location=:onsurface)
    X    = target_surface(iop)
    Y    = source_surface(iop)
    pde  = iop.kernel.pde
    T    = kernel_type(iop)
    if T === SingleLayer()
        singlelayer_dim(pde,X,Y;compress,location)
    elseif T === DoubleLayer()
        doublelayer_dim(pde,X,Y;compress,location)
    elseif T === AdjointDoubleLayer()
        adjointdoublelayer_dim(pde,X,Y;compress,location)
    elseif T === HyperSingular()
        hypersingular_dim(pde,X,Y;compress,location)
    else
        notimplemented()
    end
end

function single_doublelayer_dim(pde,X,Y=X;compress=Matrix,location=:onsurface)
    msg = "unrecognized value for kw `location`: received $location.
           Valid options are `:onsurface`, `:inside` and `:outside`."
    σ = location === :onsurface ? -0.5 : location === :inside ? 0 : location === :outside ? -1 : error(msg)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    @timeit_debug "assemble dense part" begin
        S = compress(Sop)
        D = compress(Dop)
    end
    @timeit_debug "compute near interaction list" begin
        dict_near = near_interaction_list(dofs(X),Y;atol=0)
    end
    # precompute dim quantities
    @timeit_debug "auxiliary dim quantities" begin
        basis,γ₁_basis = _basis_dim(Sop)  # list of functions γ₀ and γ₁ for each source
        γ₀B,γ₁B,R      = _auxiliary_quantities_dim(Sop,S,D,basis,γ₁_basis,σ)
    end
    # compute corrections
    @timeit_debug "compute dim correction" begin
        if pde isa Maxwell
            δS = _singular_weights_dim_maxwell(Sop,γ₀B,γ₁B,R,dict_near)
            δD = _singular_weights_dim_maxwell(Dop,γ₀B,γ₁B,R,dict_near)
        else
            δS = _singular_weights_dim(Sop,γ₀B,γ₁B,R,dict_near)
            δD = _singular_weights_dim(Dop,γ₀B,γ₁B,R,dict_near)
        end
    end
    # convert to DiscreteOp
    S_discrete = DiscreteOp(S) + DiscreteOp(δS)
    D_discrete = DiscreteOp(D) + DiscreteOp(δD)
    return S_discrete,D_discrete
end
singlelayer_dim(args...;kwargs...) = single_doublelayer_dim(args...;kwargs...)[1]
doublelayer_dim(args...;kwargs...) = single_doublelayer_dim(args...;kwargs...)[2]

function adjointdoublelayer_hypersingular_dim(pde,X,Y=X;compress=Matrix,location=:onsurface)
    msg = "unrecognized value for kw `location`: received $location.
    Valid options are `:onsurface`, `:inside` and `:outside`."
    σ = location === :onsurface ? -0.5 : location === :inside ? 0 : location === :outside ? -1 : error(msg)
    Kop  = AdjointDoubleLayerOperator(pde,X,Y)
    Hop  = HyperSingularOperator(pde,X,Y)
    # convert to a possibly more efficient compress
    K = compress(Kop)
    H = compress(Hop)
    dict_near = near_interaction_list(dofs(X),Y;atol=0)
    # precompute dim quantities
    basis,γ₁_basis = _basis_dim(Kop)
    γ₀B,γ₁B,R      = _auxiliary_quantities_dim(Kop,K,H,basis,γ₁_basis,σ)
    # compute corrections
    δK        = _singular_weights_dim(Kop,γ₀B,γ₁B,R,dict_near)
    δH        = _singular_weights_dim(Hop,γ₀B,γ₁B,R,dict_near)
    # convert to DiscreteOp
    K_discrete = DiscreteOp(K) + DiscreteOp(δK)
    H_discrete = DiscreteOp(H) + DiscreteOp(δH)
    return K_discrete,H_discrete
end
adjointdoublelayer_dim(args...;kwargs...)  = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[1]
hypersingular_dim(args...;kwargs...)       = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[2]


function singular_weights_dim(iop::IntegralOperator,compress=Matrix)
    X,Y,op = iop.X, iop.Y, iop.kernel.op
    σ = X == Y ? -0.5 : 0.0
    #
    dict_near = near_interaction_list(dofs(X),Y;atol=0)
    basis,γ₁_basis = _basis_dim(iop)
    Op1, Op2       = _auxiliary_operators_dim(iop,compress)
    γ₀B,γ₁B,R      = _auxiliary_quantities_dim(iop,Op1,Op2,basis,γ₁_basis,σ)
    Sp = _singular_weights_dim(iop,γ₀B,γ₁B,R,dict_near)
    return Sp # a sparse matrix
end

function _auxiliary_quantities_dim(iop,Op0,Op1,basis,γ₁_basis,σ)
    T      = eltype(iop)
    X,Y    = target_surface(iop), source_surface(iop)
    num_basis = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes   = dofs(Y)
    γ₀B      = Matrix{T}(undef,length(ynodes),num_basis)
    γ₁B      = Matrix{T}(undef,length(ynodes),num_basis)
    for k in 1:num_basis
        for i in 1:length(ynodes)
            γ₀B[i,k] = basis[k](ynodes[i])
            γ₁B[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    xnodes      = dofs(X)
    num_targets = length(xnodes)
    num_sources = length(ynodes)
    R = Matrix{T}(undef,num_targets,num_basis)
    @timeit_debug "integrate dim basis" begin
        @threads for k in 1:num_basis
            # integrate basis over the surface
            # FIXME: use inplace multiplication through `mul!` instead of the line
            # below. The issue at the moment is that
            R[:,k] = Op0*γ₁B[:,k] - Op1*γ₀B[:,k]
            # @views mul!(R[:,k],Op0,γ₁B[:,k])
            # @views mul!(R[:,k:k],Op1,γ₀B[:,k:k],-1,1)
            for i in 1:num_targets
                # analytic correction for on-surface evaluation of Greens identity
                if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
                    R[i,k] += σ*basis[k](xnodes[i])
                elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
                    R[i,k] += σ*γ₁_basis[k](xnodes[i])
                end
            end
        end
    end
    return γ₀B, γ₁B, R
end

function _auxiliary_operators_dim(iop,compress)
    X,Y,op = iop.X, iop.Y, iop.kernel.op
    T = eltype(iop)
    # construct integral operators required for correction
    if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
        Op1 = IntegralOperator{T}(SingleLayerKernel(op),X,Y) |> compress
        Op2 = IntegralOperator{T}(DoubleLayerKernel(op),X,Y) |> compress
    elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
        Op1 = IntegralOperator{T}(AdjointDoubleLayerKernel(op),X,Y) |> compress
        Op2 = IntegralOperator{T}(HyperSingularKernel(op),X,Y) |> compress
    end
    return Op1,Op2
end

function _basis_dim(iop)
    op = pde(kernel(iop))
    xs = _source_gen(iop) # list of Lebedev sources
    basis     = [(source) -> SingleLayerKernel(op)(x,source) for x in xs]
    γ₁_basis  = [(source) -> transpose(DoubleLayerKernel(op)(x,source)) for x in xs]
    return basis,γ₁_basis
end

function _singular_weights_dim(iop::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    # initialize vectors for the sparse matrix, then dispatch to type-stable
    # method for each element type
    T   = eltype(iop)
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for (E,list_near) in dict_near
        _singular_weights_dim!(Is,Js,Vs,iop,γ₀B,γ₁B,R,E,list_near)
    end
    Sp = sparse(Is,Js,Vs,size(iop)...)
    return Sp
end

@noinline function _singular_weights_dim!(Is,Js,Vs,iop,γ₀B,γ₁B,R,E,list_near)
    X,Y = target_surface(iop), source_surface(iop)
    T   = eltype(iop)
    num_basis = size(γ₀B,2)
    a,b = combined_field_coefficients(iop)
    el2qnodes = elt2dof(Y,E)
    num_qnodes, num_els   = size(el2qnodes)
    M                     = Matrix{T}(undef,2*num_qnodes,num_basis)
    @assert length(list_near) == num_els
    for n in 1:num_els
        j_glob                = @view el2qnodes[:,n]
        M[1:num_qnodes,:]     = @view γ₀B[j_glob,:]
        M[num_qnodes+1:end,:] = @view γ₁B[j_glob,:]
        # distinguish scalar and vectorial case
        if T <: Number
            F                     = qr(M)
        elseif T <: SMatrix
            M_mat = blockmatrix_to_matrix(M)
            F                     = qr!(M_mat)
        else
            error("unknown element type T=$T")
        end
        for (i,_) in list_near[n]
            if T <: Number
                tmp = ((R[i:i,:])/F.R)*adjoint(F.Q)
            elseif T <: SMatrix
                tmp_scalar  = (blockmatrix_to_matrix(R[i:i,:])/F.R)*adjoint(F.Q)
                tmp  = matrix_to_blockmatrix(tmp_scalar,T)
            else
                error("unknown element type T=$T")
            end
            w    = axpby!(a,view(tmp,1:num_qnodes),b,view(tmp,(num_qnodes+1):(2*num_qnodes)))
            append!(Is,fill(i,num_qnodes))
            append!(Js,j_glob)
            append!(Vs,w)
        end
    end
    return Is,Js,Vs
end

function _singular_weights_dim_maxwell(iop::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    # initialize vectors for the sparse matrix, then dispatch to type-stable
    # method for each element type
    T   = eltype(iop)
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for (E,list_near) in dict_near
        _singular_weights_dim_maxwell!(Is,Js,Vs,iop,γ₀B,γ₁B,R,E,list_near)
    end
    Sp = sparse(Is,Js,Vs,size(iop)...)
    return Sp
end

@noinline function _singular_weights_dim_maxwell!(Is,Js,Vs,iop,γ₀B,γ₁B,R,E,list_near)
    X,Y    = target_surface(iop), source_surface(iop)
    qnodes = dofs(Y)
    T   = eltype(iop)
    a,b = combined_field_coefficients(iop)
    el2qnodes = elt2dof(Y,E)
    num_qnodes, num_els   = size(el2qnodes)
    num_basis = size(γ₀B,2)
    T2 = SMatrix{2,3,ComplexF64,6}
    T3 = SMatrix{3,2,ComplexF64,6}
    M      = Matrix{T2}(undef,2*num_qnodes,num_basis)
    M_mat  = Matrix{ComplexF64}(undef,2*2*num_qnodes,3*num_basis)
    @assert length(list_near) == num_els
    for n in 1:num_els
        j_glob                = @view el2qnodes[:,n]
        for p in 1:num_basis
            for k in 1:num_qnodes
                # J,_ = jacobian(qnodes[j_glob[k]]) |> qr
                J = jacobian(qnodes[j_glob[k]])
                M[k,p]            = transpose(J)*γ₀B[j_glob[k],p]
                M[num_qnodes+k,p] = transpose(J)*γ₁B[j_glob[k],p]
            end
        end
        # M_mat                 = blockmatrix_to_matrix(M)
        blockmatrix_to_matrix!(M_mat,M)
        F                     = qr!(M_mat)
        for (i,_) in list_near[n]
            tmp_scalar  = (blockmatrix_to_matrix(R[i:i,:])/F.R)*adjoint(F.Q)
            tmp         = matrix_to_blockmatrix(tmp_scalar,T3)
            tmp         = axpby!(a,view(tmp,1:num_qnodes),b,view(tmp,(num_qnodes+1):(2*num_qnodes)))
            w           = Vector{T}(undef,num_qnodes)
            for k in 1:num_qnodes
                # J,_ = jacobian(qnodes[j_glob[k]]) |> qr
                J    = jacobian(qnodes[j_glob[k]])
                w[k] = tmp[k]*transpose(J)
            end
            append!(Is,fill(i,num_qnodes))
            append!(Js,j_glob)
            append!(Vs,w)
        end
    end
    return Is,Js,Vs
end

function _source_gen(iop::IntegralOperator,kfactor=5)
    Y      =  source_surface(iop)
    nquad  = 0
    for (E,tags) in elt2dof(Y)
        nquad = max(nquad,size(tags,1))
    end
    nbasis = 3*nquad
    # construct source basis
    return _source_gen(iop,nbasis;kfactor)
end

function _source_gen(iop,nsources;kfactor)
    N      = ambient_dimension(iop)
    Y      = source_surface(iop)
    pts    = dofs(Y)
    # create a bounding box
    bbox   = HyperRectangle(pts)
    xc     = center(bbox)
    d      = diameter(bbox)
    if N == 2
        xs = _circle_sources(;nsources,center=xc,radius=kfactor*d/2)
    elseif N == 3
        xs = _sphere_sources_lebedev(;nsources,center=xc,radius=kfactor*d/2)
    else
        error("dimension must be 2 or 3. Got $N")
    end
    return xs
end

function _sphere_sources_lebedev(;nsources, radius=10, center=SVector(0.,0.,0.))
    lpts = lebedev_points(nsources)
    Xs = SVector{3,Float64}[]
    for pt in lpts
        push!(Xs,radius*pt .+ center)
    end
    return Xs
end

function _circle_sources(;nsources, radius=10, center=SVector(0.,0.))
    par   = (s) -> center .+ radius .* SVector(cospi(2 * s[1]), sinpi(2 * s[1]))
    x     = [i/(nsources) for i in 0:nsources-1]
    Xs    = SVector{2,Float64}[]
    for pt in x
        push!(Xs,par(pt))
    end
    return Xs
end
