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
    σ = location === :onsurface ? -0.5 : location === :inside ? -1 : location === :outside ? 0 : error(msg)
    Sop  = SingleLayerOperator(pde,X,Y)
    Dop  = DoubleLayerOperator(pde,X,Y)
    # convert to a possibly more efficient format
    @timeit_debug "assemble dense part" begin
        S = compress(Sop)
        D = compress(Dop)
    end
    @timeit_debug "compute near interaction list" begin
        dict_near = _near_interaction_list_dim(X,Y)
    end
    # precompute dim quantities
    @timeit_debug "auxiliary dim quantities" begin
        basis,γ₁_basis = _basis_dim(Sop)  # list of functions γ₀ and γ₁ for each source
        γ₀B,γ₁B,R      = _auxiliary_quantities_dim(Sop,S,D,basis,γ₁_basis,σ)
    end
    # compute corrections
    @timeit_debug "compute dim correction" begin
        δS,δD = _singular_weights_dim(Sop,Dop,γ₀B,γ₁B,R,dict_near)
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
    σ = location === :onsurface ? -0.5 : location === :inside ? -1 : location === :outside ? 0 : error(msg)
    Kop  = AdjointDoubleLayerOperator(pde,X,Y)
    Hop  = HyperSingularOperator(pde,X,Y)
    # convert to a possibly more efficient compress
    K = compress(Kop)
    H = compress(Hop)
    dict_near = _near_interaction_list_dim(X,Y)
    # precompute dim quantities
    basis,γ₁_basis = _basis_dim(Kop)
    γ₀B,γ₁B,R      = _auxiliary_quantities_dim(Kop,K,H,basis,γ₁_basis,σ)
    # compute corrections
    δK,δH        = _singular_weights_dim(Kop,Hop,γ₀B,γ₁B,R,dict_near)
    # convert to DiscreteOp
    K_discrete = DiscreteOp(K) + DiscreteOp(δK)
    H_discrete = DiscreteOp(H) + DiscreteOp(δH)
    return K_discrete,H_discrete
end
adjointdoublelayer_dim(args...;kwargs...)  = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[1]
hypersingular_dim(args...;kwargs...)       = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[2]

function _auxiliary_quantities_dim(iop,Op0,Op1,basis,γ₁_basis,σ)
    T   = eltype(iop)
    X,Y = target_surface(iop), source_surface(iop)
    num_basis = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes = dofs(Y)
    γ₀B,γ₀B_block = MatrixAndBlockIndexer(T,length(ynodes),num_basis)
    γ₁B,γ₁B_block = MatrixAndBlockIndexer(T,length(ynodes),num_basis)
    @threads for i in 1:length(ynodes)
        for k in 1:num_basis
            γ₀B_block[i,k] = basis[k](ynodes[i])
            γ₁B_block[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    xnodes      = dofs(X)
    num_targets = length(xnodes)
    R,R_block = MatrixAndBlockIndexer(T,num_targets,num_basis)
    @timeit_debug "integrate dim basis" begin
        # R .= Op0*γ₁B - Op1*γ₀B
        mul!(R,Op0,γ₁B)
        mul!(R,Op1,γ₀B,-1,1)
        @threads for k in 1:num_basis
            for i in 1:num_targets
                # analytic correction for on-surface evaluation of Greens identity
                if kernel_type(iop) isa Union{SingleLayer,DoubleLayer}
                    R_block[i,k] += σ*basis[k](xnodes[i])
                elseif kernel_type(iop) isa Union{AdjointDoubleLayer,HyperSingular}
                    R_block[i,k] += σ*γ₁_basis[k](xnodes[i])
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

"""
    _near_interaction_list_dim(iop,location)

For each element `el` in the `source_surface` of `iop`, return the indices of
the nodes in the `target_surface` for which `el` is the closest element. This
map gives the indices in `iop` that need to be corrected by the `dim`
quadrature when integrating over `el`.
"""
function _near_interaction_list_dim(X,Y)
    if X === Y
        dict = Dict{DataType,Vector{Vector{Int}}}()
        # when both surfaces are the same, the "near points" of an element are
        # simply its own quadrature points
        for (E,idx_dofs) in elt2dof(Y)
            dict[E] = map(i->collect(i),eachcol(idx_dofs))
        end
    else
        # FIXME: this case is a lot less common, but deserves some attention.
        # The idea is that we should find, for each target point, the closest
        # that is less than a tolerance `atol` (there may be none). We then
        # revert this map to build a map where for each element in `Y`, we store
        # a vector of the target indices for which that element is the closest
        # since this is what is needed by the `dim` methods as implemented here.
        dict = Dict{DataType,Vector{Vector{Int}}}()
        hmax = -Inf # minimum average distance between quadrature nodes
        for (E,idx_dofs) in elt2dof(Y)
            nq,ne   = size(idx_dofs)
            for idxs in eachcol(idx_dofs)
                h = sum(i->weight(Y.dofs[i]),idxs) / length(idxs)
                hmax = max(h,hmax)
            end
            dict[E] = [Int[] for _ in 1:ne]
        end
        target2source = nearest_neighbor(dofs(X),dofs(Y),10*hmax)
        source2el     = dof2el(Y)
        for (i,s) in enumerate(target2source)
            s == -1 && continue
            E,n = source2el[s]
            push!(dict[E][n],i)
        end
    end
    return dict
end

function nearest_neighbor(X,Y,dmax=Inf)
    m,n = length(X), length(Y)
    out = Vector{Int}(undef,m)
    for i in 1:m
        dist = Inf
        jmin   = -1
        for j in 1:n
            d = norm(coords(X[i])-coords(Y[j]))
            d > dmax && continue
            if d < dist
                dist = d
                jmin = j
            end
        end
        out[i] = jmin
    end
    return out
end

function _basis_dim(iop)
    op = pde(kernel(iop))
    xs = _source_gen(iop) # list of Lebedev sources
    basis     = [(source) -> SingleLayerKernel(op)(x,source) for x in xs]
    γ₁_basis  = [(source) -> transpose(DoubleLayerKernel(op)(x,source)) for x in xs]
    return basis,γ₁_basis
end

function _singular_weights_dim(op1::IntegralOperator,op2::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    # initialize vectors for the sparse matrix, then dispatch to type-stable
    # method for each element type
    @assert (kernel(op1) isa SingleLayerKernel && kernel(op2) isa DoubleLayerKernel) ||
            (kernel(op1) isa AdjointDoubleLayerKernel && kernel(op2) isa HyperSingularKernel)
    Y  = source_surface(op1)
    T  = eltype(op1)
    sizeop = size(op1)
    # for single layer / adjoint double layer
    δSblock = BlockSparseConstructor(T,sizeop...)
    _,Scoeff = combined_field_coefficients(op1)
    # for double layer / hypersingular
    δDblock = BlockSparseConstructor(T,sizeop...)
    Dcoeff,_ = combined_field_coefficients(op2)
    for (E,list_near) in dict_near
        _singular_weights_dim!(δSblock,δDblock,Scoeff,Dcoeff,Y,γ₀B,γ₁B,R,E,list_near)
    end
    # convert to SparseMatrixCSC{<:Number}
    δS = sparse(δSblock)
    δD = sparse(δDblock)
    return δS, δD
end

@noinline function _singular_weights_dim!(δSblock,δDblock,Scoeff,Dcoeff,Y,γ₀B,γ₁B,R,E,list_near)
    T = eltype(δSblock)  # block type
    el2qnodes = elt2dof(Y,E)
    num_qnodes, num_els = size(el2qnodes)
    γ₀B_block = BlockIndexer(γ₀B,T)
    γ₁B_block = BlockIndexer(γ₁B,T)
    R_block   = BlockIndexer(R,T)
    num_basis = size(γ₀B_block,2)
    M,Mblock  = MatrixAndBlockIndexer(T,2*num_qnodes,num_basis)
    H,Hblock  = MatrixAndBlockIndexer(T,1,num_basis)
    G,Gblock  = MatrixAndBlockIndexer(T,1,2*num_qnodes)
    @assert length(list_near) == num_els
    for n in 1:num_els
        # if there is nothing near, skip immediately to next element
        isempty(list_near[n]) && continue
        # there are entries to correct, so do some work
        j_glob                     = @view el2qnodes[:,n]
        Mblock[1:num_qnodes,:]     = @view γ₀B_block[j_glob,:]
        Mblock[num_qnodes+1:end,:] = @view γ₁B_block[j_glob,:]
        F = qr!(M)
        for i in list_near[n]
            Hblock[:,:] = @view R_block[i:i,:]
            G[:,:] = (H/F.R)*adjoint(F.Q)
            # add entries to BlockSparseConstructor
            Is = fill(i,num_qnodes)
            Js = j_glob
            Ss = Scoeff * view(Gblock,(num_qnodes+1):(2*num_qnodes))
            Ds = Dcoeff * view(Gblock,1:num_qnodes)
            addentries!(δSblock,Is,Js,Ss)
            addentries!(δDblock,Is,Js,Ds)
        end
    end
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
