function assemble_dim(iop::IntegralOperator;compress=Matrix,location=:onsurface)
    X    = target_surface(iop)
    Y    = source_surface(iop)
    pde  = iop.kernel.pde
    T    = kernel_type(iop)
    if T === SingleLayer()
        return singlelayer_dim(pde,X,Y;compress,location)
    elseif T === DoubleLayer()
        return doublelayer_dim(pde,X,Y;compress,location)
    elseif T === AdjointDoubleLayer()
        return adjointdoublelayer_dim(pde,X,Y;compress,location)
    elseif T === HyperSingular()
        return hypersingular_dim(pde,X,Y;compress,location)
    elseif T === CombinedField() 
        α,β = combined_field_coefficients(iop)
        return combinedfield_dim(pde,X,Y;α,β,compress,location)
    elseif T === AdjointCombinedField() 
        α,β = combined_field_coefficients(iop)
        return adjointcombinedfield_dim(pde,X,Y;α,β,compress,location)
    else
        notimplemented()
    end
end

function _single_doublelayer_dim(pde,X,Y=X;compress=Matrix,location=:onsurface)
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
    return S,D,δS,δD
end
function single_doublelayer_dim(args...;kwargs...)
    S,D,δS,δD = _single_doublelayer_dim(args...;kwargs...)
    # convert to DiscreteOp
    Sd = DiscreteOp(S) + DiscreteOp(δS)
    Dd = DiscreteOp(D) + DiscreteOp(δD)
    return Sd,Dd
end
singlelayer_dim(args...;kwargs...) = single_doublelayer_dim(args...;kwargs...)[1]
doublelayer_dim(args...;kwargs...) = single_doublelayer_dim(args...;kwargs...)[2]
function combinedfield_dim(pde,X,Y;α,β,compress,location)
    Cop = CombinedFieldOperator(pde,X,Y;α,β)
    # convert to a possibly more efficient format
    @timeit_debug "assemble combined field dense part" begin
        C = compress(Cop)
    end
    _,_,δS,δD = _single_doublelayer_dim(pde,X,Y;compress,location)
    # convert to DiscreteOp
    return DiscreteOp(C) + DiscreteOp(α*δD-β*δS)
end

function _adjointdoublelayer_hypersingular_dim(pde,X,Y=X;compress=Matrix,location=:onsurface)
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
    δK,δH = _singular_weights_dim(Kop,Hop,γ₀B,γ₁B,R,dict_near)
    return K,H,δK,δH
end
function adjointdoublelayer_hypersingular_dim(args...;kwargs...)
    K,H,δK,δH = _adjointdoublelayer_hypersingular_dim(args...;kwargs...)
    # convert to DiscreteOp
    Kd = DiscreteOp(K) + DiscreteOp(δK)
    Hd = DiscreteOp(H) + DiscreteOp(δH)
    return Kd,Hd
end
adjointdoublelayer_dim(args...;kwargs...)  = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[1]
hypersingular_dim(args...;kwargs...)       = adjointdoublelayer_hypersingular_dim(args...;kwargs...)[2]
function adjointcombinedfield_dim(pde,X,Y;α,β,compress,location)
    Cop = AdjointCombinedFieldOperator(pde,X,Y;α,β)
    # convert to a possibly more efficient format
    @timeit_debug "assemble adjoint combined field dense part" begin
        C = compress(Cop)
    end
    _,_,δK,δH = _adjointdoublelayer_hypersingular_dim(pde,X,Y;compress,location)
    # convert to DiscreteOp
    return DiscreteOp(C) + DiscreteOp(α*δH-β*δK)
end

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
    @assert (kernel_type(op1) isa SingleLayer && kernel_type(op2) isa DoubleLayer) ||
            (kernel_type(op1) isa AdjointDoubleLayer && kernel_type(op2) isa HyperSingular)
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
        if pde(kernel(op1)) isa Maxwell
            _singular_weights_dim_maxwell!(δSblock,δDblock,Scoeff,Dcoeff,Y,γ₀B,γ₁B,R,E,list_near)
        else
            _singular_weights_dim!(δSblock,δDblock,Scoeff,Dcoeff,Y,γ₀B,γ₁B,R,E,list_near)
        end
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

@noinline function _singular_weights_dim_maxwell!(δSblock,δDblock,Scoeff,Dcoeff,Y,γ₀B,γ₁B,R,E,list_near)
    qnodes = dofs(Y)
    T = eltype(δSblock)  # block type
    T2 = SMatrix{2,3,ComplexF64,6}
    T3 = SMatrix{3,2,ComplexF64,6}
    el2qnodes = elt2dof(Y,E)
    num_qnodes, num_els = size(el2qnodes)
    γ₀B_block = BlockIndexer(γ₀B,T)
    γ₁B_block = BlockIndexer(γ₁B,T)
    R_block   = BlockIndexer(R,T)
    num_basis = size(γ₀B_block,2)
    M,Mblock  = MatrixAndBlockIndexer(T2,2*num_qnodes,num_basis)
    H,Hblock  = MatrixAndBlockIndexer(T,1,num_basis)
    G,Gblock  = MatrixAndBlockIndexer(T3,1,2*num_qnodes)
    Ss = Vector{T}(undef,num_qnodes)  # for single layer (EFIE)
    Ds = Vector{T}(undef,num_qnodes)  # for double layer (MFIE)
    @assert length(list_near) == num_els
    for n in 1:num_els
        j_glob = @view el2qnodes[:,n]
        for p in 1:num_basis
            for k in 1:num_qnodes
                J = jacobian(qnodes[j_glob[k]])
                Mblock[k,p]            = transpose(J)*γ₀B_block[j_glob[k],p]
                Mblock[num_qnodes+k,p] = transpose(J)*γ₁B_block[j_glob[k],p]
            end
        end
        F = qr!(M)
        for i in list_near[n]
            Hblock[:,:] = @view R_block[i:i,:]
            G[:,:] = (H/F.R)*adjoint(F.Q)
            for k in 1:num_qnodes
                J     = jacobian(qnodes[j_glob[k]])
                Ds[k] = Dcoeff*Gblock[k]*transpose(J)
                Ss[k] = Scoeff*Gblock[k+num_qnodes]*transpose(J)
            end
            # add entries to BlockSparseConstructor
            Is = fill(i,num_qnodes)
            Js = j_glob
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
