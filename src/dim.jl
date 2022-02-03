"""
    assemble_dim(iop::IntegralOperator;compression=nothing,location=:onsurface)

Assemble the integral operator `iop` using the *kernel independent density
interpolation  method* to compute the singular integrals.

The keyword argument `location` indicates whether the `target_surface` lies
`:inside`, `:outside`, or on the `source_surface` over which integration is
performed.

The type of the object returned will depend on the keyword `compression`.
"""
function assemble_dim(iop::IntegralOperator;compression=Matrix,location=:onsurface)
    # check validity of location
    msg = "unrecognized value for kw `location`: received $location.
           Valid options are `:onsurface`, `:inside` and `:outside`."
    σ   = location === :onsurface ? -0.5 : location === :inside ? 0 : location === :outside ? -1 : error(msg)
    # unwrap the fields of iop
    X    = target_surface(iop)
    Y    = source_surface(iop)
    # dense part
    Op1, Op2       = _auxiliary_operators_dim(iop,compression)
    # sparse correction
    basis,γ₁_basis = _basis_dim(iop)
    dict_near      = near_interaction_list(dofs(X),Y;atol=0)
    γ₀B,γ₁B,R      = _auxiliary_quantities_dim(iop,Op1,Op2,basis,γ₁_basis,σ)
    correction     = _singular_weights_dim(iop,γ₀B,γ₁B,R,dict_near)
    a,b            = combined_field_coefficients(iop)
    return (a*Op1 + b*Op2 + correction)
end

"""
    _auxiliary_operators_dim(iop,compress)

Given a `CombinedFieldOperator` (or its derivative) written in the form `a*Op1 +
b*Op2`, return `compress(Op1),compress(Op2)`.
"""
function _auxiliary_operators_dim(iop,compress)
    X,Y,op = target_surface(iop), source_surface(iop), pde(kernel(iop))
    T = eltype(iop)
    # construct integral operators required for correction
    if kernel_type(iop) isa CombinedField
        Op1 = IntegralOperator(SingleLayerKernel{T}(op),X,Y) |> compress
        Op2 = IntegralOperator(DoubleLayerKernel{T}(op),X,Y) |> compress
    elseif kernel_type(iop) isa DerivativeCombinedField
        Op1 = IntegralOperator(AdjointDoubleLayerKernel{T}(op),X,Y) |> compress
        Op2 = IntegralOperator(HyperSingularKernel{T}(op),X,Y) |> compress
    else
        error("unrecognized kernel type: $(kernel_type(iop))")
    end
    return Op1,Op2
end

function _auxiliary_quantities_dim(iop,Op1,Op2,basis,γ₁_basis,σ)
    T      = eltype(iop)
    X,Y    = target_surface(iop), source_surface(iop)
    num_basis = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes   = dofs(Y)
    γ₀B      = Matrix{T}(undef,length(ynodes),num_basis)
    γ₁B      = Matrix{T}(undef,length(ynodes),num_basis)
    Threads.@threads for k in 1:num_basis
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
        Threads.@threads for k in 1:num_basis
            # integrate basis over the surface
            R[:,k] = Op1*γ₁B[:,k] - Op2*γ₀B[:,k]
            # @views mul!(R[:,k],Op0,γ₁B[:,k])
            # @views mul!(R[:,k:k],Op1,γ₀B[:,k:k],-1,1)
            σ == 0 && continue # shortcut if nothing to do below
            # add analytic correction from Greens identity
            for i in 1:num_targets
                if kernel_type(iop) isa CombinedField
                    R[i,k] += σ*basis[k](xnodes[i])
                elseif kernel_type(iop) isa DerivativeCombinedField
                    R[i,k] += σ*γ₁_basis[k](xnodes[i])
                else
                    error("unrecognized kernel type")
                end
            end
        end
    end
    return γ₀B, γ₁B, R
end

"""
    _basis_dim(iop)

Return a vector of `basis` functions, as well as their `γ₁` trace, for the
integral operator `iop`.
"""
function _basis_dim(iop)
    op = pde(kernel(iop))
    xs = _source_gen(iop) # list of Lebedev sources
    basis     = [(source) -> SingleLayerKernel(op)(x,source) for x in xs]
    γ₁_basis  = [(source) -> transpose(DoubleLayerKernel(op)(x,source)) for x in xs]
    return basis,γ₁_basis
end

function _singular_weights_dim(iop::IntegralOperator,γ₀B,γ₁B,R,dict_near)
    X,Y = target_surface(iop), source_surface(iop)
    T   = eltype(iop)
    num_basis = size(γ₀B,2)
    a,b = combined_field_coefficients(iop)
    # we now have the residue R. For the correction we need the coefficients.
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for (E,list_near) in dict_near
        # TODO: add function barrier
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
                w    = axpby!(b,view(tmp,1:num_qnodes),-a,view(tmp,(num_qnodes+1):(2*num_qnodes)))
                append!(Is,fill(i,num_qnodes))
                append!(Js,j_glob)
                append!(Vs,w)
            end
        end
    end
    Sp = sparse(Is,Js,Vs,size(iop)...)
    return Sp
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
