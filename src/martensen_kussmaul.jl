function assemble_mk(iop::IntegralOperator)
    assert_mk_compatibility(iop)
    X = target_surface(iop)
    Y = source_surface(iop)
    N   = length(X.dofs) # number of dof
    out = zero(iop) # output matrix to be filled
    for target_ent in entities(X)
        for source_ent in entities(Y)
            if source_ent == target_ent
                _assemble_mk_self!(out,iop,target_ent)
            else
                # nothing special to do here, just fill up the matrix using the
                # regular quadrature
                target_dof  = dom2dof(X,target_ent)
                source_dof  = dom2dof(Y,source_ent)
                @. out[target_dof,source_dof] = iop[target_dof,source_dof]
            end
        end
    end
    return out
end

# Assemble the self-interaction part for a given entity
function _assemble_mk_self!(out,iop,ent::AbstractEntity)
    X         = target_surface(iop)
    etype2tag = X.ent2elt[ent]
    # loop over individual elements composing ent. Note that because we must
    # handle meshes composed of elements of possiby different types, we must
    # loop over element types, and then over the elements of that type
    # (contained in the entity)
    for (target_E,target_tags) in etype2tag, target_t in target_tags
        for (source_E,source_tags) in etype2tag, source_t in source_tags
            if source_E === target_E && target_t === source_t # same element
                dof = X.elt2dof[source_E][:,source_t] # dof for filling out in global matrix
                el   = X.elements[source_E][source_t]                 # retrieve the element (i.e. access to its parametrization)
                _assemble_mk_self!(out,iop,el,dof)
            else
                target_dof  = X.elt2dof[target_E][:,target_t] # dof for filling out in global matrix
                source_dof  = X.elt2dof[source_E][:,source_t] # dof for filling out in global matrix
                out[target_dof,source_dof] = iop[target_dof,source_dof]
            end
        end
    end
    return out
end

function _assemble_mk_self!(out,iop,el::AbstractElement,idxs)
    X     = target_surface(iop)
    qrule = X.etype2qrule[typeof(el)] # quadrature used on elements of type el
    x̂,ŵ   = qrule()                   # nodes on reference segment [0,1]
    # compute all quantities needed for mk rule, which includes high order
    # derivatives of the parametrization.
    N     = length(idxs)
    dx    =  [ParametricSurfaces.derivative(el,u)/(2π)     for u in x̂]
    d2x   =  [ParametricSurfaces.derivative2(el,u)/(4*π^2) for u in x̂]
    τ     = [norm(τ) for τ in dx]
    Δs    = 2π / N
    psi   = -0.577215664901532
    xdofs = dofs(X)
    x     = qcoords(X) |> collect
    ν     = qnormals(X) |> collect
    w     = qweights(X) |> collect
    R     = nystrom_weights(N)
    k     = iop.kernel.pde.k
    K     = kernel(iop)
    φ     = mk_split(K)
    for (iloc,iglob) in enumerate(idxs)
        for (jloc,jglob) in enumerate(idxs)
            lSin = log(4*(sin(Δs*(iloc-jloc)/2))^2) # singular part factored
            if kernel_type(iop) === SingleLayer()
                K1   = φ(x[iglob],x[jglob])*w[jglob]
                if iloc != jloc
                    K2   = K(x[iglob],x[jglob])*w[jglob]-K1*lSin # what is left after factoring out lSin and φ
                else
                    K2   = (im/4+psi/2/pi-1/4/pi*log(k^2/4*τ[jloc]^2))*w[jglob] + 2*log(w[jglob]/(τ[jloc]*Δs))*K1
                end
            elseif kernel_type(iop) == DoubleLayer()
                K1   = φ(x[iglob],x[jglob],ν[jglob])*w[jglob]
                if iloc != jloc
                    K2   = K(xdofs[iglob],xdofs[jglob])*w[jglob]-K1*lSin
                else
                    K2   = -1/(4*pi)*(dx[jloc][1]*d2x[jloc][2]-dx[jloc][2]*d2x[jloc][1])/τ[jloc]^3*w[jglob] + 2*log(w[jglob]/(τ[jloc]*Δs))*K1
                end
            end
            out[iglob,jglob] = (R[iloc,jloc]*K1 + K2)
        end
    end
    return out
end

# analytical splitting of kernel in 2d Helmholtz
function mk_split(SL::SingleLayerKernel{T,S}) where {T,S<:Helmholtz}
    k = SL.pde.k
    ϕ = (x,y) -> begin
        d = norm(x-y)
        (-1/(4*pi))*besselj0(k*d) |> T
    end
    return ϕ
end

function mk_split(DL::DoubleLayerKernel{T,S}) where {T,S<:Helmholtz}
    k = DL.pde.k
    ϕ = (x,y,ν) -> begin
        x == y && (return zero(T)) # analitic limit taken by hand to avoid 0/0
        r = x-y
        d = norm(r)
        (-k/(4*pi))*besselj1(k*d)/d*dot(r,ν) |> T
    end
    return ϕ
end

function nystrom_weights(M)
    @assert M%2 == 0  "number of points `M` must be even"
    Md2 = div(M,2)
    R   = zeros(M,M)
    for p=1:M
        tp = pi/Md2*(p-1)
        for j=1:M
            tj = pi/Md2*(j-1)
            R[p,j]=-2*sum((ones(Md2-1)./collect(1:Md2-1)).*cos.(collect(1:Md2-1)*(tp-tj)))-pi/(Md2^2)*cos(Md2*(tp-tj))
        end
    end
    return R
end

function assert_mk_compatibility(iop::IntegralOperator)
    # check 2d
    @assert ambient_dimension(iop) == 2 "Martensen-Kussmaul method only applicable to two-dimensional problems"
    # FIXME: add other checks...
end
