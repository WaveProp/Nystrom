# NOTE: assemble weakly-singular kernels using quadgk. Should be slow, but this is
# useful to verify that integrals converge as they should.

function assemble_gk(iop;compress=Matrix)
    out        = compress(iop)
    correction = singular_weights_gk(iop)
    axpy!(1,correction,out) # out <-- out + correction
end

function singular_weights_gk(iop::IntegralOperator)
    X,Y       = target_surface(iop), source_surface(iop)
    T         = eltype(iop)
    dict_near = near_interaction_list(dofs(X),Y;atol=0.1)
    Is    = Int[]
    Js    = Int[]
    Vs    = T[]
    # loop over all integration elements by types
    for E in keys(Y)
        iter      = ElementIterator(Y,E)
        qreg      = Y.etype2qrule[E]
        lag_basis = lagrange_basis(qnodes(qreg))
        _singular_weights_gk!(Is,Js,Vs,iop,iter,dict_near,lag_basis)
    end
    Sp = sparse(Is,Js,Vs,size(iop)...)
    return Sp
end

function _singular_weights_gk!(Is,Js,Vs,iop,iter,dict_near,lag_basis)
    K = kernel(iop)
    X = target_surface(iop)
    Y = source_surface(iop)
    E = eltype(iter)
    list_near = dict_near[E]
    # Over the lines below we loop over all element, find which target nodes
    # need to be regularized, and compute the regularized weights on the source
    # points for that target point.
    for (n,el) in enumerate(iter)               # loop over elements
        jglob = Y.elt2dof[E][:,n]
        for (i,jloc) in list_near[n]            # loop over near targets
            xdof   = dofs(X)[i]
            for (m,l) in enumerate(lag_basis)   # loop over lagrange basis
                val,_ = quadgk(0,1) do ŷ
                    y    = el(ŷ)
                    jac  = jacobian(el,ŷ)
                    μ    = integration_measure(jac)
                    ydof = NystromDOF(y,-1.0,jac,-1,-1)
                    K(xdof,ydof)*μ*l(ŷ)
                end
                # push data to sparse matrix
                push!(Is,i)
                j = jglob[m]
                push!(Js,j)
                push!(Vs,val-iop[i,j])
            end
        end
    end
    return Is,Js,Vs
end
