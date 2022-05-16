####
# Interpolated Factored Green Function (IFGF) method interface
####

IFGF.wavenumber(K::MaxwellPotentialKernel) = K |> Nystrom.pde |> Nystrom.parameters

function IFGF.centered_factor(K::MaxwellPotentialKernel,x,Y)
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    g  = exp(im*k*d)/(4π*d)  # Helmholtz Green's function
    return g
end

function IFGF.inv_centered_factor(K::MaxwellPotentialKernel, x, Y)
    # return inv(centered_factor(K, x, Y))
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    invg = (4π*d)/exp(im*k*d)  # inverse of Helmholtz Green's function
    return invg
end

function IFGF.transfer_factor(K::MaxwellPotentialKernel, x, Y)
    Yparent = parent(Y)
    # return inv_centered_factor(K, x, Yparent) * centered_factor(K, x, Y)
    k   = IFGF.wavenumber(K)
    yc  = IFGF.center(Y)
    ycp = IFGF.center(Yparent)
    r   = x-yc
    rp  = x-ycp
    d   = norm(r)
    dp  = norm(rp)
    return dp/d*exp(im*k*(d-dp))
end

####
# IFGF wrappers
####

function Nystrom._ifgf_compress(iop::MaxwellOperator;kwargs...)
    # The IFGF only compresses the kernel without
    # the `ncross` and `weights`, therefore they must 
    # be considered separately.
    pot_kernel = iop |> Nystrom.kernel |> potential_kernel
    Xmesh = Nystrom.target_surface(iop)
    Ymesh = Nystrom.source_surface(iop)
    X = Xmesh |> Nystrom.qcoords |> collect
    Y = Xmesh===Ymesh ? X : Ymesh|>Nystrom.qcoords|>collect
    # TODO: implement efficient version which does not save
    # the explicit matrices.
    # diagonal matrix of ncross(x)
    Nx = Nystrom.cross_product_matrix.(Xmesh|>Nystrom.qnormals)|>Diagonal
    # diagonal matrix of weights(y)
    Wy = [SMatrix{3,3,Float64,9}(w,0,0,
                                 0,w,0,
                                 0,0,w) for w in Nystrom.qweights(Ymesh)] |> Diagonal
    # IFGF operator
    L = IFGF.assemble_ifgf(pot_kernel,X,Y;kwargs...)
    return Nystrom.DiscreteOp(Nx)*Nystrom.DiscreteOp(L)*Nystrom.DiscreteOp(Wy)
end
