## IFGF interface
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

## IFGF wrapper

"""
    ifgf_compressor(;kwargs...)

Returns a `compress` function, which should be passed to `maxwell_dim`
to compress `IntegralOperator` into `IFGFOp`. The `kwargs`
arguments are passed to the `IFGF.assemble_ifgf`.
"""
function ifgf_compressor(;kwargs...)
    function compress(iop::MaxwellOperator)
        # The IFGF only compresses the kernel without
        # the `ncross` and `weights`, therefore they should 
        # be added separately.
        pot_kernel = iop |> Nystrom.kernel |> potential_kernel
        Xmesh = Nystrom.target_surface(iop)
        Ymesh = Nystrom.source_surface(iop)
        X = Xmesh |> Nystrom.qcoords |> collect
        Y = Ymesh |> Nystrom.qcoords |> collect
        # diagonal matrix of ncross(x)
        Nx = Nystrom.cross_product_matrix.(Xmesh|>Nystrom.qnormals)|>Diagonal
        # diagonal matrix of weights(y)
        Wy = Ymesh |> Nystrom.qweights |> collect |> Diagonal
        # IFGF operator
        L = IFGF.assemble_ifgf(pot_kernel,X,Y;kwargs...)
        return Nystrom.DiscreteOp(Nx)*Nystrom.DiscreteOp(L)*Nystrom.DiscreteOp(Wy)
    end
    return compress
end