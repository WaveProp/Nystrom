# Interpolated Factored Green Function (IFGF) Method interface

wavenumber(p::AbstractPDE)    = abstractmethod(p)
wavenumber(::Laplace)         = nothing
wavenumber(p::Helmholtz)      = p.k
wavenumber(p::Elastostatic)   = 2π/p.λ
wavenumber(p::Maxwell)        = p.k
wavenumber(k::AbstractKernel) = wavenumber(pde(k))

function IFGF.IFGFOperator(iop::IntegralOperator;
                           p=(3,5,5),
                           nmax=100,
                           _splitter=IFGF.DyadicSplitter,
                           _profile=false)
    K = kernel(iop)
    PDE = pde(K)
    Ypts = iop |> source_surface |> dofs
    Xpts = iop |> target_surface |> dofs
    splitter = _splitter(;nmax)
    # TODO: better way of doing this?
    datatype = Base.promote_op(*, default_kernel_eltype(PDE), default_density_eltype(PDE))
    p_func = (node) -> p 
    k = wavenumber(K)
    ds_func = isnothing(k) ? IFGF.cone_domain_size_func() : IFGF.cone_domain_size_func(k)
    ifgf = IFGF.IFGFOperator(K,Ypts,Xpts;datatype,splitter,p_func,ds_func,_profile)
    return ifgf
end

function IFGFCompressor(;p=(3,5,5),
                        nmax=100,
                        _splitter=IFGF.DyadicSplitter,
                        _profile=false)
    return (iop::IntegralOperator) -> IFGFOperator(iop;p,nmax,_splitter,_profile)
end

# AbstractKernel
IFGF.centered_factor(K::AbstractKernel,x,yc) = abstractmethod(K)

# Laplace
function IFGF.centered_factor(K::SingleLayerKernel{T,<:Laplace},x,yc) where T
    return K(x,yc)
end
function IFGF.centered_factor(::DoubleLayerKernel{T,<:Laplace},x,yc) where T
    # TODO: pick a better centered_factor
    r = coords(x)-yc
    d = norm(r)
    return 1/d
end

# Helmholtz
function IFGF.centered_factor(K::SingleLayerKernel{T,<:Helmholtz{3}},x,yc) where T
    # return exp(im*k*d)/d
    return K(x,yc)
end
function IFGF.centered_factor(K::DoubleLayerKernel{T,<:Helmholtz{3}},x,yc) where T
    k = wavenumber(K)
    r = coords(x)-yc
    d = norm(r)
    return exp(im*k*d)/d*(-im*k+1/d)
end

# Elastostatic
function IFGF.centered_factor(::SingleLayerKernel{T,<:Elastostatic{3}},x,yc) where T
    r = coords(x)-yc
    d = norm(r)
    return 1/d
end
function IFGF.centered_factor(::DoubleLayerKernel{T,<:Elastostatic{3}},x,yc) where T
    # TODO: pick a better centered_factor
    r = coords(x)-yc
    d = norm(r)
    return 1/d
end

# Maxwell
function IFGF.centered_factor(K::SingleLayerKernel{T,<:Maxwell},x,yc) where T
    # TODO: pick a better centered_factor
    k = wavenumber(K)
    r = coords(x)-yc
    d = norm(r)
    g = exp(im*k*d)/(4π*d)
    # gp  = im*k*g - g/d
    # gpp = im*k*gp - gp/d + g/d^2
    # RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    return g   
end
function IFGF.centered_factor(K::DoubleLayerKernel{T,<:Maxwell},x,yc) where T
    # TODO: pick a better centered_factor
    k = wavenumber(K)
    r = coords(x)-yc
    d = norm(r)
    g = exp(im*k*d)/(4π*d)
    return g   
end
