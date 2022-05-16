####
# Interpolated Factored Green Function (IFGF) method interface
####

IFGF.wavenumber(::AbstractKernel) = notimplemented()

IFGF.centered_factor(::AbstractKernel,x,Y)     = notimplemented()
IFGF.inv_centered_factor(::AbstractKernel,x,Y) = notimplemented()
IFGF.transfer_factor(::AbstractKernel,x,Y)     = notimplemented()

####
# IFGF wrappers
####

"""
    _ifgf_compress(iop::IntegralOperator;kwargs...)

Compress an `iop::IntegralOperator` into a `IFGFOp`.
The `kwargs` arguments are passed to `IFGF.assemble_ifgf`.
The returned object is a `DiscreteOp` which contains the
`IFGFOp`.
"""
function _ifgf_compress(iop::IntegralOperator;kwargs...)
    # The IFGF only compresses the kernel without
    # the `weights`, therefore they must 
    # be considered separately.
    K = kernel(iop)
    T = default_density_eltype(pde(K))
    Xmesh = target_surface(iop)
    Ymesh = source_surface(iop)
    Wy_list = qweights(Ymesh)
    X = Xmesh |> qcoords |> collect
    Y = Xmesh===Ymesh ? X : Ymesh|>qcoords|>collect
    ifgf = IFGF.assemble_ifgf(K,X,Y;kwargs...)  # IFGF operator
    # diagonal matrix of weights
    if T <: Number
        Wy = collect(Wy_list) |> Diagonal
    else
        notimplemented()
    end
    L = DiscreteOp(ifgf)*DiscreteOp(Wy)
    return L
end

"""
    ifgf_compressor(;kwargs...)

Return a `compress` function, which should be passed to `assemble_dim`
to compress `IntegralOperator`s into `IFGFOp`s. The `kwargs`
arguments are passed to `IFGF.assemble_ifgf`.
"""
function ifgf_compressor(;kwargs...)
    compress = (iop::IntegralOperator) -> _ifgf_compress(iop;kwargs...)
    return compress
end

####
# Interface with `Density`
####

function LinearAlgebra.mul!(y::Density{<:SVector},
                            A::IFGF.IFGFOp{<:SMatrix},
                            x::Density{<:SVector},
                            a::Number,b::Number)
    mul!(vals(y),A,vals(x),a,b)
return y
end

####
# Laplace
####

####
# Helmholtz
####

const HelmholtzSingleLayerKernel3D = SingleLayerKernel{T,S} where {T,S<:Helmholtz{3}}

IFGF.wavenumber(K::HelmholtzSingleLayerKernel3D) = K |> Nystrom.pde |> Nystrom.parameters

function IFGF.centered_factor(K::HelmholtzSingleLayerKernel3D,x,Y)
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    g  = exp(im*k*d)/(4π*d)  # Helmholtz Green's function
    return g
end

function IFGF.inv_centered_factor(K::HelmholtzSingleLayerKernel3D, x, Y)
    # return inv(centered_factor(K, x, Y))
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    invg = (4π*d)/exp(im*k*d)  # inverse of Helmholtz Green's function
    return invg
end

function IFGF.transfer_factor(K::HelmholtzSingleLayerKernel3D, x, Y)
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
# Elastostatic
####
