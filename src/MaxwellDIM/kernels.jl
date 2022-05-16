###
# PDE
###
"""
    Maxwell{T} <: Nystrom.AbstractPDE{3}

Normalized Maxwell's equation ∇ × ∇ × E - k² E = 0.
"""
struct Maxwell{T} <: Nystrom.AbstractPDE{3}
    k::T
end

Maxwell(;k::T) where {T} = Maxwell{T}(k)

Nystrom.parameters(pde::Maxwell) = pde.k

function Base.show(io::IO,::Maxwell)
    # k = parameters(pde)
    print(io,"∇ × ∇ × E - k² E = 0")
end

Nystrom.default_kernel_eltype(::Maxwell)   = SMatrix{3,3,ComplexF64,9}
Nystrom.default_density_eltype(::Maxwell)  = SVector{3,ComplexF64}

###
# Potentials
###
# The Potential kernels are different from the IntegralOperator kernels.

# EFIE Potential Kernel
struct EFIEPotentialKernel{T,Op<:Maxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# MFIE Potential Kernel
struct MFIEPotentialKernel{T,Op<:Maxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

const MaxwellPotentialKernel = Union{EFIEPotentialKernel,MFIEPotentialKernel}

# EFIE Potential Kernel definition
function _efie_potential_kernel(target,source,k)
    x = Nystrom.coords(target)
    y = Nystrom.coords(source)
    rvec = x - y
    r = norm(rvec)
    g = exp(im*k*r)/(4π*r) # Helmholtz greens function
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    G = g*I + 1/k^2*(gp/r*I + (gpp/r^2 - gp/r^3)*RRT)
    return (!iszero(r))*G
end
function (SL::EFIEPotentialKernel)(target,source)
    k = Nystrom.parameters(Nystrom.pde(SL))
    G = _efie_potential_kernel(target,source,k)
    return G
end

# MFIE Potential Kernel definition
function _mfie_potential_kernel(target,source,k)
    x = Nystrom.coords(target)
    y = Nystrom.coords(source)
    rvec = x - y
    r = norm(rvec)
    g   = exp(im*k*r)/(4π*r) # Helmholtz greens function
    gp  = im*k*g - g/r
    rcross = Nystrom.cross_product_matrix(rvec)
    curl_G = gp/r*rcross
    return (!iszero(r))*curl_G
end
function (SL::MFIEPotentialKernel)(target,source)
    k = Nystrom.parameters(Nystrom.pde(SL))
    curl_G = _mfie_potential_kernel(target,source,k)
    return curl_G
end

EFIEPotential(op::Maxwell,surf) = Nystrom.IntegralPotential(EFIEPotentialKernel(op),surf)
MFIEPotential(op::Maxwell,surf) = Nystrom.IntegralPotential(MFIEPotentialKernel(op),surf)

const MaxwellPotential = Nystrom.IntegralOperator{<:MaxwellPotentialKernel}

###
# Integral Operators Kernel
###
# EFIE Kernel
struct EFIEKernel{T,Op<:Maxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# MFIE Kernel
struct MFIEKernel{T,Op<:Maxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

const MaxwellKernel = Union{EFIEKernel,MFIEKernel}

# EFIE Kernel definition
function (SL::EFIEKernel)(target,source)
    k = Nystrom.parameters(Nystrom.pde(SL))
    G = _efie_potential_kernel(target,source,k)
    nx = Nystrom.normal(target)
    ncross = Nystrom.cross_product_matrix(nx)
    return ncross*G
end

# MFIE Kernel definition
function (DL::MFIEKernel)(target,source)
    k = Nystrom.parameters(Nystrom.pde(DL))
    curl_G = _mfie_potential_kernel(target,source,k)
    nx = Nystrom.normal(target)
    ncross = Nystrom.cross_product_matrix(nx)
    return ncross*curl_G
end

EFIEOperator(op::Maxwell,X,Y=X) = Nystrom.IntegralOperator(EFIEKernel(op), X, Y)
MFIEOperator(op::Maxwell,X,Y=X) = Nystrom.IntegralOperator(MFIEKernel(op), X, Y)

const MaxwellOperator = Nystrom.IntegralOperator{T,<:MaxwellKernel} where T

"""
    potential_kernel(K::EFIEKernel)
    potential_kernel(K::MFIEKernel)

Returns the underlying `MaxwellPotentialKernel` (e.g. `EFIEPotentialKernel`)
associated with a `MaxwellKernel` kernel (e.g. `EFIEKernel`).
"""
potential_kernel(K::EFIEKernel) = EFIEPotentialKernel(Nystrom.pde(K))
potential_kernel(K::MFIEKernel) = MFIEPotentialKernel(Nystrom.pde(K))
