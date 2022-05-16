###
# PDE
###
"""
    TrueMaxwell{T} <: Nystrom.AbstractPDE{3}

Normalized Maxwell's equation ∇ × ∇ × E - k² E = 0.
"""
struct TrueMaxwell{T} <: Nystrom.AbstractPDE{3}
    k::T
end

TrueMaxwell(;k::T) where {T} = TrueMaxwell{T}(k)

Nystrom.parameters(pde::TrueMaxwell) = pde.k

function Base.show(io::IO,::TrueMaxwell)
    # k = parameters(pde)
    print(io,"∇ × ∇ × E - k² E = 0")
end

Nystrom.default_kernel_eltype(::TrueMaxwell)   = SMatrix{3,3,ComplexF64,9}
Nystrom.default_density_eltype(::TrueMaxwell)  = SVector{3,ComplexF64}

###
# Potentials
###
# The Potential kernels are different from the IntegralOperator kernels.

# EFIE Potential Kernel
struct EFIEPotentialKernel{T,Op<:TrueMaxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# MFIE Potential Kernel
struct MFIEPotentialKernel{T,Op<:TrueMaxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# EFIE Potential Kernel definition
function (SL::EFIEPotentialKernel{T})(target,source)::T  where T
    k = Nystrom.parameters(Nystrom.pde(SL))
    x  = Nystrom.coords(target)
    y  = Nystrom.coords(source)
    rvec = x - y
    r = norm(rvec)
    r == 0 && (return zero(T))
    g = exp(im*k*r)/(4π*r) # Helmholtz greens function
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    G = g*I + 1/k^2*(gp/r*I + (gpp/r^2 - gp/r^3)*RRT)
    return G
end

# MFIE Potential Kernel definition
function (SL::MFIEPotentialKernel{T})(target,source)::T  where T
    k = Nystrom.parameters(Nystrom.pde(SL))
    x  = Nystrom.coords(target)
    y  = Nystrom.coords(source)
    rvec = x - y
    r = norm(rvec)
    r == 0 && (return zero(T))
    g   = exp(im*k*r)/(4π*r) # Helmholtz greens function
    gp  = im*k*g - g/r
    rcross = Nystrom.cross_product_matrix(rvec)
    curl_G = gp/r*rcross
    return curl_G
end

EFIEPotential(op::TrueMaxwell,surf) = Nystrom.IntegralPotential(EFIEPotentialKernel(op),surf)
MFIEPotential(op::TrueMaxwell,surf) = Nystrom.IntegralPotential(MFIEPotentialKernel(op),surf)

###
# Integral Operators Kernel
###
# EFIE Kernel
struct EFIEKernel{T,Op<:TrueMaxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# MFIE Kernel
struct MFIEKernel{T,Op<:TrueMaxwell} <: Nystrom.AbstractPDEKernel{T,Op}
    pde::Op
end

# EFIE Kernel definition
function (SL::EFIEKernel{T})(target,source)::T  where T
    pot = EFIEPotentialKernel(Nystrom.pde(SL))
    nx = Nystrom.normal(target)
    ncross = Nystrom.cross_product_matrix(nx)
    return ncross*pot(target,source)
end

# MFIE Kernel definition
function (DL::MFIEKernel{T})(target,source)::T where T
    pot = MFIEPotentialKernel(Nystrom.pde(DL))
    nx = Nystrom.normal(target)
    ncross = Nystrom.cross_product_matrix(nx)
    return ncross*pot(target,source)
end

EFIEOperator(op::TrueMaxwell,X,Y=X) = Nystrom.IntegralOperator(EFIEKernel(op), X, Y)
MFIEOperator(op::TrueMaxwell,X,Y=X) = Nystrom.IntegralOperator(MFIEKernel(op), X, Y)




