abstract type AbstractPDE{N} end

Geometry.ambient_dimension(pde::AbstractPDE{N}) where {N} = N

"""
    abstract type AbstractKernel{T}

A kernel functions `K` with the signature `K(target,source)::T`.

# See also: [`GenericKernel`](@ref), [`SingleLayerKernel`](@ref), [`DoubleLayerKernel`](@ref), [`AdjointDoubleLayerKernel`](@ref), [`HyperSingularKernel`](@ref)
"""
abstract type AbstractKernel{T} end

return_type(::AbstractKernel{T}) where {T} = T

ambient_dimension(K::AbstractKernel) = ambient_dimension(pde(K))

"""
    pde(K::AbstractKernel)

Return the underlying `AbstractPDE` when `K` correspond to the kernel of an
integral operator derived from a partial differential equation.
"""
pde(k::AbstractKernel) = k.pde

"""
    struct GenericKernel{T,F} <: AbstractKernel{T}

An [`AbstractKernel`](@ref) with `kernel` of type `F`.
"""
struct GenericKernel{T,F} <: AbstractKernel{T}
    kernel::F
end

"""
    struct SingleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op::Abstract`, construct its free-space single-layer kernel (i.e. the
fundamental solution).
"""
struct SingleLayerKernel{T,Op} <: AbstractKernel{T}
    pde::Op
end
SingleLayerKernel{T}(op) where {T} = SingleLayerKernel{T,typeof(op)}(op)
SingleLayerKernel(op)              = SingleLayerKernel{default_kernel_eltype(op)}(op)

"""
    struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space double-layer kernel. This
corresponds to the `γ₁` trace of the [`SingleLayerKernel`](@ref). For operators
such as [`Laplace`](@ref) or [`Helmholtz`](@ref), this is simply the normal
derivative of the fundamental solution respect to the source variable.
"""
struct DoubleLayerKernel{T,Op} <: AbstractKernel{T}
    pde::Op
end
DoubleLayerKernel{T}(op) where {T}     = DoubleLayerKernel{T,typeof(op)}(op)
DoubleLayerKernel(op)                  = DoubleLayerKernel{default_kernel_eltype(op)}(op)

"""
    struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space adjoint double-layer kernel. This
corresponds to the `transpose(γ₁,ₓ[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable.
"""
struct AdjointDoubleLayerKernel{T,Op} <: AbstractKernel{T}
    pde::Op
end
AdjointDoubleLayerKernel{T}(op) where {T} = AdjointDoubleLayerKernel{T,typeof(op)}(op)
AdjointDoubleLayerKernel(op)              = AdjointDoubleLayerKernel{default_kernel_eltype(op)}(op)

"""
    struct HyperSingularKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space hypersingular kernel. This
corresponds to the `transpose(γ₁,ₓγ₁[G])`, where `G` is the
[`SingleLayerKernel`](@ref). For operators such as [`Laplace`](@ref) or
[`Helmholtz`](@ref), this is simply the normal derivative of the fundamental
solution respect to the target variable of the `DoubleLayerKernel`.
"""
struct HyperSingularKernel{T,Op} <: AbstractKernel{T}
    pde::Op
end
HyperSingularKernel{T}(op) where {T} = HyperSingularKernel{T,typeof(op)}(op)
HyperSingularKernel(op)              = HyperSingularKernel{default_kernel_eltype(op)}(op)

"""
    struct CombinedFieldKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space combined field kernel. This
corresponds to a linear combination of `S::SingleLayerKernel` and `D::DoubleLayerKernel`
of the form `α*D - β*S`.
"""
struct CombinedFieldKernel{T,Op,Tm<:Number} <: AbstractKernel{T}
    pde::Op
    α::Tm
    β::Tm
end
CombinedFieldKernel{T}(op,α::Tm,β::Tm) where {T,Tm} = CombinedFieldKernel{T,typeof(op),Tm}(op,α,β)
CombinedFieldKernel(op,α,β) = CombinedFieldKernel{default_kernel_eltype(op)}(op,promote(α,β)...)

"""
    struct AdjointCombinedFieldKernel{T,Op} <: AbstractKernel{T}

Given an operator `Op`, construct its free-space adjoint combined field kernel. This
corresponds to a linear combination of `K::AdjointDoubleLayerKernel` and `H::HyperSingularKernel`
of the form `α*H - β*K`.
"""
struct AdjointCombinedFieldKernel{T,Op,Tm<:Number} <: AbstractKernel{T}
    pde::Op
    α::Tm
    β::Tm
end
AdjointCombinedFieldKernel{T}(op,α::Tm,β::Tm) where {T,Tm} = AdjointCombinedFieldKernel{T,typeof(op),Tm}(op,α,β)
AdjointCombinedFieldKernel(op,α,β) = AdjointCombinedFieldKernel{default_kernel_eltype(op)}(op,promote(α,β)...)

# a trait for the kernel type
struct SingleLayer end
struct DoubleLayer end
struct AdjointDoubleLayer end
struct HyperSingular end
struct CombinedField end
struct AdjointCombinedField end

kernel_type(::SingleLayerKernel)          = SingleLayer()
kernel_type(::DoubleLayerKernel)          = DoubleLayer()
kernel_type(::AdjointDoubleLayerKernel)   = AdjointDoubleLayer()
kernel_type(::HyperSingularKernel)        = HyperSingular()
kernel_type(::CombinedFieldKernel)        = CombinedField()
kernel_type(::AdjointCombinedFieldKernel) = AdjointCombinedField()

combined_field_coefficients(::SingleLayerKernel)        = (0,-1)
combined_field_coefficients(::DoubleLayerKernel)        = (1,0)
combined_field_coefficients(::AdjointDoubleLayerKernel) = (0,-1)
combined_field_coefficients(::HyperSingularKernel)      = (1,0)
combined_field_coefficients(r::CombinedFieldKernel)     = (r.α,r.β)
combined_field_coefficients(r::AdjointCombinedFieldKernel) = (r.α,r.β)

# definition of combined field kernels
function (CF::CombinedFieldKernel)(target,source)
    SL = SingleLayerKernel(pde(CF))
    DL = DoubleLayerKernel(pde(CF))
    α,β = combined_field_coefficients(CF) 
    return α*DL(target,source) - β*SL(target,source)
end

function (CF::AdjointCombinedFieldKernel)(target,source)
    K = AdjointDoubleLayerKernel(pde(CF))
    H = HyperSingularKernel(pde(CF))
    α,β = combined_field_coefficients(CF) 
    return α*H(target,source) - β*K(target,source)
end

################################################################################
################################# LAPLACE ######################################
################################################################################

"""
    struct Laplace{N}

Laplace equation in `N` dimension: Δu = 0.
"""
struct Laplace{N} <: AbstractPDE{N} end

Laplace(;dim=3) = Laplace{dim}()

function Base.show(io::IO,pde::Laplace)
    print(io,"Δu = 0")
end

default_kernel_eltype(::Laplace)  = Float64
default_density_eltype(::Laplace) = Float64

function (SL::SingleLayerKernel{T,Laplace{N}})(target,source)::T  where {N,T}
    x = coords(target)
    y = coords(source)
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N==2
        return -1/(2π)*log(d)
    elseif N==3
        return 1/(4π)/d
    else
        notimplemented()
    end
end

function (DL::DoubleLayerKernel{T,Laplace{N}})(target,source)::T where {N,T}
    x = coords(target)
    y = coords(source)
    ny = normal(source)
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N == 2
        return 1/(2π)/(d^2) .* dot(r,ny)
    elseif N==3
        return 1/(4π)/(d^3) .* dot(r,ny)
    else
        notimplemented()
    end
end

function (ADL::AdjointDoubleLayerKernel{T,Laplace{N}})(target,source)::T where {N,T}
    x = coords(target)
    y = coords(source)
    nx = normal(target)
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N==2
        return -1/(2π)/(d^2) .* dot(r,nx)
    elseif N==3
        return -1/(4π)/(d^3) .* dot(r,nx)
    end
end

function (HS::HyperSingularKernel{T,Laplace{N}})(target,source)::T where {N,T}
    x = coords(target)
    y = coords(source)
    nx = normal(target)
    ny = normal(source)
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N==2
        ID = SMatrix{2,2,Float64,4}(1,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(2π)/(d^2) * transpose(nx)*(( ID -2*RRT/d^2  )*ny)
    elseif N==3
        ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        RRT = r*transpose(r) # r ⊗ rᵗ
        return 1/(4π)/(d^3) * transpose(nx)*(( ID -3*RRT/d^2  )*ny)
    end
end

################################################################################
################################# Helmholtz ####################################
################################################################################

"""
    struct Helmholtz{N,T}

Helmholtz equation in `N` dimensions: Δu + k²u = 0.
"""
struct Helmholtz{N,K} <: AbstractPDE{N}
    k::K
end

Helmholtz(;k,dim=3) = Helmholtz{dim,typeof(k)}(k)

function Base.show(io::IO,pde::Helmholtz)
    # k = parameters(pde)
    print(io,"Δu + k u = 0")
end

parameters(pde::Helmholtz) = pde.k

default_kernel_eltype(::Helmholtz)  = ComplexF64
default_density_eltype(::Helmholtz) = ComplexF64

function (SL::SingleLayerKernel{T,S})(target, source)::T where {T,S <: Helmholtz}
    x = coords(target)
    y = coords(source)
    N = ambient_dimension(pde(SL))
    k = parameters(pde(SL))
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N == 2
        return im / 4 * hankelh1(0, k * d)
    elseif N == 3
        return 1 / (4π) / d * exp(im * k * d)
    end
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(target, source)::T where {T,S <: Helmholtz}
    x, y, ny = coords(target), coords(source), normal(source)
    N = ambient_dimension(pde(DL))
    k = parameters(pde(DL))
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N == 2
        return im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
    elseif N == 3
        return 1 / (4π) / d^2 * exp(im * k * d) * ( -im * k + 1 / d ) * dot(r, ny)
    end
end

# Adjoint double Layer Kernel
function (ADL::AdjointDoubleLayerKernel{T,S})(target, source)::T where {T,S <: Helmholtz}
    x, y, nx = coords(target), coords(source), normal(target)
    N = ambient_dimension(pde(ADL))
    k = parameters(pde(ADL))
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N == 2
        return -im * k / 4 / d * hankelh1(1, k * d) .* dot(r, nx)
    elseif N == 3
        return -1 / (4π) / d^2 * exp(im * k * d) * ( -im * k + 1 / d ) * dot(r, nx)
    end
end

# Hypersingular kernel
function (HS::HyperSingularKernel{T,S})(target, source)::T where {T,S <: Helmholtz}
    x, y, nx, ny = coords(target), coords(source), normal(target), normal(source)
    N = ambient_dimension(pde(HS))
    k = parameters(pde(HS))
    r = x - y
    d = norm(r)
    d == 0 && (return zero(T))
    if N == 2
        RRT = r * transpose(r) # r ⊗ rᵗ
        # TODO: rewrite the operation below in a more clear/efficient way
        return transpose(nx) * ((-im * k^2 / 4 / d^2 * hankelh1(2, k * d) * RRT + im * k / 4 / d * hankelh1(1, k * d) * I) * ny)
    elseif N == 3
        RRT   = r * transpose(r) # r ⊗ rᵗ
        term1 = 1 / (4π) / d^2 * exp(im * k * d) * ( -im * k + 1 / d ) * I
        term2 = RRT / d * exp(im * k * d) / (4 * π * d^4) * (3 * (d * im * k - 1) + d^2 * k^2)
        return  transpose(nx) * (term1 + term2) * ny
    end
end

################################################################################
################################# Elastostatic #################################
################################################################################

"""
    struct Elastostatic{N,T} <: AbstractPDE{N}

Elastostatic equation in `N` dimensions: μΔu + (μ+λ)∇(∇⋅u) = 0. Note that the
displacement u is a vector of length `N` since this is a vectorial problem.
"""
struct Elastostatic{N,T} <: AbstractPDE{N}
    μ::T
    λ::T
end
Elastostatic(;μ,λ,dim=3)               = Elastostatic{dim}(promote(μ,λ)...)
Elastostatic{N}(μ::T,λ::T) where {N,T} = Elastostatic{N,T}(μ,λ)

function Base.show(io::IO,pde::Elastostatic)
    print(io,"μΔu + (μ+λ)∇(∇⋅u) = 0")
end

parameters(pde::Elastostatic) = pde.μ, pde.λ

default_kernel_eltype(::Elastostatic{N}) where {N}  = SMatrix{N,N,Float64,N*N}
default_density_eltype(::Elastostatic{N}) where {N} = SVector{N,Float64}

function (SL::SingleLayerKernel{T,S})(target,source)::T  where {T,S<:Elastostatic}
    N   = ambient_dimension(pde(SL))
    μ,λ = parameters(pde(SL))
    ν = λ/(2*(μ+λ))
    x = coords(target)
    y = coords(source)
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r*transpose(r) # r ⊗ rᵗ
    if N==2
        ID = SMatrix{2,2,Float64,4}(1,0,0,1)
        return 1/(8π*μ*(1-ν))*(-(3-4*ν)*log(d)*ID + RRT/d^2)
        # return (λ + 3μ)/(4*π*(N-1)*μ*(λ+2μ))* (-log(d)*ID + (λ+μ)/(λ+3μ)*RRT/d^2)
    elseif N==3
        ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        return 1/(16π*μ*(1-ν)*d)*((3-4*ν)*ID + RRT/d^2)
    end
end

function (DL::DoubleLayerKernel{T,S})(target,source)::T where {T,S<:Elastostatic}
    N = ambient_dimension(pde(DL))
    μ,λ = parameters(pde(DL))
    ν = λ/(2*(μ+λ))
    x = coords(target)
    y = coords(source)
    ny = normal(source)
    ν = λ/(2*(μ+λ))
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = -dot(r,ny)/d
    if N==2
        ID = SMatrix{2,2,Float64,4}(1,0,0,1)
        return -1/(4π*(1-ν)*d)*(drdn*((1-2ν)*ID + 2*RRT/d^2) + (1-2ν)/d*(r*transpose(ny) - ny*transpose(r)))
    elseif N==3
        ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        return -1/(8π*(1-ν)*d^2)*(drdn * ((1-2*ν)*ID + 3*RRT/d^2) + (1-2*ν)/d*(r*transpose(ny) - ny*transpose(r)))
    end
end

function (ADL::AdjointDoubleLayerKernel{T,S})(target,source)::T where {T,S<:Elastostatic}
    # DL = DoubleLayerKernel{T}(pde(ADL))
    # return -DL(x,y,nx) |> transpose
    N   = ambient_dimension(pde(ADL))
    μ,λ = parameters(pde(ADL))
    ν = λ/(2*(μ+λ))
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    ν = λ/(2*(μ+λ))
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT = r*transpose(r) # r ⊗ rᵗ
    drdn = -dot(r,nx)/d
    if N==2
        ID = SMatrix{2,2,Float64,4}(1,0,0,1)
        out = -1/(4π*(1-ν)*d)*(drdn*((1-2ν)*ID + 2*RRT/d^2) + (1-2ν)/d*(r*transpose(nx) - nx*transpose(r)))
        return -transpose(out)
    elseif N==3
        ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        out =  -1/(8π*(1-ν)*d^2)*(drdn * ((1-2*ν)*ID + 3*RRT/d^2) + (1-2*ν)/d*(r*transpose(nx) - nx*transpose(r)))
        return -transpose(out)
    end
end

function (HS::HyperSingularKernel{T,S})(target,source)::T where {T,S<:Elastostatic}
    N = ambient_dimension(pde(HS))
    μ,λ = parameters(pde(HS))
    ν = λ/(2*(μ+λ))
    x = coords(target)
    nx = normal(target)
    y = coords(source)
    ny = normal(source)
    r = x .- y
    d = norm(r)
    d == 0 && return zero(T)
    RRT   = r*transpose(r) # r ⊗ rᵗ
    drdn  = dot(r,ny)/d
    if N==2
        ID = SMatrix{2,2,Float64,4}(1,0,0,1)
        return μ/(2π*(1-ν)*d^2)* (2*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(dot(r,nx)*ID + r*transpose(nx)) - 4*dot(r,nx)*RRT/d^2 ) +
                                  2*ν/d^2*(dot(r,nx)*ny*transpose(r) + dot(nx,ny)*RRT) +
                                  (1-2*ν)*(2/d^2*dot(r,nx)*r*transpose(ny) + dot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    elseif N==3
        ID = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
        return μ/(4π*(1-ν)*d^3)* (3*drdn/d*( (1-2ν)*nx*transpose(r) + ν*(dot(r,nx)*ID + r*transpose(nx)) - 5*dot(r,nx)*RRT/d^2 ) +
                                  3*ν/d^2*(dot(r,nx)*ny*transpose(r) + dot(nx,ny)*RRT) +
                                  (1-2*ν)*(3/d^2*dot(r,nx)*r*transpose(ny) + dot(nx,ny)*ID + ny*transpose(nx)) -
                                  (1-4ν)*nx*transpose(ny)
                                  )
    end
end

################################################################################
################################# Maxwell ######################################
################################################################################

"""
    Maxwell{T} <: AbstractPDE{3}

Normalized Maxwell's equation ∇ × ∇ × E - k² E = 0, where
k = ω √ϵμ.
"""
struct Maxwell{T} <: AbstractPDE{3}
    k::T
end

Maxwell(;dim=3,k::T) where {T}        = Maxwell{T}(k)

parameters(pde::Maxwell) = pde.k

function Base.show(io::IO,pde::Maxwell)
    # k = parameters(pde)
    print(io,"∇ × ∇ × E - k² E = 0")
end

default_kernel_eltype(::Maxwell)   = SMatrix{3,3,ComplexF64,9}
default_density_eltype(::Maxwell)  = SVector{3,ComplexF64}

# Single Layer kernel for Maxwell is the dyadic Greens function
function (SL::SingleLayerKernel{T,S})(target,source)::T  where {T,S<:Maxwell}
    k  = parameters(pde(SL))
    x  = coords(target)
    y  = coords(source)
    rvec = x - y
    r = norm(rvec)
    r == 0 && return zero(T)
    # helmholtz greens function
    g   = exp(im*k*r)/(4π*r)
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    G   = g*I + 1/k^2*(gp/r*I + (gpp/r^2 - gp/r^3)*RRT)
    # TODO: when multiplying by a density, it is faster to exploit the outer
    # product format isntead of actually assemblying the matrices.
    return G
end

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(target,source)::T where {T,S<:Maxwell}
    k  = parameters(pde(DL))
    x  = coords(target)
    y  = coords(source)
    ny = normal(source)
    rvec = x - y
    r      = norm(rvec)
    r == 0 && return zero(T)
    g      = exp(im*k*r)/(4π*r)
    gp     = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    ncross = cross_product_matrix(ny)
    return ncross * rcross * gp/r |> transpose
end

"""
    _curl_y_green_tensor_maxwell(x, y, k)

Returns `∇ʸ × G(x, y)` where `G` is the Green tensor for Maxwell's equations
with wavenumber `k`.
"""
function _curl_y_green_tensor_maxwell(x, y, k)
    rvec = x - y
    r    = norm(rvec)
    g    = exp(im*k*r)/(4π*r)
    gp   = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    curl_G = -gp/r*rcross
    return curl_G
end

"""
    _curl_x_green_tensor_maxwell(x, y, k)

Returns `∇ˣ × G(x, y)` where `G` is the Green tensor for Maxwell's equations
with wavenumber `k`.
"""
function _curl_x_green_tensor_maxwell(x, y, k)
    rvec = x - y
    r = norm(rvec)
    g    = exp(im*k*r)/(4π*r)
    gp  = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    curl_G = gp/r*rcross
    return curl_G
end
