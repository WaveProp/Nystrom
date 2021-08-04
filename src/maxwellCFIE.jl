"""
    MaxwellCFIE{T} <: Nystrom.AbstractPDE{3}

Normalized Maxwell's equation ∇ × ∇ × E - k² E = 0, where
k = ω √ϵμ.
"""
struct MaxwellCFIE{T} <: Nystrom.AbstractPDE{3}
    k::T   # wavenumber
end
MaxwellCFIE(;k) = MaxwellCFIE{ComplexF64}(k)
parameters(pde::MaxwellCFIE) = pde.k

function Base.show(io::IO,pde::MaxwellCFIE)
    # k = parameters(pde)
    print(io,"∇ × ∇ × E - k² E = 0")
end

default_kernel_eltype(::MaxwellCFIE)   = SMatrix{3,3,ComplexF64,9}
default_density_eltype(::MaxwellCFIE)  = SVector{3,ComplexF64}

# Single Layer kernel for Maxwell is the dyadic Greens function
function (SL::SingleLayerKernel{T,S})(target,source)::T where {T,S<:MaxwellCFIE}
    k = parameters(pde(SL))
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
    nx = cross_product_matrix(normal(target))
    return nx*G
end 

# Double Layer Kernel
function (DL::DoubleLayerKernel{T,S})(target,source)::T where {T,S<:MaxwellCFIE}
    k = parameters(pde(DL))
    x  = coords(target)
    y  = coords(source)
    rvec = x - y
    r      = norm(rvec)
    r == 0 && return zero(T)
    g      = exp(im*k*r)/(4π*r)
    gp     = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    nx = cross_product_matrix(normal(target))
    return nx * rcross * gp/r
end

function maxwell_green_tensor(target, source, k)
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
    return G
end

function maxwell_curl_green_tensor(target, source, k)
    x  = coords(target)
    y  = coords(source)
    rvec = x - y
    r      = norm(rvec)
    r == 0 && return zero(T)
    g      = exp(im*k*r)/(4π*r)
    gp     = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    return rcross * gp/r
end

"""
    single_layer_kernel(x, y, k, nx) 
    single_layer_kernel(x, y, k, nx, ϕy)  

Returns the single layer integral operator 
kernel `γ₀G = nₓ × G` or `γ₀(G*ϕy) = nₓ × G*ϕy`.
"""
function single_layer_kernel(x, y, k, nx)  
    G = maxwell_green_tensor(x, y, k)
    ncross = cross_product_matrix(nx)
    SL_kernel = ncross * G
    return SL_kernel
end
function single_layer_kernel(x, y, k, nx, ϕy)  
    Gϕy = maxwell_green_tensor(x, y, k)* ϕy
    SL_kernel = cross(nx, Gϕy)
    return SL_kernel
end

"""
    double_layer_kernel(x, y, k, nx) 
    double_layer_kernel(x, y, k, nx, ϕy)  

Returns the double layer integral operator 
kernel `γ₁G = nₓ × ∇ₓ × G` or `γ₁(G*ϕy) = nₓ × ∇ₓ × (G*ϕy)`.
"""
function double_layer_kernel(x, y, k, nx) 
    curl_G = maxwell_curl_green_tensor(x, y, k)
    ncross = cross_product_matrix(nx)
    DL_kernel = ncross * curl_G
    return DL_kernel
end
function double_layer_kernel(x, y, k, nx, ϕy) 
    curl_Gϕy = maxwell_curl_green_tensor(x, y, k)* ϕy
    DL_kernel = cross(nx, curl_Gϕy)
    return DL_kernel
end

function maxwellCFIE_SingleLayerPotencial(pde, mesh)
    k = parameters(pde)
    Γ = mesh
    function out(σ, x)
        iter = zip(dofs(Γ),σ)
        return sum(iter) do (source,σ)
            w = weight(source)
            maxwell_green_tensor(x, source, k)*σ*w
        end
    end
    return out
end
function maxwellCFIE_DoubleLayerPotencial(pde, mesh)
    k = parameters(pde)
    Γ = mesh
    function out(σ, x)
        iter = zip(dofs(Γ),σ)
        return sum(iter) do (source,σ)
            w = weight(source)
            maxwell_curl_green_tensor(x, source, k)*σ*w
        end
    end
    return out
end