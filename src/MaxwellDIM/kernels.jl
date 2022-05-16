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

EFIEPotential(op::Maxwell,surf) = Nystrom.IntegralPotential(EFIEPotentialKernel(op),surf)
MFIEPotential(op::Maxwell,surf) = Nystrom.IntegralPotential(MFIEPotentialKernel(op),surf)

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

EFIEOperator(op::Maxwell,X,Y=X) = Nystrom.IntegralOperator(EFIEKernel(op), X, Y)
MFIEOperator(op::Maxwell,X,Y=X) = Nystrom.IntegralOperator(MFIEKernel(op), X, Y)

###
# Extras
###
function diagonal_ncross_and_jacobian_matrices(nmesh)
    qnodes = Nystrom.dofs(nmesh)
    n_qnodes = length(qnodes)
    # construct diagonal matrices as sparse arrays using BlockSparseConstructor
    Tn = qnodes |> first |> Nystrom.normal |> Nystrom.cross_product_matrix |> typeof
    Tj = qnodes |> first |> Nystrom.jacobian |> typeof
    Td = SMatrix{2,3,Float64,6}  # TODO: remove harcoded type
    nblock = Nystrom.BlockSparseConstructor(Tn,n_qnodes,n_qnodes)
    jblock = Nystrom.BlockSparseConstructor(Tj,n_qnodes,n_qnodes)
    dblock = Nystrom.BlockSparseConstructor(Td,n_qnodes,n_qnodes)
    for i in 1:n_qnodes
        q = qnodes[i]
        n = Nystrom.cross_product_matrix(normal(q))
        j = Nystrom.jacobian(q)
        d = Td(pinv(j))
        Nystrom.addentry!(nblock,i,i,n)
        Nystrom.addentry!(jblock,i,i,j)
        Nystrom.addentry!(dblock,i,i,d)
    end
    return sparse(nblock), sparse(jblock), sparse(dblock)
end
diagonal_ncross_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[1]
diagonal_jacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[2]
diagonal_dualjacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[3]




