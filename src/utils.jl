"""
    enable_debug()

Activate debugging messages.
"""
function enable_debug()
    ENV["JULIA_DEBUG"] = Nystrom
end

error_green_formula(SL,DL,γ₀u,γ₁u,u,σ)                      = σ*u + SL*γ₁u  - DL*γ₀u
error_derivative_green_formula(ADL,H,γ₀u,γ₁u,un,σ)          = σ*un + ADL*γ₁u - H*γ₀u
error_interior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,-1/2)
error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,-1/2)
error_exterior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,1/2)
error_exterior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,1/2)

"""
    diagonalblockmatrix_to_matrix(A::Matrix{B}) where {B<:SMatrix}

Convert a diagonal block matrix `A::AbstractVector{B}`, where `A` is the list of diagonal blocks
and `B<:SMatrix`, to the equivalent `SparseMatrixCSC{T}`, where `T = eltype(B)`.
"""
function diagonalblockmatrix_to_matrix(A::AbstractVector{B}) where B<:SMatrix
    T = eltype(B) 
    sblock = size(B)
    ss = size(A) .* sblock  # matrix size when viewed as matrix over T
    I = Int64[]
    J = Int64[]
    V = T[]
    i_full, j_full = (1, 1)
    for subA in A
        i_tmp = i_full
        for j in 1:sblock[2]
            i_full = i_tmp
            for i in 1:sblock[1]
                push!(I, i_full)
                push!(J, j_full)
                push!(V, subA[i, j])
                i_full += 1
            end
            j_full += 1
        end
    end
    return sparse(I, J, V, ss[1], ss[2])
end
