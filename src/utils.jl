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
    @hprofile

A macro which
- resets the default `TimerOutputs.get_defaulttimer` to zero
- executes the code block
- prints the profiling details

This is useful as a coarse-grained profiling strategy to get a rough idea of
where time is spent. Note that this relies on `TimerOutputs` annotations
manually inserted in the code.
"""
macro hprofile(block)
    return quote
        TimerOutputs.enable_debug_timings(Nystrom)
        reset_timer!()
        $(esc(block))
        print_timer()
    end
end

###
# `Array`s of `SArray` bug fixes
###
# The following functions are intended to fix some bugs
# in Julia that happen when `Matrix{<:SMatrix}` are operated
# with `Vector{<:SVector}`, or some similar combination.

# convenient aliases
const AbstractMatrixOrDiagonal = Union{AbstractMatrix{T},
                                       Diagonal{T}} where T
const MatrixOrReinterpretMatrix = Union{Matrix{T},
                                        Base.ReinterpretArray{T,D,V,<:Matrix}} where {T,D,V}

"""
    _mymul!(y,A,x,a,b)

Same as `LinearAlgebra.mul!`, but fixes some bugs that
occur when the arguments are `Array`s of `SArray`s 
(e.g. `y::Vector{<:SVector}`, `A::Matrix{<:SMatrix}` and `x::Vector{<:SVector}`)
"""
_mymul!(y,A,x,a,b) = mul!(y,A,x,a,b)
function _mymul!(y::AbstractVector{T},
                 A::Matrix{<:SMatrix},
                 x::AbstractVector{<:SVector},a,b) where {T<:SVector}
    # y.=a*A*x+b*y
    # implement a hand-written loop
    @assert size(A) == (length(y),length(x))
    ytemp = zero(T)
    for i in 1:size(A,1)
        ytemp = zero(T)
        for j in 1:size(A,2)
            ytemp += A[i,j]*x[j]
        end
        y[i] = a*ytemp + b*y[i]
    end
    return y
end
function _mymul!(y::AbstractVector{<:SVector},
                 D::Diagonal{<:SMatrix},
                 x::AbstractVector{<:SVector},a,b)
    # y.=a*D*x+b*y
    # implement a hand-written loop
    @assert size(D) == (length(y),length(x))
    d = diag(D)
    for i in eachindex(d)
        y[i] = a*d[i]*x[i]+b*y[i]
    end
    return y
end
function _mymul!(y::AbstractVector{<:SVector},
                 D::UniformScaling,
                 x::AbstractVector{<:SVector},a,b)
    @. y = a*D.λ*x + b*y
    return y
end

"""
    _mymul(A,B)
    
Same as `*(A,B)`, but fixes a bug that
occurs when an argument is a `Diagonal{<:SMatrix}` and
the other is a `MatrixOrReinterpretMatrix{<:SArray}`.
"""
_mymul(A,B)                                     = A*B
_mymul(A::Diagonal{<:SMatrix},
       B::MatrixOrReinterpretMatrix{<:SArray})  = A.diag.*B
_mymul(A::MatrixOrReinterpretMatrix{<:SMatrix},
       B::Diagonal{<:SMatrix})                  = A.*permutedims(B.diag)
