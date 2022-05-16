"""
    struct Density{T,S} <: AbstractVector{T}

Discrete density with values `vals` on the quadrature nodes of `mesh::S`.
"""
struct Density{V,S<:NystromMesh} <: AbstractVector{V}
    vals::Vector{V}
    mesh::S
end

# AbstractArray interface
Base.size(σ::Density,args...)      = size(vals(σ),args...)
Base.getindex(σ::Density,args...)  = getindex(vals(σ),args...)
Base.setindex!(σ::Density,args...) = setindex!(vals(σ),args...)
Base.similar(σ::Density)           = Density(similar(vals(σ)),mesh(σ))

vals(σ::Density) = σ.vals
mesh(σ::Density) = σ.mesh

Density(etype::DataType,surf)  = Density(zeros(etype,length(dofs(surf))),surf)
Density(pde::AbstractPDE,surf) = Density(default_density_eltype(pde),surf)

function Density(f::Function,X)
    vals = [f(dof) for dof in dofs(X)]
    return Density(vals,X)
end

function γ₀(f,X)
    Density(x->f(coords(x)),X)
end

function γ₁(f,X)
    Density(dof->f(coords(dof),normal(dof)),X)
end

# overload some unary/binary operations for convenience
Base.:-(σ::Density) = Density(-vals(σ),mesh(σ))
Base.:+(σ::Density) = σ
Base.:*(a::Number,σ::Density) = Density(a*vals(σ),mesh(σ))
Base.:*(σ::Density,a::Number) = a*σ
Base.:/(σ::Density,a::Number) = Density(vals(σ)/a,mesh(σ))

# `Density` should be able to handle the following cases:
# mul!(y::Density{<:Number},A::AbstractMatrix{<:Number},x::Density{<:Number})
# mul!(y::Density{<:SVector},A::AbstractMatrix{<:SMatrix},x::Density{<:SVector})
# mul!(y::Density{<:SVector},A::AbstractMatrix{<:Number},x::Density{<:SVector})

function LinearAlgebra.mul!(::AbstractVector,
                            ::AbstractMatrix,
                            ::Density,
                            ::Number,::Number)
    notimplemented()
end
function LinearAlgebra.mul!(y::Density{<:Number},
                            A::AbstractMatrix{<:Number},
                            x::Density{<:Number},
                            a::Number,b::Number)
    _mymul!(vals(y),A,vals(x),a,b)
    return y
end
function LinearAlgebra.mul!(y::Density{<:SVector},
                            A::AbstractMatrixOrDiagonal{<:SMatrix},
                            x::Density{<:SVector},
                            a::Number,b::Number)
    _mymul!(vals(y),A,vals(x),a,b)
    return y
end
function LinearAlgebra.mul!(y::Density{<:SVector},
                            A::AbstractMatrix{<:Number},
                            x::Density{<:SVector},
                            a::Number,b::Number)
    
    yvals = reinterpret(eltype(eltype(y)),vals(y))
    xvals = reinterpret(eltype(eltype(y)),vals(x))
    _mymul!(yvals,A,xvals,a,b)
    return y
end

function Base.:*(A::AbstractMatrix{<:Number},x::Density{<:Number})
    # assume `y::Density{<:Number}` with same length as `x`
    @assert size(A,1) == size(A,2)
    y = similar(x)
    return mul!(y,A,x)
end
function Base.:*(A::AbstractMatrixOrDiagonal{<:SMatrix},x::Density{<:SVector})
    # assume `y::Density` with same length as `x`
    @assert size(A,1) == size(A,2)
    T = Base.promote_op(*, eltype(A), eltype(x))
    y = Density(zeros(T,length(x)),mesh(x))
    return mul!(y,A,x)
end
function Base.:*(A::AbstractMatrix{<:Number},
                 x::Density{<:SVector})
    # Infer the resulting Density eltype
    V = eltype(x)
    nqnodes,res = divrem(size(A,2),length(V)) # number of qnodes
    @assert nqnodes == length(x)
    @assert iszero(res)
    lengthT,res = divrem(size(A,1),nqnodes)
    @assert iszero(res)
    T = SVector{lengthT,eltype(V)}  # new Density eltype
    y = Density(zeros(T,nqnodes),mesh(x))
    return mul!(y,A,x)
end

Base.:\(A::AbstractMatrix{<:Number},σ::Density{<:Number}) = Density(A\vals(σ),mesh(σ))
function Base.:\(A::AbstractMatrix{<:Number},σ::Density{V}) where {V<:SVector}
    @assert size(A,1) == size(A,2)
    σvec = reinterpret(eltype(V),vals(σ))
    μlen,res = divrem(size(A,1),length(V))
    @assert res == 0
    μ = Density(zeros(V,μlen),mesh(σ))
    μvec = reinterpret(eltype(V),vals(μ))
    μvec[:] = A\σvec
    return μ
end

function IterativeSolvers.gmres!(σ::Density{V},A::AbstractMatrix{<:Number},μ::Density{V},args...;kwargs...) where V
    log = haskey(kwargs,:log) ? kwargs[:log] : false
    σvec = reinterpret(eltype(V),vals(σ))
    μvec = reinterpret(eltype(V),vals(μ))
    if log
        vals,hist = gmres!(σvec,A,μvec,args...;kwargs...)
        return σ,hist
    else
        vals = gmres!(σvec,A,μvec,args...;kwargs...)
        return σ
    end
end
IterativeSolvers.gmres(A,μ::Density,args...;kwargs...) = gmres!(zero(μ),A,μ,args...;kwargs...)

function ncross(σ::Density)
    Γ = mesh(σ)
    iter = zip(vals(σ),dofs(Γ))
    v = map(iter) do (v,dof)
        cross(normal(dof),v)
    end
    Density(v,Γ)
end

function istangential(σ::Density,tol=1e-8)
    Γ = mesh(σ)
    iter = zip(vals(σ),dofs(Γ))
    all(iter) do (v,dof)
        abs(dot(v,normal(dof))) < tol
    end
end

"""
    struct TangentialDensity{T,S} <: AbstractVector{T}

A density tangential to the surface defined by `mesh`. The `vals` field stores
the components on the tangential basis defined by the surface `jacobian` at each
`mesh` point.

!!! note
    Calling `TangentialDensity` on a `Density` object `σ` will compute `σ - (σ⋅n)n`,
    and express it using the the basis induced by the jacobian at each point.
    Unless `σ` is already a tangetial field, this is a projection and not only
    a change of basis.
"""
struct TangentialDensity{V,S<:NystromMesh} <: AbstractVector{V}
    vals::Vector{V}
    mesh::S
end

vals(σ::TangentialDensity) = σ.vals
mesh(σ::TangentialDensity) = σ.mesh

Base.size(σ::TangentialDensity,args...)     = size(vals(σ),args...)
Base.getindex(σ::TangentialDensity,args...) = getindex(vals(σ),args...)
Base.setindex!(σ::TangentialDensity,args...) = setindex!(vals(σ),args...)

function TangentialDensity(σ::Density)
    # original density type must be a 3D vector
    @assert eltype(vals(σ)) <: SVector{3}
    Γ = mesh(σ)
    @assert typeof(Γ|>dofs|>first|>jacobian) <: SMatrix{3,2}
    iter = zip(vals(σ),dofs(Γ))
    v = map(iter) do (σ,dof)
        jac = jacobian(dof)
        rhs = transpose(jac)*σ
        A   = transpose(jac)*jac
        A\rhs
    end
    TangentialDensity(v,Γ)
end

function ncross(σ::TangentialDensity)
    # original density type must be a 2D vector
    @assert eltype(vals(σ)) <: SVector{2}
    Γ = mesh(σ)
    @assert typeof(Γ|>dofs|>first|>jacobian) <: SMatrix{3,2}
    iter = zip(vals(σ),dofs(Γ))
    vlist = map(iter) do (v,dof)
        jac = jacobian(dof)
        t1 = jac[:,1]
        t2 = jac[:,2]
        # metric tensor coefficients
        E = dot(t1,t1)
        G = dot(t2,t2)
        F = dot(t1,t2)
        dS = sqrt(E*G - F^2)  # differential area
        SVector(-F*v[1] - G*v[2], E*v[1] + F*v[2]) / dS
    end
    return TangentialDensity(vlist,Γ)
end

function Density(σ::TangentialDensity)
    Γ = mesh(σ)
    iter = zip(vals(σ),dofs(Γ))
    v = map(iter) do (σ,dof)
        jac = jacobian(dof)
        jac*σ
    end
    Density(v,Γ)
end
