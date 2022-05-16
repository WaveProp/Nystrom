"""
    struct Density{T,S} <: AbstractVector{T}

Discrete density with values `vals` on the quadrature nodes of `mesh::S`.
"""
struct Density{V,S<:NystromMesh} <: AbstractVector{V}
    vals::Vector{V}
    mesh::S
end

vals(σ::Density) = σ.vals
mesh(σ::Density) = σ.mesh

Base.size(σ::Density,args...)     = size(σ.vals,args...)
Base.getindex(σ::Density,args...) = getindex(σ.vals,args...)
Base.setindex(σ::Density,args...) = setindex(σ.vals,args...)

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

Base.zero(σ::Density) = Density(zero(σ.vals),mesh(σ))

# overload some unary/binary operations for convenience
Base.:-(σ::Density) = Density(-σ.vals,σ.mesh)
Base.:+(σ::Density) = σ
Base.:*(a::Number,σ::Density) = Density(a*σ.vals,σ.mesh)
Base.:*(σ::Density,a::Number) = a*σ
Base.:/(σ::Density,a::Number) = Density(σ.vals/a,σ.mesh)

Base.:*(A::AbstractMatrix{<:Number},σ::Density{<:Number}) = Density(A*σ.vals,σ.mesh)
function Base.:*(A::AbstractMatrix{<:Number},σ::Density{V}) where {V<:SVector}
    σvec = reinterpret(eltype(V),σ.vals)
    @assert size(A,2) % length(V) == 0
    μlen,res = divrem(size(A,1),length(V))
    @assert res == 0
    μ = Density(zeros(V,μlen),σ.mesh)
    μvec = reinterpret(eltype(V),μ.vals)
    mul!(μvec,A,σvec)
    return μ
end

Base.:\(A::AbstractMatrix{<:Number},σ::Density{<:Number}) = Density(A\σ.vals,σ.mesh)
function Base.:\(A::AbstractMatrix{<:Number},σ::Density{V}) where {V<:SVector}
    @assert size(A,1) == size(A,2)
    σvec = reinterpret(eltype(V),σ.vals)
    μlen,res = divrem(size(A,1),length(V))
    @assert res == 0
    μ = Density(zeros(V,μlen),σ.mesh)
    μvec = reinterpret(eltype(V),μ.vals)
    μvec[:] = A\σvec
    return μ
end

function IterativeSolvers.gmres!(σ::Density{V},A::AbstractMatrix{<:Number},μ::Density{V},args...;kwargs...) where V
    log = haskey(kwargs,:log) ? kwargs[:log] : false
    σvec = reinterpret(eltype(V),σ.vals)
    μvec = reinterpret(eltype(V),μ.vals)
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

Base.size(σ::TangentialDensity,args...)     = size(σ.vals,args...)
Base.getindex(σ::TangentialDensity,args...) = getindex(σ.vals,args...)
Base.setindex(σ::TangentialDensity,args...) = setindex(σ.vals,args...)

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
