"""
    ParametricEntity <: AbstractEntity

A geometric entity given by an explicit `parametrization` field. The valid
arguments should lie in its `domain`. This differs from an `ElementaryEntity` in
that the underlying representation of the geometric entity is available.
"""
struct ParametricEntity <: AbstractEntity
    dim::UInt8
    tag::Int
    parametrization
    domain
    boundary::Vector{<:AbstractEntity}
    function ParametricEntity(dim, tag, f, d)
        ent = new(dim, tag, f, d,AbstractEntity[])
        # every entity gets added to a global variable ENTITIES so that we can
        # ensure the (d,t) pair is a UUID for an entity, and to easily retrieve
        # different entities.
        global_add_entity!(ent)
        return ent
    end
end

domain(p::ParametricEntity)              = p.domain
parametrization(p::ParametricEntity)     = p.parametrization
geometric_dimension(p::ParametricEntity) = p.dim
tag(p::ParametricEntity) = p.tag
boundary(p::ParametricEntity) = p.boundary

function flip_normal(ent::ParametricEntity)
    @assert ambient_dimension(ent) == geometric_dimension(ent) + 1
    ParametricEntity(geometric_dimension(ent),-tag(ent),parametrization(ent),domain(ent))
end

function ParametricEntity(f,dom)
    d = geometric_dimension(dom)
    t = new_tag(d) # automatically generate a new (valid) tag
    ParametricEntity(d, t, f, dom)
end

function return_type(p::ParametricEntity)
    # NOTE: this relies on the brittle promote_op
    d = domain(p)
    x = center(d)
    f = parametrization(p)
    T = Base.promote_op(f, typeof(x))
    return T
end

ambient_dimension(p::ParametricEntity) = length(return_type(p))

function (par::ParametricEntity)(x)
    @assert x in domain(par)
    par.parametrization(x)
end

jacobian(psurf::ParametricEntity,s::SVector)  = ForwardDiff.jacobian(psurf.parametrization, s::SVector)
jacobian(psurf::ParametricEntity,s)           = jacobian(psurf, SVector(s))

"""
    line(a,b)

Create a straight line connecting points `a` and `b`. Returns an instance
of [`ParametricEntity`](@ref).
"""
function line(a::SVector,b::SVector)
    f = (u) -> a + u[1]*(b-a)
    d = HyperRectangle(0.,1.)
    ParametricEntity(f,d)
end
line(a,b) = line(SVector(a),SVector(b))
