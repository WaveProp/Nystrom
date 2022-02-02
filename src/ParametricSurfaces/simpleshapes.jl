####################################################################################
# Two-dimensional shapes with parametric boundary
####################################################################################

# FIXME: this should probably be called a disk
struct Circle <: AbstractEntity
    # dim = 2
    tag::Int
    center::SVector{2,Float64}
    radius::Float64
    boundary::Vector{ParametricEntity}
    function Circle(t,c,r,bnd)
        ent = new(t,c,r,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Circle(;center=(0, 0),radius=1)
    f          = (s) -> center .+ radius .* SVector(cospi(2 * s[1]), sinpi(2 * s[1]))
    domain     = HyperRectangle(0,1)
    ent        = ParametricEntity(f, domain)
    tag        = new_tag(2) # generate a unique tag for entities of dimension 2
    return Circle(tag,center, radius, [ent])
end
Base.in(pt,circ::Circle) = norm(pt .- circ.center) < circ.radius
key(ent::Circle) = (2,ent.tag)
geometric_dimension(ent::Circle) = 2
ambient_dimension(ent::Circle) = 2

struct Kite <: AbstractEntity
    # dim = 2
    tag::Int
    center::SVector{2,Float64}
    radius::Float64
    boundary::Vector{ParametricEntity}
    function Kite(t,c,r,bnd)
        ent = new(t,c,r,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Kite(;radius=1,center=(0, 0))
    f = (s) -> center .+ radius .* SVector(cospi(2 * s[1]) + 0.65 * cospi(4 * s[1]) - 0.65,
                              1.5 * sinpi(2 * s[1]))
    domain = HyperRectangle(0,1)
    surf   = ParametricEntity(f, domain)
    t      = new_tag(2)
    return Kite(t,center, radius, [surf])
end
key(ent::Kite) = (2,ent.tag)
geometric_dimension(ent::Kite) = 2

struct Droplet <: AbstractEntity
    # dim = 2
    tag::Int
    center::SVector{2,Float64}
    radius::Float64
    boundary::Vector{ParametricEntity}
    function Droplet(t,c,r,bnd)
        ent = new(t,c,r,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Droplet(;radius=1,center=(0,0))
    f = (s) -> center .+ radius .* SVector(2 * sinpi(s[1]),-sinpi(2*s[1]))
    domain = HyperRectangle(0,1)
    surf   = ParametricEntity(f, domain)
    t      = new_tag(2)
    return Droplet(t,center, radius, [surf])
end
key(ent::Droplet) = (2,ent.tag)
geometric_dimension(ent::Droplet) = 2

struct Boomerang <: AbstractEntity
    # dim = 2
    tag::Int
    center::SVector{2,Float64}
    radius::Float64
    boundary::Vector{ParametricEntity}
    function Boomerang(t,c,r,bnd)
        ent = new(t,c,r,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Boomerang(;radius=1,center=(0,0))
    f = (s) -> center .+ radius .* SVector(-2/3 * sinpi(3*s[1]),-sinpi(2*s[1]))
    domain = HyperRectangle(0,1)
    surf   = ParametricEntity(f, domain)
    t      = new_tag(2)
    return Droplet(t,center, radius, [surf])
end
geometric_dimension(ent::Boomerang) = 2

####################################################################################
# Three-dimensional shapes with parametric boundary
####################################################################################

# TODO: rename to Ball?
struct Sphere <: AbstractEntity
    # dim = 3
    tag::Int
    center::SVector{3,Float64}
    radius::Float64
    boundary::Vector{ParametricEntity}
    function Sphere(t,c,r,bnd)
        ent = new(t,c,r,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Sphere(;center=(0,0,0),radius=0.5)
    nparts = 6
    domain = HyperRectangle((-1,-1),(1,1))
    parts  = Vector{ParametricEntity}(undef,nparts)
    for id=1:nparts
        param     = (x) -> _sphere_parametrization(x[1],x[2],id,radius,center)
        parts[id] = ParametricEntity(param,domain)
    end
    tag = new_tag(3)
    return Sphere(tag,center,radius,parts)
end
Base.in(pt,sph::Sphere) = norm(pt .- sph.center) < sph.radius
geometric_dimension(ent::Sphere) = 3
ambient_dimension(ent::Sphere) = 3

struct Ellipsoid <: AbstractEntity
    tag::Int
    center::SVector{3,Float64}
    paxis::SVector{3,Float64}
    boundary::Vector{ParametricEntity}
    function Ellipsoid(t,c,paxis,bnd)
        ent = new(t,c,paxis,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Ellipsoid(;center=zeros(3),paxis=ones(3))
    nparts = 6
    domain = HyperRectangle((-1.,-1.),(1.,1.))
    parts  = Vector{ParametricEntity}(undef,nparts)
    for id=1:nparts
        param     = (x) -> _ellipsoid_parametrization(x[1],x[2],id,paxis,center)
        parts[id] = ParametricEntity(param,domain,[domain])
    end
    return Ellipsoid(center,paxis,parts)
end


struct Bean <: AbstractEntity
    # dim = 3
    tag::Int
    center::SVector{3,Float64}
    paxis::SVector{3,Float64}
    boundary::Vector{ParametricEntity}
    function Bean(t,c,paxis,bnd)
        ent = new(t,c,paxis,bnd)
        global_add_entity!(ent)
        return ent
    end
end
geometric_dimension(ent::Bean) = 3
ambient_dimension(ent::Bean)   = 3

function Bean(;center=(0,0,0),paxis=(1,1,1))
    nparts = 6
    domain = HyperRectangle((-1,-1),(1,1))
    parts  = Vector{ParametricEntity}(undef,nparts)
    for id=1:nparts
        param     = (x) -> _bean_parametrization(x[1],x[2],id,paxis,center)
        parts[id] = ParametricEntity(param,domain)
    end
    tag = new_tag(3)
    return Bean(tag,center,paxis,parts)
end

struct Cube <: AbstractEntity
    # dim = 3
    tag::Int
    low_corner::SVector{3,Float64}
    widths::SVector{3,Float64}
    boundary::Vector{ParametricEntity}
    function Cube(t,c,p,bnd)
        ent = new(t,c,p,bnd)
        global_add_entity!(ent)
        return ent
    end
end

function Cube(;center=SVector(0,0,0),widths=(2,2,2))
    nparts = 6
    domain = HyperRectangle((-1.,-1.),(1.,1.))
    parts  = Vector{ParametricEntity}(undef,nparts)
    for id=1:nparts
        param     = (x) -> _cube_parametrization(x[1],x[2],id,widths,center)
        parts[id] = ParametricEntity(param,domain)
    end
    tag = new_tag(3)
    return Cube(tag,center,widths,parts)
end
geometric_dimension(ent::Cube) = 3
ambient_dimension(ent::Cube) = 3

function _cube_parametrization(u,v,id,widths,center)
    @assert -1 ≤ u ≤ 1
    @assert -1 ≤ v ≤ 1
    if id==1
        x = SVector(1.,u,v)
    elseif id==2
        x = SVector(-u,1.,v)
    elseif id==3
        x = SVector(u,v,1.)
    elseif id==4
        x = SVector(-1.,-u,v)
    elseif id==5
        x = SVector(u,-1.,v)
    elseif id==6
        x = SVector(-u,v,-1.)
    end
    return @. center + widths/2 * x
end

function _sphere_parametrization(u,v,id,rad=1,center=SVector(0,0,0))
    @assert -1 ≤ u ≤ 1
    @assert -1 ≤ v ≤ 1
    # parametrization of 6 patches. First gets a point on the cube [-1,1] ×
    # [-1,1] × [-1,1], the maps it onto the sphere
    if id==1
        x = SVector(1.,u,v)
    elseif id==2
        x = SVector(-u,1.,v)
    elseif id==3
        x = SVector(u,v,1.)
    elseif id==4
        x = SVector(-1.,-u,v)
    elseif id==5
        x = SVector(u,-1.,v)
    elseif id==6
        x = SVector(-u,v,-1.)
    end
    return center .+ rad.*x./sqrt(u^2+v^2+1)
end

function _ellipsoid_parametrization(u,v,id,paxis,center)
    x = _sphere_parametrization(u,v,id)
    return x .* paxis .+ center
end

function _bean_parametrization(u,v,id,paxis,center)
    x̂ = _sphere_parametrization(u,v,id)
    a = 0.8; b = 0.8; alpha1 = 0.3; alpha2 = 0.4; alpha3=0.1
    x = SVector(
        a*sqrt(1.0-alpha3*cospi(x̂[3])).*x̂[1],
        -alpha1*cospi(x̂[3])+b*sqrt(1.0-alpha2*cospi(x̂[3])).*x̂[2],
        x̂[3])
    return x .* paxis .+ center
end

function _acorn_parametrization(u,v,id,radius,center,rot)
    Rx = @SMatrix [1 0 0;0 cos(rot[1]) sin(rot[1]);0 -sin(rot[1]) cos(rot[1])]
    Ry = @SMatrix [cos(rot[2]) 0 -sin(rot[2]);0 1 0;sin(rot[2]) 0 cos(rot[2])]
    Rz = @SMatrix [cos(rot[3]) sin(rot[3]) 0;-sin(rot[3]) cos(rot[3]) 0;0 0 1]
    R  = Rz*Ry*Rx;
    x = _sphere_parametrization(u,v,id)
    th,phi,_ = cart2sph(x...)
    r=0.6+sqrt(4.25+2*cos(3*(phi+pi/2)))
    x[1] = r.*cos(th).*cos(phi)
    x[2] = r.*sin(th).*cos(phi)
    x[3] = r.*sin(phi)
    x    = R*x
    return radius.*x .+ center
end

function _cushion_parametrization(u,v,id,radius,center,rot)
    Rx = @SMatrix [1 0 0;0 cos(rot[1]) sin(rot[1]);0 -sin(rot[1]) cos(rot[1])]
    Ry = @SMatrix [cos(rot[2]) 0 -sin(rot[2]);0 1 0;sin(rot[2]) 0 cos(rot[2])]
    Rz = @SMatrix [cos(rot[3]) sin(rot[3]) 0;-sin(rot[3]) cos(rot[3]) 0;0 0 1]
    R  = Rz*Ry*Rx;
    x = _sphere_parametrization(u,v,id)
    th,phi,_ = cart2sph(x...)
    r = sqrt(0.8+0.5*(cos(2*th)-1).*(cos(4*phi)-1));
    x[1] = r.*cos(th).*cos(phi)
    x[2] = r.*sin(th).*cos(phi)
    x[3] = r.*sin(phi)
    x    = R*x
    return radius.*x .+ center
end
