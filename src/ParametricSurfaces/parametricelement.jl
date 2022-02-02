"""
    ParametricElement <: AbstractElement{D,T}

An element represented through a explicit function `f` mapping `D` into the
element.

The underlying implementation uses an affine map from `D = domain(el)` into a
`preimage::HyperRectangle`, and then uses `f` on this `preimage`. This is done
to avoid having to create a new function `f` (and therefore a new type) when
parametric elements are split. This is an implementation detail, and the
underlying usage is identical to e.g. a `LagrangeElement`.
"""
struct ParametricElement{D,T,F,R} <: AbstractElement{R,T}
    parametrization::F
    preimage::D
    function ParametricElement{D,T,F}(f::F,d::D) where {D,T,F}
        if D <: HyperRectangle{1}
            R = ReferenceLine
        elseif D <: HyperRectangle{2}
            R = ReferenceSquare
        else
            notimplemented()
        end
        return new{D,T,F,R}(f,d)
    end
end

preimage(el::ParametricElement)        = el.preimage
parametrization(el::ParametricElement) = el.parametrization

domain(::ParametricElement{D,T,F,R}) where {D,T,F,R} = R()

return_type(p::ParametricElement{D,T,F}) where {D,T,F} = T

geometric_dimension(p::ParametricElement) = geometric_dimension(domain(p))
ambient_dimension(p::ParametricElement)   = length(return_type(p))

# constructor which infers the return type of f. To be on the safe side, error
# if inferred type is not concrete
function ParametricElement(f,d)
    x = center(d)
    T = Base.promote_op(f,typeof(x))
    assert_concrete_type(T)
    isbitstype(T) || (@warn "non bitstype detected for ParametricElement")
    D = typeof(d)
    F = typeof(f)
    return ParametricElement{D,T,F}(f,d)
end

function (el::ParametricElement)(u)
    @assert u ∈ domain(el)
    rec = preimage(el)
    lc  = low_corner(rec)
    hc  = high_corner(rec)
    # map from reference domain to preimage
    v = @. lc + (hc - lc)*u
    # map from preimage to element
    f = parametrization(el)
    return f(v)
end

function jacobian(el::ParametricElement,u::SVector)
    @assert u ∈ domain(el)
    rec = preimage(el)
    lc  = low_corner(rec)
    hc  = high_corner(rec)
    # map from reference domain to preimage
    scal = hc .- lc
    v    = @. lc + scal*u
    # compute jacobian
    f = parametrization(el)
    jacobian(f,v) * SDiagonal(scal)
end
jacobian(f::Function,v) = ForwardDiff.jacobian(f,v)
jacobian(psurf::ParametricElement,s) = jacobian(psurf,SVector(s))

# higher order derivatives used in some Nystrom methods for lines
function derivative(l::ParametricElement,s)
    @assert domain(l) == ReferenceLine()
    ForwardDiff.derivative(l,s[1])
end
derivative2(l::ParametricElement,s) = ForwardDiff.derivative(s -> derivative(l,s[1]),s[1])
derivative3(l::ParametricElement,s) = ForwardDiff.derivative(s -> derivative2(l,s[1]),s[1])

# Plot recipes
@recipe function f(el::ParametricElement;npts=10)
    D = el.preimage
    grid   --> false
    aspect_ratio --> :equal
    label --> ""
    if D isa HyperRectangle{1}
        s       = LinRange(0,1,npts)
        pts     = [el(v) for v in s]
        x       = [pt[1] for pt in pts]
        y       = [pt[2] for pt in pts]
        x,y
    elseif D isa HyperRectangle{2}
        seriestype := :surface
        xrange = LinRange(0,1,npts)
        yrange = LinRange(0,1,npts)
        pts    = [el((x,y)) for x in xrange, y in yrange]
        x      =  [pt[1] for pt in pts]
        y      =  [pt[2] for pt in pts]
        z      =  [pt[3] for pt in pts]
        x,y,z
    else
        notimplemented()
    end
end
