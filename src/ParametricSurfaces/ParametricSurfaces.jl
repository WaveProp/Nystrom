module ParametricSurfaces

using StaticArrays
using LinearAlgebra
using ForwardDiff # for computing derivatives of parametric elements
using RecipesBase

import WavePropBase:
    HyperRectangle,
    AbstractEntity,
    AbstractElement,
    Domain,
    GenericMesh,
    ElementIterator,
    UniformCartesianMesh,
    geometric_dimension,
    ambient_dimension,
    new_tag,
    global_add_entity!,
    boundary,
    entities,
    clear_entities!,
    center,
    normal,
    assert_concrete_type,
    ReferenceSquare,
    ReferenceLine,
    ElementaryEntity,
    skeleton,
    internal_boundary,
    external_boundary,
    jacobian,
    mesh,
    domain,
    low_corner,
    high_corner

export
    # re-exported from WavePropBase for convenience
    clear_entities!,
    ElementaryEntity,
    Domain,
    skeleton,
    internal_boundary,
    external_boundary,
    HyperRectangle,
    ReferenceLine,
    ReferenceSquare,
    entities,
    boundary,
    geometric_dimension,
    ambient_dimension,
    jacobian,
    normal,
    #types
    ParametricEntity,
    ParametricElement,
    #functions
    line,
    meshgen

include("parametricentity.jl")
include("parametricelement.jl")
include("meshgen.jl")
include("simpleshapes.jl")

end # module
