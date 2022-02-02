module ParametricSurfaces

using StaticArrays
using LinearAlgebra
using ForwardDiff # for computing derivatives of parametric elements
using RecipesBase

using WavePropBase
using WavePropBase.Utils
using WavePropBase.Geometry
using WavePropBase.Interpolation
using WavePropBase.Mesh

WavePropBase.@import_interface

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
