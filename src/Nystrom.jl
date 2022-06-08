module Nystrom

using StaticArrays
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using SpecialFunctions
using QuadGK
using RecipesBase
using TimerOutputs
using Base.Threads

import WavePropBase
import Lebedev  # to obtain the Lebedev points

# module to handle some simple parametric geometries
include("ParametricSurfaces/ParametricSurfaces.jl")
using .ParametricSurfaces

# base utilities
import WavePropBase:
    HyperRectangle,
    AbstractMesh,
    AbstractQuadratureRule,
    AbstractElement,
    AbstractEntity,
    GenericMesh,
    ElementIterator,
    SubMesh,
    entities,
    integrate,
    ambient_dimension,
    geometric_dimension,
    domain,
    qrule_for_reference_shape,
    low_corner,
    high_corner,
    jacobian,
    integration_measure,
    clear_entities!,
    normal,
    coords,
    center,
    diameter,
    blockmatrix_to_matrix,
    cross_product_matrix,
    assert_concrete_type


# Interpolated Factored Green Function Method
import IFGF

export
    # re-exported from WavePropBase
    clear_entities!,
    Domain,
    boundary,
    #
    ParametricSurfaces,
    Laplace,
    Helmholtz,
    Elastostatic,
    Maxwell,
    NystromMesh,
    NystromDOF,
    SingleLayerKernel,
    DoubleLayerKernel,
    AdjointDoubleLayerKernel,
    HyperSingularKernel,
    SingleLayerPotential,
    DoubleLayerPotential,
    SingleLayerOperator,
    DoubleLayerOperator,
    AdjointDoubleLayerOperator,
    HyperSingularOperator,
    Density,
    TangentialDensity,
    # functions
    qweights,
    trace,
    ncross,
    coords,
    qcoords,
    γ₀,
    γ₁,
    ifgf_compressor

include("utils.jl")
include("nystrommesh.jl")
include("kernels.jl")
include("blockindexer.jl")
include("density.jl")
include("potential.jl")
include("integraloperator.jl")
include("lebedevpoints.jl")
include("dim.jl")
include("gausskronrod.jl")
include("discreteoperator.jl")
include("ifgf.jl")

# module that extends the DIM method to the Maxwell's equations
include("MaxwellDIM/MaxwellDIM.jl")

end # module
