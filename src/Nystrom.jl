module Nystrom

using StaticArrays: minimum
using StaticArrays
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using SpecialFunctions
using QuadGK
using RecipesBase
using BlockArrays
using MieSeries

# packages to generate the geometry
using GmshSDK
using ParametricSurfaces

# base utilities
using WavePropBase
using WavePropBase.Utils
using WavePropBase.Geometry
using WavePropBase.Interpolation
using WavePropBase.Integration
using WavePropBase.Mesh
using WavePropBase.Simulation

WavePropBase.@import_interface

export
    # re-export useful stuff
    Geometry,
    GmshSDK,
    ParametricSurfaces,
    Laplace,
    Helmholtz,
    Elastostatic,
    Maxwell,
    # types
    NystromMesh,
    NystromDOF,
    SingleLayerKernel,
    DoubleLayerKernel,
    AdjointDoubleLayerKernel,
    HyperSingularKernel,
    SingleLayerPotential,
    DoubleLayerPotential,
    MaxwellEFIEPotential,
    MaxwellMFIEPotential,
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
    γ₁

WavePropBase.@export_interface

include("utils.jl")
include("nystrommesh.jl")
include("kernels.jl")
include("potential.jl")
include("integraloperator.jl")
include("lebedevpoints.jl")
include("blockmatrices.jl")
include("dim.jl")
include("gausskronrod.jl")
include("density.jl")
include("maxwellwrappers.jl")
include("maxwellCFIE.jl")
include("maxwellCFIE_dim.jl")

end # module
