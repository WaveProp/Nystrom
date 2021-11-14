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
using WavePropBase.Trees
WavePropBase.@import_interface

# Interpolated Factored Green Function Method
import IFGF
import IFGF: IFGFOperator, centered_factor

export
    # Gmsh related stuff
    GmshSDK,
    @gmsh,
    gmsh,
    # re-export useful stuff
    Geometry,
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
    IFGFCompressor

WavePropBase.@export_interface

include("utils.jl")
include("nystrommesh.jl")
include("kernels.jl")
include("density.jl")
include("potential.jl")
include("integraloperator.jl")
include("lebedevpoints.jl")
include("dim.jl")
include("gausskronrod.jl")
include("maxwellwrappers.jl")
include("discreteoperator.jl")
include("ifgf.jl")

end # module
