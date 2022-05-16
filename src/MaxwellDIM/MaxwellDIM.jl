module MaxwellDIM

using StaticArrays
using LinearAlgebra
using Nystrom
using SparseArrays

export 
    # types
    TrueMaxwell,
    # functions
    EFIEPotential,
    MFIEPotential,
    EFIEOperator,
    MFIEOperator

include("kernels.jl")
include("dim.jl")

end