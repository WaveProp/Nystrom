module MaxwellDIM

using StaticArrays
using LinearAlgebra
using Nystrom

export 
    # types
    TrueMaxwell,
    # functions
    EFIEPotential,
    MFIEPotential,
    EFIEOperator,
    MFIEOperator

include("kernel.jl")
include("dim.jl")

end