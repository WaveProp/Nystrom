module MaxwellDIM

using StaticArrays
using LinearAlgebra
using Nystrom
using SparseArrays
import Nystrom.IFGF

export 
    # types
    Maxwell,
    # functions
    EFIEPotential,
    MFIEPotential,
    EFIEOperator,
    MFIEOperator,
    diagonal_ncross_and_jacobian_matrices,
    ifgf_compressor

include("kernels.jl")
include("dim.jl")
include("utils.jl")
include("ifgf.jl")

end