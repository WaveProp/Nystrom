using Test
using SafeTestsets

@safetestset "Parametric entity" begin include("parametricentity_test.jl") end

@safetestset "Meshgen" begin include("meshgen_test.jl") end
