using Test
using SafeTestsets

@safetestset "Maxwell kernels" begin include("kernels_test.jl") end