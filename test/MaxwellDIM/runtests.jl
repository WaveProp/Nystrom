using Test
using SafeTestsets

@safetestset "Maxwell kernels" begin include("kernels_test.jl") end

@safetestset "Maxwell DIM" begin include("dim_test.jl") end