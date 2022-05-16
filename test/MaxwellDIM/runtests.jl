using Test
using SafeTestsets

@safetestset "Maxwell kernels" begin include("kernels_test.jl") end

@safetestset "Maxwell DIM" begin include("dim_test.jl") end

@safetestset "Utils" begin include("utils_test.jl") end

@safetestset "IFGF" begin include("ifgf_test.jl") end
