using SafeTestsets

@safetestset "Nystrom mesh" begin include("nystrommesh_test.jl") end

@safetestset "Kernels" begin include("kernels_test.jl") end

@safetestset "Density" begin include("density_test.jl") end

@safetestset "Potentials" begin include("potential_test.jl") end

@safetestset "Integral operators" begin include("integraloperator_test.jl") end

@safetestset "Density interpolation method" begin include("dim_test.jl") end

@safetestset "Discrete operators" begin include("discreteoperator_test.jl") end

# @safetestset "IFGF" begin include("ifgf_test.jl") end
