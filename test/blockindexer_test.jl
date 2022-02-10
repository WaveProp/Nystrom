using Test
using StaticArrays
using Random
using Nystrom
using LinearAlgebra
Random.seed!(1)

@testset "BlockIndexer test" begin
    n,m = 30,25
    nblock,mblock = 3,5
    @test (n,m) .% (nblock,mblock) == (0,0)
    A = rand(n,m)
    T = SMatrix{nblock,mblock,Float64,nblock*mblock}
    Ablock = Nystrom.BlockIndexer(A,T)
    @test Matrix(Ablock) === A
    @test Nystrom.BlockIndexer(A,eltype(A)) === A
    @test size(Ablock) == (n,m) .÷ (nblock,mblock)
    @test size(Ablock,1) == n ÷ nblock
    @test size(Ablock,2) == m ÷ mblock 
    @test Matrix(Ablock) === Matrix(copy(Ablock))
    for i in 1:nblock
        for j in 1:mblock
            @test Ablock[i,j] == A[((i-1)*nblock+1):(i*nblock), ((j-1)*mblock+1):(j*mblock)]
        end
    end
    c = rand(T)
    Ablock[1,1] = c
    @test Ablock[1,1] == c
    Ablock[:,end] = @view Ablock[:,1]
    @test @view(Ablock[:,end]) == @view(Ablock[:,1])
    Ablock[end,:] = @view Ablock[1,:]
    @test @view(Ablock[end,:]) == @view(Ablock[1,:])

    S = SMatrix{nblock,mblock,ComplexF64,nblock*mblock}
    B,Bblock = Nystrom.MatrixAndBlockIndexer(S,((n,m).÷(nblock,mblock))...)
    @test size(B) == (n,m)
    @test size(Bblock) == (n,m) .÷ (nblock,mblock)
    @test B isa Matrix{eltype(S)}
    C,Cblock = Nystrom.MatrixAndBlockIndexer(eltype(S),n,m)
    @test size(C) == (n,m)
    @test C===Cblock
    @test C isa Matrix{eltype(S)}
end