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
    A = rand(ComplexF64,n,m)
    T = SMatrix{nblock,mblock,ComplexF64,nblock*mblock}
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

@testset "blocksparse test" begin
    n,m = 30,25
    nblock,mblock = 3,5
    @test (n,m) .% (nblock,mblock) == (0,0)
    A = rand(ComplexF64,n,m)
    T = SMatrix{nblock,mblock,ComplexF64,nblock*mblock}
    Ablock = Nystrom.BlockIndexer(A,T)

    entries = 5
    I = rand(1:size(Ablock,1),entries)
    J = rand(1:size(Ablock,2),entries)
    V = [Ablock[i,j] for (i,j) in zip(I,J)]
    Asparse = Nystrom.blocksparse(I,J,V,size(Ablock)...)
    @test Asparse isa SparseMatrixCSC{eltype(A)}
    @test nnz(Asparse) == entries*length(eltype(Ablock))
    mask = Asparse.!=0
    @test Asparse[mask] == A[mask]

    # Scalar case
    I = rand(1:size(A,1),entries)
    J = rand(1:size(A,2),entries)
    V = [A[i,j] for (i,j) in zip(I,J)]
    A1 = Nystrom.blocksparse(I,J,V,size(A)...)
    A2 = sparse(I,J,V,size(A)...)
    @test A1 == A2
end