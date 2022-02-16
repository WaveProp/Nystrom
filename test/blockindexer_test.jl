using Test
using StaticArrays
using Random
using Nystrom
using LinearAlgebra
using Nystrom.SparseArrays
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

@testset "BlockSparseConstructor test" begin
    n,m = 30,25
    nblock,mblock = 3,5
    @test (n,m) .% (nblock,mblock) == (0,0)
    A = rand(ComplexF64,n,m)
    T = SMatrix{nblock,mblock,ComplexF64,nblock*mblock}
    Ablock = Nystrom.BlockIndexer(A,T)

    entries = 5
    AC = Nystrom.BlockSparseConstructor(T,size(Ablock)...)
    @test size(AC) == size(Ablock)
    @test eltype(AC) == eltype(Ablock)
    prev_entries = []
    while length(prev_entries) < entries
        i = rand(1:size(AC,1))
        j = rand(1:size(AC,2))
        (i,j) ∈ prev_entries && continue
        push!(prev_entries, (i,j))
        v = Ablock[i,j]
        Nystrom.addentry!(AC,i,j,v)
    end
    @test length(AC.I) == length(AC.J) == length(AC.V) == entries*length(T)
    Asparse = sparse(AC)
    @test Asparse isa SparseMatrixCSC{eltype(A)}
    @test nnz(Asparse) == entries*length(T)
    mask = Asparse.!=0
    @test Asparse[mask] == A[mask]

    AC2 = Nystrom.BlockSparseConstructor(T,size(Ablock)...)
    I = [i for (i,_) in prev_entries]
    J = [j for (_,j) in prev_entries]
    V = [Ablock[i,j] for (i,j) in prev_entries]
    Nystrom.addentries!(AC2,I,J,V)
    @test Asparse == sparse(AC2)

    # Scalar case
    AC = Nystrom.BlockSparseConstructor(eltype(A),size(A)...)
    @test size(AC) == size(A)
    @test eltype(AC) == eltype(A)
    prev_entries = []
    while length(prev_entries) < entries
        i = rand(1:size(AC,1))
        j = rand(1:size(AC,2))
        (i,j) ∈ prev_entries && continue
        push!(prev_entries, (i,j))
        v = A[i,j]
        Nystrom.addentry!(AC,i,j,v)
    end
    @test length(AC.I) == length(AC.J) == length(AC.V) == entries
    @test prev_entries == collect(zip(AC.I,AC.J))
    Asparse = sparse(AC)
    @test Asparse isa SparseMatrixCSC{eltype(A)}
    @test nnz(Asparse) == entries
    mask = Asparse.!=0
    @test Asparse[mask] == A[mask]
    @test Asparse == sparse(AC.I,AC.J,AC.V,size(A)...)

    AC2 = Nystrom.BlockSparseConstructor(T,size(Ablock)...)
    I = [i for (i,_) in prev_entries]
    J = [j for (_,j) in prev_entries]
    V = [A[i,j] for (i,j) in prev_entries]
    Nystrom.addentries!(AC2,I,J,V)
    @test Asparse == sparse(AC2)
end
