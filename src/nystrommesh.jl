"""
    struct NystromDOF

Structure containing information of a degree-of-freedom in Nyström methods.
"""
struct NystromDOF{N,T,M,NM}
    # index::Int64         # QNode global index
    # element_index::Int64 # Index of the element to which QNode belongs
    coords::SVector{N,T}           # *Lifted* quadrature nodes
    weight::T                      # *Lifted* quadrature weigth
    jacobian::SMatrix{N,M,T,NM}    # Jacobian matrix at qnode
    locidx::Int                    # Index within element
    globidx::Int                   # Index in global system
end

coords(dof::NystromDOF)   = dof.coords
weight(dof::NystromDOF)   = dof.weight
jacobian(dof::NystromDOF) = dof.jacobian
normal(dof::NystromDOF)   = normal(jacobian(dof))

function NystromDOF(q::SVector{N,T},w::T,jac::SMatrix{N,_,T}) where {N,_,T}
    NystromDOF{N,T}(q,w,jac)
end

function Base.show(io::IO,dof::NystromDOF)
    print(io,
    "Nystrom DOF:
        coords  = $(dof.coords)
        locidx  = $(dof.locidx)
        globidx = $(dof.globidx)")
end

bounding_box(qnodes::Vector{<:NystromDOF}) = bounding_box(coords(q) for q in qnodes)

"""
    struct NystromMesh{N,T} <: AbstractMesh{N,T}

A mesh data structure for solving boundary integral equation using Nyström
methods.

A `NystromMesh` can be constructed from a `mesh::AbstractMesh` and a dictionary
`etype2qrule` mapping element types in `mesh` (the keys) to an appropriate
quadrature rule for integration over elements of that type (the value).

The degrees of freedom in a `NystromMesh` are associated to nodal values at the
quadrature nodes, and are reprensented using [`NystromDOF`](@ref).
"""
struct NystromMesh{N,T,M,NM} <: AbstractMesh{N,T}
    elements::Dict{DataType,Any}                                # Geometrical elements
    etype2qrule::Dict{DataType,AbstractQuadratureRule}          # Element type --> quadrature
    dofs::Vector{NystromDOF{N,T,M,NM}}                          # Degrees of freedom
    elt2dof::Dict{DataType,Matrix{Int}}                         # Elements --> dofs
    ent2elt::Dict{AbstractEntity,Dict{DataType,Vector{Int}}}    # Entitiy --> element
end

function Base.show(io::IO,msh::NystromMesh)
    print(io," NystromMesh with $(length(dofs(msh))) DOFs")
end

# getters
elements(m::NystromMesh)             = m.elements
elements(m::NystromMesh,E::DataType) = m.elements[E]
etype2qrule(m)   = m.etype2qrule
etype2qrule(m,E) = etype2qrule(m)[E]
dofs(m::NystromMesh)  = m.dofs
elt2dof(m::NystromMesh) = m.elt2dof
elt2dof(m::NystromMesh,E::DataType) = m.elt2dof[E]
ent2elt(m::NystromMesh) = m.ent2elt
ent2elt(m::NystromMesh,ent::AbstractEntity) = m.ent2elt[ent]

# generators for iterating over fields of dofs
qcoords(m::NystromMesh)  = (coords(q) for q in dofs(m))
qweights(m::NystromMesh) = (weight(q) for q in dofs(m))
qjacobians(m::NystromMesh) = (jacobian(q) for q in dofs(m))
qnormals(m::NystromMesh) = (normal(q) for q in dofs(m))

# entities and domain
entities(mesh::NystromMesh) = collect(keys(mesh.ent2elt))
domain(mesh::NystromMesh)   = Domain(entities(mesh))

Base.keys(m::NystromMesh) = keys(elements(m))

# other maps constructed from ent2elt and elt2dof
function dom2elt(mesh::NystromMesh,domain::Domain)
    dict = Dict{DataType,Vector{Int}}()
    for ent in entities(domain)
        other = ent2elt(mesh,ent)
        mergewith!(append!,dict,other)
    end
    return dict
end

function dom2dof(mesh::NystromMesh,domain::Domain)
    idxs = Int[]
    for ent in entities(domain)
        dict = ent2elt(mesh,ent)
        for (E,tags) in dict
            append!(idxs,view(elt2dof(mesh,E),:,tags))
        end
    end
    return idxs
end
dom2dof(mesh,ent::AbstractEntity) = dom2dof(mesh,Domain(ent))

ent2dof(mesh,ent::AbstractEntity) = dom2dof(mesh,ent)

function NystromMesh(msh::AbstractMesh{N,T},Ω::Domain,e2qrule) where {N,T}
    # initialize mesh with empty fields
    M       = geometric_dimension(Ω) |> Int
    NM      = N*M
    nys_msh = NystromMesh{N,T,M,NM}(
        Dict{DataType,Any}(),                               # elemements
        e2qrule,                                            # etype2qrule
        NystromDOF{N,T,M,NM}[],                             # dofs
        Dict{DataType,Matrix{Int}}(),                       # elt2dof
        Dict{AbstractEntity,Dict{DataType,Vector{Int}}}()   # ent2elt
    )
    # loop over entities, generate quadrature, and fill in various fields
    for ent in entities(Ω)
        ent_msh = view(msh,ent)
        dict    = Dict{DataType,Vector{Int}}() # store ent2elt for current ent
        for E in keys(ent_msh) # element types
            @assert haskey(e2qrule,E) "no quadrature rule found for element of type $E"
            qrule  = e2qrule[E]
            els    = get!(nys_msh.elements,E,Vector{E}())
            istart = length(els)+1
            iter   = ElementIterator(ent_msh,E)
            _build_nystrom_mesh!(nys_msh,iter,qrule)
            iend    = length(els)
            dict[E] = collect(istart:iend)
        end
        nys_msh.ent2elt[ent] = dict # new entry
    end
    return nys_msh
end
function NystromMesh(msh::GenericMesh, Ω::Domain;order)
    etypes = keys(view(msh,Ω))
    e2qrule = Dict(E=>qrule_for_reference_shape(domain(E),order) for E in etypes)
    NystromMesh(msh,Ω,e2qrule)
end
function NystromMesh(submesh::SubMesh,args...;kwargs...)
    NystromMesh(parent(submesh),domain(submesh),args...;kwargs...)
end

@noinline function _build_nystrom_mesh!(msh,iter::ElementIterator,qrule::AbstractQuadratureRule)
    E               = eltype(iter)
    els::Vector{E}  = msh.elements[E]
    el2dofs         = get!(msh.elt2dof,E,Matrix{Int}(undef,0,0))
    el2dofs         = vec(el2dofs)
    x̂,ŵ             = qrule() #nodes and weights on reference element
    num_nodes       = length(ŵ)
    for el in iter
        # add element
        push!(els,el)
        # and all qnodes for that element
        for i in 1:num_nodes
            x       = el(x̂[i])
            jac     = jacobian(el,x̂[i])
            μ       = integration_measure(jac)
            w       = μ*ŵ[i]
            # global index of the dof that is going to be created
            globidx = length(dofs(msh)) + 1
            locidx  = i
            dof     = NystromDOF(x,w,jac,locidx,globidx)
            push!(dofs(msh),dof)
            push!(el2dofs,globidx)
        end
    end
    el2dofs        = reshape(el2dofs,num_nodes,:)
    msh.elt2dof[E] = el2dofs
    return msh
end

# ElementIterator for NystromMesh
function Base.size(iter::ElementIterator{<:AbstractElement,<:NystromMesh})
    E = eltype(iter)
    M = mesh(iter)
    return size(elements(M,E))
end

function Base.getindex(iter::ElementIterator{<:AbstractElement,<:NystromMesh},i::Int)
    E = eltype(iter)
    M = mesh(iter)
    return elements(M,E)[i]
end

function Base.iterate(iter::ElementIterator{<:AbstractElement,<:NystromMesh}, state=1)
    state > length(iter) && (return nothing)
    iter[state], state + 1
end


"""
    near_interaction_list(pts,Y;atol)

Given target points `pts` and a `Y::NystromMesh`, return a dictionary with keys
given by element types of the source mesh `Y`. Each key has a value given by a
vector whose `i`-th entry encodes information about the points in `X` which are
close to a given element of the key type.
"""
function near_interaction_list(pts,Y::NystromMesh;atol)
    dict = Dict{DataType,Vector{Vector{Tuple{Int,Int}}}}()
    for E in keys(Y)
        ilist = _etype_near_interaction_list(pts,Y,E,atol)
        push!(dict,E=>ilist)
    end
    return dict
end

function _etype_near_interaction_list(pts,Y,E,atol)
    ilist    = Vector{Vector{Tuple{Int,Int}}}()
    e2n      = elt2dof(Y,E)
    npts,nel = size(e2n)
    for n in 1:nel
        ynodes = @views dofs(Y)[e2n[:,n]]
        inear  = _near_interaction_list(pts,ynodes,atol)
        push!(ilist,inear)
    end
    ilist
end

function _near_interaction_list(pts,ynodes,atol)
    ilist    = Tuple{Int,Int}[]
    for (i,x) in enumerate(pts)
        d,j = findmin([norm(coords(x)-coords(y)) for y in ynodes])
        if d ≤ atol
            push!(ilist,(i,j))
        end
    end
    return ilist
end

# function Base.append!(msh1::NystromMesh,msh2::NystromMesh)
#     Ω = domain(msh2)
#     @assert ambient_dimension(msh1) == ambient_dimension(msh2)
#     @assert eltype(msh1) == eltype(msh2)
#     # extract relevant quadrature rules
#     etype2dof = Dict{DataType,Vector{Int}}()
#     for (E,q) in etypes(msh2,Ω)
#         if haskey(msh1.elements,E)
#             etype2dof[E]        = msh1.elt2dof[E] |> vec
#             @assert msh1.etype2qrule[E] == q
#         else
#             etype2dof[E]        = Int[]
#             msh1.elements[E]    = E[]
#             msh1.etype2qrule[E] = q
#         end
#     end
#     # loop over entities and extract the information
#     for ent in entities(msh2)
#         haskey(msh1.ent2elt,ent) && continue # entity already meshed
#         msh1.ent2elt[ent] = Dict{DataType,Vector{Int}}()
#         for (E,v) in ent2elt(msh2,ent)
#             # add dofs information
#             msh2_dofs  = msh2.elt2dof[E][:,v]
#             idx_start = length(msh1.qnodes) + 1
#             append!(msh1.qnodes,msh2.qnodes[msh2_dofs])
#             append!(msh1.qweights,msh2.qweights[msh2_dofs])
#             if !isempty(qnormals(msh2)) # otherwise assume it did not have normal stored
#                 append!(msh1.qnormals,msh2.qnormals[msh2_dofs])
#             end
#             idx_end = length(msh1.qnodes) # end index of dofs added
#             append!(etype2dof[E],collect(idx_start:idx_end))
#             # append elements
#             idx_start = length(msh1.elements[E]) + 1
#             append!(msh1.elements[E],msh2.elements[E][v])
#             idx_end   = length(msh1.elements[E])
#             # and mapping ent -> elt
#             msh1.ent2elt[ent][E] = collect(idx_start:idx_end)
#         end
#     end
#     # finally do some rehshaping
#     for (E,v) in etype2dof
#         dof_per_el = size(msh2.elt2dof[E],1)
#         msh1.elt2dof[E] = reshape(v,dof_per_el,:)
#     end
#     return msh1
# end

function diagonal_ncross_and_jacobian_matrices(nmesh)
    qnodes = dofs(nmesh)
    nmatrix = Diagonal([cross_product_matrix(normal(q)) for q in qnodes])
    jmatrix = Diagonal([jacobian(q) for q in qnodes])
    dual_jmatrix = Diagonal([SMatrix{2,3,Float64,6}(pinv(jacobian(q))) for q in qnodes])
    return nmatrix, jmatrix, dual_jmatrix
end
diagonal_ncross_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[1]
diagonal_jacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[2]
diagonal_dualjacobian_matrix(nmesh) = diagonal_ncross_and_jacobian_matrices(nmesh)[3]

# Plot recipes

struct PlotQuadrature end

@recipe function f(mesh::NystromMesh)
    # plot elements
    label --> ""
    grid  --> false
    aspect_ratio --> :equal
    for (E,els) in mesh.elements
        for el in els
            @series begin
                el
            end
        end
    end
    # # plot quadrature
    # PlotQuadrature(),mesh
end

@recipe function f(::PlotQuadrature,mesh::NystromMesh)
    marker --> :circle
    pts = qcoords(mesh) |> collect
    WavePropBase.IO.PlotPoints(), pts
end
