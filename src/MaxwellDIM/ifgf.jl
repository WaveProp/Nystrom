
IFGF.wavenumber(K::MaxwellPotentialKernel) = K |> Nystrom.pde |> Nystrom.parameters

function IFGF.centered_factor(K::MaxwellPotentialKernel,x,Y)
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    g  = exp(im*k*d)/(4π*d)  # Helmholtz Green's function
    return g
end
