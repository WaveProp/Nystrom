## IFGF interface
IFGF.wavenumber(K::MaxwellPotentialKernel) = K |> Nystrom.pde |> Nystrom.parameters

function IFGF.centered_factor(K::MaxwellPotentialKernel,x,Y)
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    g  = exp(im*k*d)/(4π*d)  # Helmholtz Green's function
    return g
end

function IFGF.inv_centered_factor(K::MaxwellPotentialKernel, x, Y)
    # return inv(centered_factor(K, x, Y))
    yc = IFGF.center(Y)
    r  = x-yc
    k  = IFGF.wavenumber(K)
    d  = norm(r)
    invg = (4π*d)/exp(im*k*d)  # inverse of Helmholtz Green's function
    return invg
end

function IFGF.transfer_factor(K::MaxwellPotentialKernel, x, Y)
    Yparent = parent(Y)
    # return inv_centered_factor(K, x, Yparent) * centered_factor(K, x, Y)
    k   = IFGF.wavenumber(K)
    yc  = IFGF.center(Y)
    ycp = IFGF.center(Yparent)
    r   = x-yc
    rp  = x-ycp
    d   = norm(r)
    dp  = norm(rp)
    return dp/d*exp(im*k*(d-dp))
end
