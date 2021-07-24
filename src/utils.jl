"""
    enable_debug()

Activate debugging messages.
"""
function enable_debug()
    ENV["JULIA_DEBUG"] = Nystrom
end

error_green_formula(SL,DL,γ₀u,γ₁u,u,σ)                      = σ*u + SL*γ₁u  - DL*γ₀u
error_derivative_green_formula(ADL,H,γ₀u,γ₁u,un,σ)          = σ*un + ADL*γ₁u - H*γ₀u
error_interior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,-1/2)
error_interior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,-1/2)
error_exterior_green_identity(SL,DL,γ₀u,γ₁u)                = error_green_formula(SL,DL,γ₀u,γ₁u,γ₀u,1/2)
error_exterior_derivative_green_identity(ADL,H,γ₀u,γ₁u)     = error_derivative_green_formula(ADL,H,γ₀u,γ₁u,γ₁u,1/2)
