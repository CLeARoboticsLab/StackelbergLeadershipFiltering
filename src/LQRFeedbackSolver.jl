using LinearAlgebra

# Solve a finite horizon, discrete time LQR problem.
# Returns feedback matrices P[:, :, time].
export solve_lqr_feedback
function solve_lqr_feedback(dyn::Dynamics, costs::Cost, horizon::Int)
    A = dyn.A
    Q = costs.Q

    # There should only be one "player" for an LQR problem.
    B = dyn.Bs[1]
    R = costs.Rs[1]

    Ps = zeros((udim(dyn, 1), xdim(dyn), horizon))
    Zₜ₊₁ = Q

    # base case
    if horizon == 1
        return Ps
    end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:1

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Pₜ₊₁ * B

        # This is equivalent to inv(r_terms) * (B' * Zₜ₊₁ * A)
        Ps[:, :, tt] = r_terms \ (B' * Zₜ₊₁ * A)
        
        # Update Zₜ₊₁ at t+1 to be the one at t as we go to t-1.
        Zₜ₊₁ = Q + A' * Zₜ₊₁ * A - A' * Zₜ₊₁ * B * r_terms_inv * B' * Zₜ₊₁ * A
    end

    return Ps
end