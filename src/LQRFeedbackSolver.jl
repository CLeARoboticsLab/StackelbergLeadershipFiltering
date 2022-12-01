using LinearAlgebra

# Solve a finite horizon, discrete time LQR problem.
# Returns feedback matrices P[:, :, time].

# Shorthand function for LQ time-invariant dynamics and costs.
function solve_lqr_feedback(dyn::LinearDynamics, cost::QuadraticCost, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    costs = [cost for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyn::LinearDynamics, costs::AbstractVector{QuadraticCost}, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyns::AbstractVector{LinearDynamics}, cost::QuadraticCost, horizon::Int)
    costs = [cost for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyns::AbstractVector{LinearDynamics}, costs::AbstractVector{QuadraticCost}, horizon::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(costs) && size(costs, 1) == horizon

    # Note: There should only be one "player" for an LQR problem.
    num_states = xdim(dyns[1])
    num_ctrls = udim(dyns[1], 1)

    Ps = zeros((num_ctrls, num_states, horizon))
    Zₜ₊₁ = costs[horizon].Q

    # base case
    if horizon == 1
        return Ps
    end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:-1:1

        A = dyns[tt].A
        Q = costs[tt].Q
        B = dyns[tt].Bs[1]
        R = costs[tt].Rs[1]

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Zₜ₊₁ * B

        # This is equivalent to inv(r_terms) * B' * Zₜ₊₁ * A
        Ps[:, :, tt] = r_terms \ B' * Zₜ₊₁ * A
        
        # Update Zₜ₊₁ at t+1 to be the one at t as we go to t-1.
        Zₜ₊₁ = Q + A' * Zₜ₊₁ * A - A' * Zₜ₊₁ * B * Ps[:, :, tt]
    end

    return Ps
end


# Solve a finite horizon, discrete time LQR problem by approximating non-LQ dynamics/costs as LQ at each timestep.
# Returns feedback matrices P[:, :, time].

# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lqr_feedback(dyn::Dynamics,
                                         cost::Cost,
                                         horizon::Int,
                                         t0::Float64,
                                         x_refs::AbstractArray{Float64},
                                         u_refs::AbstractArray{Float64})
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Array{LinearDynamics}(undef, T)
    quad_costs = Array{QuadraticCost}(undef, T)

    for tt in 1:T
        # u_refs_at_tt = [u_refs[ii][tt, :] for ii in 1:N]
        current_time = t0 + tt
        lin_dyns[tt] = linearize_dynamics(dyn, current_time, x_refs[tt, :], [u_refs[tt, :]])
        quad_costs[tt] = quadraticize_costs(cost, current_time, x_refs[tt, :], [u_refs[tt, :]])
    end

    return solve_lqr_feedback(lin_dyns, quad_costs, T)
end

export solve_lqr_feedback, solve_approximated_lqr_feedback
