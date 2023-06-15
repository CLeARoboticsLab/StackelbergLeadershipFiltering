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

    # Ensure that all dynamics objects are discretized.
    for tt in 1:horizon
        @assert !is_continuous(dyns[tt]) string("Dynamics object at time ", tt, " should be discretized.")
    end

    # Note: There should only be one "player" for an LQR problem.
    num_states = xhdim(dyns[1])
    num_ctrls = uhdim(dyns[1], 1)

    Ps = zeros((num_ctrls, num_states, horizon))
    Zs = zeros((num_states, num_states, horizon))
    Zₜ₊₁ = get_homogenized_state_cost_matrix(costs[horizon]) # Q at last horizon
    Zs[:, :, horizon] = Zₜ₊₁

    # base case
    if horizon == 1
        return Ps
    end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:-1:1

        A = get_homogenized_state_dynamics_matrix(dyns[tt])
        B = get_homogenized_control_dynamics_matrix(dyns[tt], 1)
        Q = get_homogenized_state_cost_matrix(costs[tt])
        R = get_homogenized_control_cost_matrix(costs[tt], 1)

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Zₜ₊₁ * B

        # This is equivalent to inv(r_terms) * B' * Zₜ₊₁ * A
        Ps[:, :, tt] = r_terms \ B' * Zₜ₊₁ * A
        
        # Update Zₜ₊₁ at t+1 to be the one at t as we go to t-1.
        Zₜ₊₁ = Q + A' * Zₜ₊₁ * A - A' * Zₜ₊₁ * B * Ps[:, :, tt]
        Zs[:, :, tt] = Zₜ₊₁
    end

    # Cut off the extra dimension of the homgenized coordinates system.
    extra_dim = xhdim(dyns[1])
    Ks = Ps[1:udim(dyns[1], 1), 1:xdim(dyns[1]), :]
    ks = Ps[1:udim(dyns[1], 1), extra_dim, :]

    Qs = Zs[1:xdim(dyns[1]), 1:xdim(dyns[1]), :]
    qs = Zs[extra_dim, 1:xdim(dyns[1]), :]
    cqs = Zs[extra_dim, extra_dim, :]

    Ps_strategies = FeedbackGainControlStrategy([Ks], [ks])
    Zs_future_costs = [QuadraticCost(Qs[:,:,tt], qs[:, tt], cqs[tt]) for tt in 1:horizon]
    return Ps_strategies, Zs_future_costs
end

export solve_lqr_feedback
