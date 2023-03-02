using LinearAlgebra

# Solve a finite horizon, discrete time LQR problem.
# Returns feedback matrices P[:, :, time].

# Shorthand function for LQ time-invariant dynamics and costs.
function solve_lqr_feedback(dyn::LinearDynamics, cost::PureQuadraticCost, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    costs = [cost for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyn::LinearDynamics, costs::AbstractVector{PureQuadraticCost}, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyns::AbstractVector{LinearDynamics}, cost::PureQuadraticCost, horizon::Int)
    costs = [cost for _ in 1:horizon]
    return solve_lqr_feedback(dyns, costs, horizon)
end

function solve_lqr_feedback(dyns::AbstractVector{LinearDynamics}, costs::AbstractVector{PureQuadraticCost}, horizon::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(costs) && size(costs, 1) == horizon

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
    Z_future_costs = [PureQuadraticCost(Zs[:, :, tt]) for tt in 1:horizon]
    return Ps[1:udim(dyns[1], 1),:,:], Z_future_costs
end


# Solve a finite horizon, discrete time LQR problem by approximating non-LQ dynamics/costs as LQ at each timestep.
# Returns feedback matrices P[:, :, time].

# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lqr_feedback(dyn::Dynamics,
                                         cost::Cost,
                                         horizon::Int,
                                         t0::Float64,
                                         xs_1::AbstractArray{Float64},
                                         us_1::AbstractArray{Float64})
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Array{LinearDynamics}(undef, T)
    quad_costs = Array{PureQuadraticCost}(undef, T)

    for tt in 1:T
        prev_time = t0 + ((tt == 1) ? 0 : tt-1)
        current_time = t0 + tt
        time_range = (prev_time, current_time)
        lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_1[:, tt], [us_1[:, tt]])
        quad_costs[tt] = quadraticize_costs(cost, time_range, xs_1[:, tt], [us_1[:, tt]])
    end

    return solve_lqr_feedback(lin_dyns, quad_costs, T)
end

export solve_lqr_feedback, solve_approximated_lqr_feedback
