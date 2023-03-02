using LinearAlgebra

# Solve a finite horizon, discrete time LQR problem.
# Returns feedback matrices P[:, :, time].

# Shorthand function for LQ time-invariant dynamics and costs.
function solve_new_lqr_feedback(dyn::LinearDynamics, cost::QuadraticCost, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    costs = [cost for _ in 1:horizon]
    return solve_new_lqr_feedback(dyns, costs, horizon)
end

function solve_new_lqr_feedback(dyn::LinearDynamics, costs::AbstractVector{QuadraticCost}, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    return solve_new_lqr_feedback(dyns, costs, horizon)
end

function solve_new_lqr_feedback(dyns::AbstractVector{LinearDynamics}, cost::QuadraticCost, horizon::Int)
    costs = [cost for _ in 1:horizon]
    return solve_new_lqr_feedback(dyns, costs, horizon)
end

function solve_new_lqr_feedback(dyns::AbstractVector{LinearDynamics}, costs::AbstractVector{QuadraticCost}, horizon::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(costs) && size(costs, 1) == horizon

    # Note: There should only be one "player" for an LQR problem.
    num_states = xdim(dyns[1])
    num_ctrls = udim(dyns[1], 1)

    Ps = zeros((num_ctrls, num_states, horizon))
    ps = zeros((num_ctrls, horizon))
    Zs = zeros((num_states, num_states, horizon))
    zs = zeros((num_states, horizon))
    czs = zeros(horizon)

    Zₜ₊₁ = get_quadratic_state_term(costs[horizon])
    zₜ₊₁ = get_linear_state_term(costs[horizon])
    czₜ₊₁ = get_constant_state_term(costs[horizon])
    Zs[:, :, horizon] = Zₜ₊₁

    # base case
    if horizon == 1
        return Ps, costs[horizon]
    end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:-1:1
        A = get_homogenized_state_dynamics_matrix(dyns[tt])
        B = get_homogenized_control_dynamics_matrix(dyns[tt], 1)

        Q = get_quadratic_state_term(costs[tt])
        q = get_linear_state_term(costs[tt])
        cq = get_constant_state_term(costs[tt])
        R = get_quadratic_control_term(costs[tt], 1)
        r = get_linear_control_term(costs[tt], 1)
        cr = get_constant_control_term(costs[tt], 1)

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Zₜ₊₁ * B

        # This is equivalent to inv(r_terms) * B' * Zₜ₊₁ * A but more numerically stable.
        Ps[:, :, tt] = r_terms \ B' * Zₜ₊₁ * A
        ps[:, tt] = r_terms \ (B' * zₜ₊₁ + r)
        
        # Update Zₜ₊₁ at t+1 to be the one at t as we go to t-1.
        Pₜ = Ps[:, :, tt]
        pₜ = ps[:, tt]
        Zₜ = Q + Pₜ' * R * Pₜ + (A - B * Pₜ)' * Zₜ₊₁ * (A - B * Pₜ)
        # zₜ = pₜ' * R * Pₜ + (zₜ₊₁' - pₜ' * B' * Zₜ₊₁) * (A - B * Pₜ)
        # zₜ = zₜ'
        zₜ = Pₜ' * R * pₜ + (A - B * Pₜ)' * (zₜ₊₁ - Zₜ₊₁ * B * pₜ)
        czₜ = pₜ' * (R + B' * Zₜ₊₁ * B) * pₜ - 2 * zₜ₊₁' * B * pₜ

        Zs[:, :, tt] = Zₜ₊₁
        zs[:, tt] = zₜ₊₁
        czs[tt] = czₜ₊₁

        Zₜ₊₁ = Zₜ
        zₜ₊₁ = zₜ
    end

    Ps_feedback_strategies = FeedbackGainControlStrategy([Ps], [ps])
    Zs_future_costs = [QuadraticCost(Zs[:, :, tt], zs[:, tt], czs[tt]) for tt in 1:horizon]
    return Ps_feedback_strategies, Zs_future_costs
end

export solve_new_lqr_feedback
