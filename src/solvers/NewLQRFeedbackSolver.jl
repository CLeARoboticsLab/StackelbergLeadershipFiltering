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
    ps = zeros((num_ctrls, horizon))
    Zs = zeros((num_states, num_states, horizon))
    zs = zeros((num_states, horizon))
    czs = zeros(horizon)

    Zₜ₊₁ = get_quadratic_state_cost_term(costs[horizon])
    zₜ₊₁ = get_linear_state_cost_term(costs[horizon])
    czₜ₊₁ = get_constant_state_cost_term(costs[horizon])
    Zs[:, :, horizon] = Zₜ₊₁

    # base case
    # if horizon == 1
    #     return Ps, costs[horizon]
    # end

    # At each horizon running backwards, solve the LQR problem inductively.
    for tt in horizon:-1:1
        A = get_linear_state_dynamics(dyns[tt])
        a = get_constant_state_dynamics(dyns[tt])
        B = get_control_dynamics(dyns[tt], 1)

        Q = get_quadratic_state_cost_term(costs[tt])
        q = get_linear_state_cost_term(costs[tt])
        cq = get_constant_state_cost_term(costs[tt])
        R = get_quadratic_control_cost_term(costs[tt], 1)
        r = get_linear_control_cost_term(costs[tt], 1)
        cr = get_constant_control_cost_term(costs[tt], 1)

        # Solve the LQR using induction and optimizing the quadratic cost for P and Z.
        r_terms = R + B' * Zₜ₊₁ * B

        # This solves a constrained system to identify the feedback gain and constant gain.
        lhs = vcat(hcat(r_terms, r), hcat(r', cr))
        rhs = vcat(hcat(B' * Zₜ₊₁ * A,  B' * (Zₜ₊₁ * a + zₜ₊₁)), zeros(1, num_states+1))
        Pp = lhs \ rhs
        # println(Pp)
        # Ps[:, :, tt] = r_terms \ (B' * Zₜ₊₁ * A)
        Ps[:, :, tt] = Pp[1:num_ctrls, 1:num_states][:,:]
        ps[:, tt] = Pp[1:num_ctrls, num_states+1]
        # = r' \ (B' * zₜ₊₁ + B' * Zₜ₊₁ * a)
        # r_terms \ (r + B' * zₜ₊₁ + B' * Zₜ₊₁ * a)
        # r_terms \ (B' * zₜ₊₁ + r + B' * Zₜ₊₁ * a)


        # println(tt, " - new r_terms", r_terms)
        # println(tt, " - new divider 1: ", B' * Zₜ₊₁ * A)
        # println(tt, " - new divider 2: ", r + B' * zₜ₊₁ + B' * Zₜ₊₁ * a)
        
        # Update feedback and cost terms.
        Pₜ = Ps[:, :, tt]
        pₜ = ps[:, tt]
        println(tt, " - new P: ", Pₜ)
        println(tt, " - new p: ", pₜ)
        Zₜ = Q + Pₜ' * R * Pₜ + (A - B * Pₜ)' * Zₜ₊₁ * (A - B * Pₜ)
        # println(tt, " - new Zₜ: ", Zₜ, Zₜ')
        # Zₜ = (Zₜ + Zₜ')/2.
        # @assert Zₜ == Zₜ'
        zₜ = q 
        zₜ += Pₜ' * (R * pₜ - r)
        zₜ += (A - B * Pₜ)' * Zₜ₊₁ * (a - B * pₜ)
        zₜ += (A - B * Pₜ)' * zₜ₊₁
        czₜ = czₜ₊₁ + cq
        println(tt, " - new cz 1: ", czₜ₊₁ + cq)
        czₜ += pₜ' * (R * pₜ - 2 * r)
        println(tt, " - new cz 2: ", pₜ' * (R * pₜ - 2 * r))
        czₜ += ((a - B * pₜ)' * Zₜ₊₁ + 2 * zₜ₊₁') * (a - B * pₜ)
        println(tt, " - new cz 3: ", ((a - B * pₜ)' * Zₜ₊₁ + 2 * zₜ₊₁') * (a - B * pₜ))
        println(tt, " - new Z: ", Zₜ)
        println(tt, " - new z: ", zₜ)
        println(tt, " - new cz: ", czₜ)

        # zₜ = Pₜ' * R * pₜ + (A - B * Pₜ)' * (zₜ₊₁ + Zₜ₊₁ * (a - B * pₜ) )
        # czₜ = pₜ' * (R + B' * Zₜ₊₁ * B) * pₜ - 2 * zₜ₊₁' * B * pₜ
        # czₜ += 2 * (zₜ₊₁' - pₜ' * B' * Zₜ₊₁) * a + a' * Zₜ₊₁ * a

        # czₜ = pₜ' * (R + B' * Zₜ₊₁ * B) * pₜ 
        # czₜ -= 2 * zₜ₊₁' * B * pₜ 
        # czₜ += (2 * zₜ₊₁ + Zₜ₊₁ * (a - B * pₜ))' * a

        Zs[:, :, tt] = Zₜ # 0.5 * (Zₜ' + Zₜ)
        zs[:, tt] = zₜ
        czs[tt] = czₜ₊₁

        Zₜ₊₁ = Zₜ
        zₜ₊₁ = zₜ
        czₜ₊₁ = czₜ
    end

    Ps_feedback_strategies = FeedbackGainControlStrategy([Ps], [ps])
    Zs_future_costs = [QuadraticCost(Zs[:, :, tt], zs[:, tt], czs[tt]) for tt in 1:horizon]
    return Ps_feedback_strategies, Zs_future_costs
end

export solve_lqr_feedback
