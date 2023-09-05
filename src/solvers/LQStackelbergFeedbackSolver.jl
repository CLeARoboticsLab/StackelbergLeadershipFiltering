using LinearAlgebra

function compute_stackelberg_recursive_step(A, B1, B2, Q1, Q2, L1ₜ₊₁, L2ₜ₊₁, R11, R22, R12, R21)
    """
    A helper which accepts all of the inputs needed for computing the Stackelberg recursion and produces the result of
    one recursive step.

    Returns S1, S2, L1, L2 at the current timestep.
    """
    Dₜ = (R22 + B2' * L2ₜ₊₁ * B2) \ (B2' * L2ₜ₊₁)

    lhs = R11 + B1' * Dₜ' * R12 * Dₜ * B1 + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (B1 - B2 * Dₜ * B1)
    rhs = ((B1' * Dₜ' * R12 * Dₜ) + (B1 - B2 * Dₜ * B1)' * L1ₜ₊₁ * (I - B2 * Dₜ)) * A

    S1ₜ = lhs \ rhs
    S2ₜ = Dₜ * (A - B1 * S1ₜ)

    dynamics_tp1 = A - B1 * S1ₜ - B2 * S2ₜ
    L1 = Q1 + S1ₜ' * R11 * S1ₜ + S2ₜ' * R12 * S2ₜ + dynamics_tp1' * L1ₜ₊₁ * dynamics_tp1
    L2 = Q2 + S1ₜ' * R21 * S1ₜ + S2ₜ' * R22 * S2ₜ + dynamics_tp1' * L2ₜ₊₁ * dynamics_tp1
    return [S1ₜ, S2ₜ, L1, L2]
end

# Solve a finite horizon, discrete time LQ game to feedback Stackelberg equilibrium.
# Returns feedback matrices P[player][:, :, time]
function solve_lq_stackelberg_feedback(dyns::AbstractVector{LinearDynamics},
                                       all_costs::AbstractVector{<:AbstractVector{QuadraticCost}},
                                       horizon::Int,
                                       leader_idx::Int; state_reg_param=0., control_reg_param=1e-32, ensure_pd=false)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(all_costs) && size(all_costs, 1) == horizon

    # Ensure that all dynamics objects are discretized.
    # Ensure that the number of controls matches number of players at each horizon.
    for tt in 1:horizon
        @assert !is_continuous(dyns[tt]) string("Dynamics object at time ", tt, " should be discretized.")
        @assert size(all_costs[tt], 1) == num_agents(dyns[tt])
    end

    # TODO(hamzah) If we ever go beyond a 2-player game, figure out multiple followers.
    follower_idx = (leader_idx == 2) ? 1 : 2
    num_players = num_agents(dyns[1])
    num_states = xhdim(dyns[1])

    # Define recursive variables and initialize variables - the number of players, states, and control sizes are assumed
    # to be constant over time.
    all_Ss = [zeros(uhdim(dyns[1], ii), num_states, horizon) for ii in 1:num_players]
    all_Ls = [zeros(num_states, num_states, horizon) for _ in 1:num_players]
    all_Ls[leader_idx][:, :, horizon] = get_homogenized_state_cost_matrix(all_costs[horizon][leader_idx])
    all_Ls[follower_idx][:, :, horizon] = get_homogenized_state_cost_matrix(all_costs[horizon][follower_idx])

    # t will increment from 1 ... K-1. k will decrement from K-1 ... 1.
    for tt = horizon-1:-1:1

        # Get the dynamics and costs at the current time.
        dyn = dyns[tt]
        costs = all_costs[tt]

        # Define control variables which are the same over all horizon.
        A = get_homogenized_state_dynamics_matrix(dyn)
        B_leader = get_homogenized_control_dynamics_matrix(dyn, leader_idx)
        B_follower = get_homogenized_control_dynamics_matrix(dyn, follower_idx)

        Q_leader = get_homogenized_state_cost_matrix(costs[leader_idx]; reg_param=state_reg_param, ensure_pd=ensure_pd)
        Q_follower = get_homogenized_state_cost_matrix(costs[follower_idx]; reg_param=state_reg_param, ensure_pd=ensure_pd)

        R₁₁ = get_homogenized_control_cost_matrix(costs[leader_idx], leader_idx; reg_param=control_reg_param, ensure_pd=ensure_pd)
        R₂₂ = get_homogenized_control_cost_matrix(costs[follower_idx], follower_idx; reg_param=control_reg_param, ensure_pd=ensure_pd)
        R₁₂ = get_homogenized_control_cost_matrix(costs[leader_idx], follower_idx)
        R₂₁ = get_homogenized_control_cost_matrix(costs[follower_idx], leader_idx)

        # Ensure that we have (semi-)positive definiteness in the relevant costs when ensure_pd enabled.
        @assert !ensure_pd || isposdef(Q_leader) || minimum(eigvals(Q_leader)) == 0 "Q leader not semi-posdef, min eigvals: $(minimum(eigvals(Q_leader))) $(eigvals(Q_leader)))"
        @assert !ensure_pd || isposdef(Q_follower) || minimum(eigvals(Q_follower)) == 0 "Q follower not semi-posdef, min eigvals: $(minimum(eigvals(Q_follower))) $(eigvals(Q_follower)))"
        @assert !ensure_pd || isposdef(R₁₁) "R₁₁ not semi-posdef, min eigvals: $(minimum(eigvals(R₁₁))) $(eigvals(R₁₁)))"
        @assert !ensure_pd || isposdef(R₂₂) "R₂₂ not semi-posdef, min eigvals: $(minimum(eigvals(R₂₂))) $(eigvals(R₂₂)))"

        Lₖ₊₁ = [all_Ls[leader_idx][:, :, tt+1], all_Ls[follower_idx][:, :, tt+1]]

        # Run recursive computation for one step.
        outputs = compute_stackelberg_recursive_step(A, B_leader, B_follower, Q_leader, Q_follower, Lₖ₊₁[leader_idx], Lₖ₊₁[follower_idx], R₁₁, R₂₂, R₁₂, R₂₁)

        all_Ss[leader_idx][:, :, tt] = outputs[1]
        all_Ss[follower_idx][:, :, tt] = outputs[2]
        all_Ls[leader_idx][:, :, tt] = outputs[3]
        all_Ls[follower_idx][:, :, tt] = outputs[4]
    end

    # Cut off the extra dimension of the homogenized coordinates system.
    extra_dim = xhdim(dyns[1])
    Ks = [all_Ss[ii][1:udim(dyns[1], ii), 1:xdim(dyns[1]), :] for ii in 1:num_players]
    ks = [all_Ss[ii][1:udim(dyns[1], ii), extra_dim, :] for ii in 1:num_players]

    Qs = [all_Ls[ii][1:xdim(dyns[1]), 1:xdim(dyns[1]), :] for ii in 1:num_players]
    qs = [all_Ls[ii][extra_dim, 1:xdim(dyns[1]), :] for ii in 1:num_players]
    cqs = [all_Ls[ii][extra_dim, extra_dim, :] for ii in 1:num_players]

    Ps_strategies = FeedbackGainControlStrategy(Ks, ks)
    Zs_future_costs = [[QuadraticCost(Qs[ii][:,:,tt], qs[ii][:, tt], cqs[ii][tt]) for tt in 1:horizon] for ii in 1:num_players]

    return Ps_strategies, Zs_future_costs
end

# Shorthand function for LQ time-invariant dynamics and costs.
function solve_lq_stackelberg_feedback(dyn::LinearDynamics, costs::AbstractVector{QuadraticCost}, horizon::Int, leader_idx::Int)
    dyns = [dyn for _ in 1:horizon]
    all_costs = [costs for _ in 1:horizon]
    return solve_lq_stackelberg_feedback(dyns, all_costs, horizon, leader_idx)
end

function solve_lq_stackelberg_feedback(dyn::LinearDynamics, all_costs::AbstractVector{<:AbstractVector{QuadraticCost}}, horizon::Int, leader_idx::Int)
    dyns = [dyn for _ in 1:horizon]
    return solve_lq_stackelberg_feedback(dyns, all_costs, horizon, leader_idx)
end

function solve_lq_stackelberg_feedback(dyns::AbstractVector{LinearDynamics}, costs::AbstractVector{QuadraticCost}, horizon::Int, leader_idx::Int)
    all_costs = [costs for _ in 1:horizon]
    return solve_lq_stackelberg_feedback(dyns, all_costs, horizon, leader_idx)
end

export compute_stackelberg_recursive_step, solve_lq_stackelberg_feedback
