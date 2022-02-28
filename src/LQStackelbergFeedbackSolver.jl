using LinearAlgebra

# Solve a finite horizon, discrete time LQ game to feedback Stackelberg equilibrium.
# Returns feedback matrices P[player][:, :, time]

# Shorthand function for LTI dynamics and costs.
export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyn::Dynamics, costs::AbstractArray{Cost}, horizon::Int, leader_idx::Int)
    dyns = [dyn for _ in 1:horizon]
    costs = repeat(costs, 1, horizon)
    return solve_lq_stackelberg_feedback(dyns, costs, horizon, leader_idx)
end

export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyns::AbstractArray{Dynamics}, costs::AbstractArray{Cost}, horizon::Int, leader_idx::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert(ndims(dyns) == 1 && size(dyns, 1) == horizon)
    @assert(ndims(costs) == 2 && size(costs, 2) == horizon)

    num_players = size(costs, 1)

    # TODO(hamzah) If we ever go beyond a 2-player game, figure out multiple followers.
    follower_idx = (leader_idx == 2) ? 1 : 2
    num_states = xdim(dyns[1])
    num_leader_ctrls = udim(dyns[1], leader_idx)
    num_follower_ctrls = udim(dyns[1], follower_idx)

    # Define recursive variables and initialize variables.
    all_Ss = [zeros(udim(dyns[1], i), num_states, horizon) for i in 1:num_players]
    Lₖ₊₁ = [costs[i, 1].Q for i in 1:num_players]

    # t will increment from 1 ... K-1. k will decrement from K-1 ... 1.
    for tt = 1:horizon-1
        k = horizon - tt

        # Define control variables which are the same over all horizon.
        A_t = dyns[tt].A
        B_leader = dyns[tt].Bs[leader_idx]
        B_follower = dyns[tt].Bs[follower_idx]
        Q_leader = costs[leader_idx, tt].Q
        Q_follower = costs[follower_idx, tt].Q

        # TODO: Change the incentives later, but for now it's identity since it must be positive definite.
        R₁₂ = zeros(num_leader_ctrls, num_follower_ctrls)
        # This one can be 0.
        R₂₁ = zeros(num_follower_ctrls, num_leader_ctrls)

        # 1. Compute Sₖ for each player.
        common_ctrl_cost_term = I + B_follower' * Lₖ₊₁[follower_idx] * B_follower

        G₁ = I + Lₖ₊₁[follower_idx] * B_follower * B_follower'
        G₂ = I + B_follower * B_follower' * Lₖ₊₁[follower_idx]
        F  = I + B_follower' * Lₖ₊₁[follower_idx] * B_follower
        H  = B_leader' * inv(G₁) * Lₖ₊₁[leader_idx] * inv(G₂) * B_leader
        J  = B_leader' * Lₖ₊₁[follower_idx]' * B_follower * inv(F) * R₁₂ * inv(F) * B_follower' * Lₖ₊₁[follower_idx] * B_leader
        M  = inv(G₁) * Lₖ₊₁[leader_idx] * G₂
        N  = Lₖ₊₁[follower_idx]' * B_follower * inv(F) * R₁₂ * inv(F) * B_follower' * Lₖ₊₁[follower_idx] 
        S1ₖ = inv(H * J + I) * B_leader' * (M + N) * A_t
        all_Ss[leader_idx][:, :, k] = S1ₖ

        S2ₖ = (inv(common_ctrl_cost_term)
              * B_follower'
              * Lₖ₊₁[follower_idx]
              * (A_t - B_leader * S1ₖ))
        all_Ss[follower_idx][:, :, k] = S2ₖ

        # 2. Compute feedback matrices Lₖ.
        ẋ = A_t - B_leader * S1ₖ - B_follower * S2ₖ

        Lₖ₊₁[leader_idx] = ẋ' * Lₖ₊₁[leader_idx] * ẋ
                           + S1ₖ' * S1ₖ
                           + S2ₖ' * R₁₂ * S2ₖ
                           + Q_leader

        Lₖ₊₁[follower_idx] = ẋ' * Lₖ₊₁[follower_idx] * ẋ
                           + S2ₖ' * S2ₖ
                           + S1ₖ' * R₂₁ * S1ₖ
                           + Q_follower
        # recurse!
    end

    return all_Ss
end
