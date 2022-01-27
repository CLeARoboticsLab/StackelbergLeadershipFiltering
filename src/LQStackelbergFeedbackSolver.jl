using LinearAlgebra

# A helper function to compute P for all players at time t.
function compute_L_at_t(dyn_at_t::Dynamics, costs_at_t, Zₜ₊₁)

    num_players = size(costs_at_t)[1]
    num_states = xdim(dyn_at_t)
    A = dyn_at_t.A
    lhs_rows = Array{Float64}(undef, 0, num_states ÷ num_players)

    for player_idx in 1:num_players

        # Identify terms.
        B = dyn_at_t.Bs[player_idx]
        Rⁱⁱ = costs_at_t[player_idx].Rs[player_idx]

        # Compute terms for the matrices. First term is (*) in class notes, second is (**).
        first_term = Rⁱⁱ + B' *  Zₜ₊₁[player_idx] * B
        sum_of_other_player_control_matrices = sum(dyn_at_t.Bs) - B
        second_term = B' * Zₜ₊₁[player_idx] * sum_of_other_player_control_matrices

        # Create the LHS rows for the ith player and add to LHS rows.
        lhs_ith_row  = hcat([(i == player_idx) ? first_term : second_term for i in 1:num_players]...)
        lhs_rows = vcat(lhs_rows, lhs_ith_row)
    end

    # Construct the matrices we will use to solve for P.
    lhs_matrix = lhs_rows
    rhs_matrix_terms = [dyn_at_t.Bs[i]' * Zₜ₊₁[i] * A for i in 1:num_players]
    rhs_matrix = vcat(Array{Float64}(undef, 0, num_states), rhs_matrix_terms...)

    # Finally compute P.
    return lhs_matrix \ rhs_matrix
end

# TODO(hamzah) Implement an LQ Stackelberg game.

# Solve a finite horizon, discrete time LQ game to feedback Stackelberg equilibrium.
# Returns feedback matrices P[player][:, :, time]
export solve_lq_stackelberg_feedback
function solve_lq_stackelberg_feedback(
    dyn::Dynamics, costs::AbstractArray{Cost}, horizon::Int, leader_idx::Int)

    # TODO: Add checks for correct input lengths - they should match the horizon.
    num_players = size(costs)[1]
    # horizon = size(dyn)[1]
    # TODO(hamzah) If we ever go beyond a 2-player game, figure out multiple followers.
    follower_idx = (leader_idx == 2) ? 1 : 2
    num_states = xdim(dyn)

    # Define initial variables.
    Q̃ₜ₊₁ = [costs[i].Q for i in 1:num_players]

    # Define control variables which are the same over all horizon.
    # TODO(hamzah) Alter all of these to be time-indexed and move to loop.
    A = dyn.A
    B_leader = dyn.Bs[leader_idx]
    B_follower = dyn.Bs[follower_idx]
    Q_leader = costs[leader_idx].Q
    Q_follower = costs[follower_idx].Q

    # TODO: Change the incentives later, but for now it's identity since it must be positive definite.
    R₁₂ = 0.1 * I
    # This one can be 0.
    R₂₁ = I


    # Define recursive variables and initialize variables.
    Sₖ = [zeros(udim(dyn, i), num_states) for i in 1:num_players]
    all_Ls = [zeros(udim(dyn, i), num_states, horizon) for i in 1:num_players]
    all_Fs = [zeros(num_states, num_states) for i in 1:horizon]
    Mₖ₊₁ = zeros(num_states, num_states)

    # Compute recurrences at time K and K+1.
    T = inv(B_follower' * Q_leader * B_follower + I) * B_follower' * Q_follower

    common_tilde_term = I - B_follower * T
    Q̃ₜ₊₁[leader_idx] = common_tilde_term' * Q_leader * common_tilde_term + T' * R₁₂ * T
    Q̃ₜ₊₁[follower_idx] = Q_follower * common_tilde_term

    # S and L for followers not needed at horizon by game definition.
    Sₖ[leader_idx] = B_leader' * Q̃ₜ₊₁[leader_idx]
    L_leaderₖ = inv(I + Sₖ[leader_idx] * B_leader) * Sₖ[leader_idx] * A
    all_Ls[leader_idx][:, :, horizon] = L_leaderₖ
    Fₖ = A - B_leader * L_leaderₖ
    all_Fs[horizon] = Fₖ

    # This recursive variable will be at time K, used for time K-1.
    Mₖ₊₁ = Q_leader + Fₖ' * Q̃ₜ₊₁[leader_idx] * Fₖ + L_leaderₖ' * L_leaderₖ

    # t will incerment from 1 ... K-1. k will decrement from K-1 ... 1.
    for t = 1:horizon-1
        k = horizon - t

        # TODO(hamzah) When making the constants change each time step, put the definitions here.
        # Aₖ = ...

        # 1. Compute Sₖ for each player.
        innermost_term_l = R₁₂ + B_follower' * Mₖ₊₁ * B_follower
        inner_term_l = I - B_follower * inv(innermost_term_l) * B_follower' * Mₖ₊₁
        Sₖ[leader_idx] = B_leader' * Mₖ₊₁ * inner_term_l

        innermost_term_f = I + B_leader' * Mₖ₊₁ * B_leader
        inner_term_f = I - B_leader * inv(innermost_term_f) * B_leader' * Mₖ₊₁
        Sₖ[follower_idx] = B_follower' * Mₖ₊₁ * inner_term_f


        # 2. Compute feedback matrices Lₖ.
        L_leaderₖ = inv(I + Sₖ[leader_idx] * B_leader) * Sₖ[leader_idx] * A
        all_Ls[leader_idx][:, :, k] = L_leaderₖ

        L_followerₖ = inv(R₁₂ + Sₖ[follower_idx] * B_follower) * Sₖ[follower_idx] * A
        all_Ls[follower_idx][:, :, k] = L_followerₖ

        # 3. Compute state update matrices Fₖ.
        Fₖ = A - B_leader * L_leaderₖ - B_follower * L_followerₖ
        all_Fs[k] = Fₖ

        # 4. Compute cost weights M at time k, which becomes k+1 for the next iteration.
        Mₖ = Q_leader + Fₖ' * Mₖ₊₁ * Fₖ + L_leaderₖ' * L_leaderₖ + L_followerₖ' * R₁₂ * L_followerₖ
        Mₖ₊₁ = Mₖ 

        # recurse!
    end

    return all_Ls


    # M_next = I
    # S = [ for i in 1:num_players]
    # std::vector<Matrix> S(numPlayers);
    # std::vector<std::vector<Matrix>> Ls(numPlayers);
    # std::vector<Matrix> Fs(numPlayers);
    # Fs.resize(horizon, Matrix(num_states, num_states));
    # for(std::size_t i = 0; i < numPlayers; ++i)
    # {
    #     Ls[i].resize(horizon, Matrix(dyn[0].udim(i), dyn[0].xdim()));
    # }


    # # 1. Start at t=T, setting Z^i_t = Q^i_t.
    # # t = horizon
    # Zₜ₊₁ = [costs[i].Q for i in 1:num_players]
    # Zₜ = [zeros(size(Zₜ₊₁[i])) for i in 1:num_players]

    # num_states = xdim(dyn)
    # all_Ps = [zeros(udim(dyn, i), num_states, horizon) for i in 1:num_players]

    # while t > 1
    #     # 2. Decrement t, compute P^{i*}_t and Z^i_t.
    #     t -= 1

    #     # Compute Ps for all players at time t and store them.
    #     Ps = compute_L_at_t(dyn, costs, Zₜ₊₁)
    #     for i in 1:num_players
    #         num_inputs = udim(dyn, i)
    #         index_range = (i-1) * num_inputs + 1 : i * num_inputs

    #         all_Ps[i][:, :, t] = reshape(Ps[index_range, :], (num_inputs, xdim(dyn), 1))
    #     end

    #     for i in 1:num_players
    #         # Extract other values.
    #         Qₜ = [costs[i].Q for i in 1:num_players]
    #         Pⁱₜ = all_Ps[i][:, :, t]

    #         # Compute Z terms. There are no nonzero off diagonal Rij terms, so we just need to compute the terms with Rii.
    #         summation_1_terms = [Pⁱₜ' * costs[i].Rs[i][1] * Pⁱₜ]
    #         summation_1 = sum(summation_1_terms)

    #         summation_2_terms = [dyn.Bs[j] * all_Ps[j][:, :, t] for j in 1:num_players]
    #         summation_2 = dyn.A - sum(summation_2_terms)

    #         Zₜ[i] = Qₜ[i] - summation_1 + summation_2' * Zₜ₊₁[i] * summation_2
    #     end

    #     # Update Z_{t+1}
    #     Zₜ₊₁ = Zₜ

    #     # 3. Go to (2) until t=1.
    # end

    # return all_Ps
end
