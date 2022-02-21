# A function containing useful but otherwise homeless code.

# A helper function to compute P for all players at time t in a team Stackelberg game with joint cost function
# optimization.
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