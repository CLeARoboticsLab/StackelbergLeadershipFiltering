# A file containing functions that solve LQ and approximated LQ Nash games for N players.
# TODO: This doesn't currently support cross-control costs.

# A helper function to compute P for all players at time t.
# TODO: The solver as currently written doesn't handle different sized control vectors. Fix this (p. 33 of class notes).
function compute_P_at_t(dyn_at_t::LinearDynamics, costs_at_t::AbstractVector{QuadraticCost}, Zₜ₊₁)

    num_players = num_agents(dyn_at_t)
    num_states = xdim(dyn_at_t)
    A = dyn_at_t.A

    # TODO: This line breaks very easily under weirdly-dimensioned systems. Change the code here.
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

# Solve a finite horizon, discrete time LQ game to feedback Nash equilibrium.
# Returns feedback matrices P[player][:, :, time] and state costs matrices Z[player][:, :, time]
# The number of players, states, and control sizes are assumed to be constant over time.
function solve_lq_nash_feedback(
    dyns::AbstractVector{LinearDynamics}, all_costs::AbstractVector{<:AbstractVector{QuadraticCost}}, horizon::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(all_costs) && size(all_costs, 1) == horizon

    # Ensure that the number of controls matches number of players at each horizon.
    for tt in 1:horizon
        @assert(size(all_costs[tt], 1) == num_agents(dyns[tt]))
    end

    # The number of players, states, and control sizes are assumed to be constant over time.
    num_players = num_agents(dyns[1])
    num_states = xdim(dyns[1])

    # Initialize the feedbacks gains and state costs.
    all_Zs = [zeros(num_states, num_states, horizon) for _ in 1:num_players]
    all_Ps = [zeros(udim(dyns[1], ii), num_states, horizon) for ii in 1:num_players]

    # 1. Start at the final timestep (t=T), setting Z^i_T = Q^i_T.
    Zₜ₊₁ = [all_costs[horizon][ii].Q for ii in 1:num_players]
    Zₜ = [zeros(size(Zₜ₊₁[ii])) for ii in 1:num_players]

    for ii in 1:num_players
        all_Zs[ii][:, :, horizon] = all_costs[horizon][ii].Q
    end

    for tt = horizon-1:-1:1

        # Get the dynamics and costs at the current time.
        dyn = dyns[tt]
        costs = all_costs[tt]

        # 2. Compute P^{i*}_t and Z^i_t for all players at time t and store them.
        Ps = compute_P_at_t(dyn, costs, Zₜ₊₁)
        for ii in 1:num_players
            num_inputs = udim(dyn, ii)
            index_range = (ii-1) * num_inputs + 1 : ii * num_inputs

            all_Ps[ii][:, :, tt] = reshape(Ps[index_range, :], (num_inputs, xdim(dyn), 1))
        end

        for ii in 1:num_players
            # Extract other values.
            Qₜ = [costs[ii].Q for i in 1:num_players]
            Pⁱₜ = all_Ps[ii][:, :, tt]

            # Compute Z terms. There are no nonzero off diagonal Rij terms, so we just need to compute the terms with Rii.
            summation_1_terms = [Pⁱₜ' * costs[ii].Rs[ii] * Pⁱₜ]
            summation_1 = sum(summation_1_terms)

            summation_2_terms = [dyn.Bs[jj] * all_Ps[jj][:, :, tt] for jj in 1:num_players]
            summation_2 = dyn.A - sum(summation_2_terms)

            Zₜ[ii] = Qₜ[ii] + summation_1 + summation_2' * Zₜ₊₁[ii] * summation_2
        end

        # Update Z_{t+1}
        for ii in 1:num_players
            all_Zs[ii][:, :, tt] = Zₜ[ii]
        end
        Zₜ₊₁ = Zₜ

        # 3. Go to (2) until t=1.
    end

    return all_Ps, all_Zs
end

# Shorthand function for LQ time-invariant dynamics and costs.
function solve_lq_nash_feedback(dyn::LinearDynamics, costs::AbstractVector{QuadraticCost}, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    all_costs = [costs for _ in 1:horizon]
    return solve_lq_nash_feedback(dyns, all_costs, horizon)
end

function solve_lq_nash_feedback(dyn::LinearDynamics, all_costs::AbstractVector{<:AbstractVector{QuadraticCost}}, horizon::Int)
    dyns = [dyn for _ in 1:horizon]
    return solve_lq_nash_feedback(dyns, all_costs, horizon)
end

function solve_lq_nash_feedback(dyns::AbstractVector{LinearDynamics}, costs::AbstractVector{QuadraticCost}, horizon::Int)
    all_costs = [costs for _ in 1:horizon]
    return solve_lq_nash_feedback(dyns, all_costs, horizon)
end


# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lq_nash_feedback(dyn::Dynamics,
                                             costs::AbstractVector{<:Cost},
                                             horizon::Int,
                                             t0::Float64,
                                             x_refs::AbstractArray{Float64},
                                             u_refs::AbstractVector{<:AbstractArray{Float64}})
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Array{LinearDynamics}(undef, T)
    all_quad_costs = Vector{Vector{QuadraticCost}}(undef, T)

    for tt in 1:T
        quad_costs = Vector{QuadraticCost}(undef, N)
        u_refs_at_tt = [u_refs[ii][:, tt] for ii in 1:N]
        current_time = t0 + tt

        # Linearize and quadraticize the dynamics/costs.
        lin_dyns[tt] = linearize_dynamics(dyn, current_time, x_refs[:, tt], u_refs_at_tt)
        for ii in 1:N
            quad_costs[ii] = quadraticize_costs(costs[ii], current_time, x_refs[:, tt], u_refs_at_tt)
        end
        all_quad_costs[tt] = quad_costs
    end

    return solve_lq_nash_feedback(lin_dyns, all_quad_costs, T)
end

export solve_lq_nash_feedback, solve_approximated_lq_nash_feedback
