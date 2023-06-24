# A file containing functions that solve LQ and approximated LQ Nash games for N players.
# TODO: This doesn't currently support cross-control costs.

# A helper function to compute P for all players at time t.
# TODO: The solver as currently written doesn't handle different sized control vectors. Fix this (p. 33 of class notes).
function compute_P_at_t(dyn_at_t::LinearDynamics, costs_at_t::AbstractVector{QuadraticCost}, Zₜ₊₁)

    num_players = num_agents(dyn_at_t)
    num_states = xhdim(dyn_at_t)
    A = get_homogenized_state_dynamics_matrix(dyn_at_t)

    # TODO: This line breaks very easily under weirdly-dimensioned systems. Change the code here.
    lhs_dim = uhdim(dyn_at_t)
    lhs_rows = Array{Float64}(undef, 0, lhs_dim)

    for ii in 1:num_players

        # Identify terms.
        Bⁱ = get_homogenized_control_dynamics_matrix(dyn_at_t, ii)
        Rⁱⁱ = get_homogenized_control_cost_matrix(costs_at_t[ii], ii)

        # Compute terms for the matrices. Self term is (*) in class notes, cross term is (**).
        lhs_ith_rows = Array{Float64}(undef, uhdim(dyn_at_t, ii), 0)
        for jj in 1:num_players
            Bʲ = get_homogenized_control_dynamics_matrix(dyn_at_t, jj)
            if ii == jj
                self_term = Rⁱⁱ + Bⁱ' *  Zₜ₊₁[ii] * Bⁱ
                lhs_ith_rows = hcat(lhs_ith_rows, self_term)
            else
                cross_term = Bⁱ' * Zₜ₊₁[ii] * Bʲ
                lhs_ith_rows = hcat(lhs_ith_rows, cross_term)
            end
        end

        # Create the LHS rows for the ith player and add to LHS rows.
        lhs_rows = vcat(lhs_rows, lhs_ith_rows)
    end

    # Construct the matrices we will use to solve for P.
    lhs_matrix = lhs_rows
    B(dyn, ii) = get_homogenized_control_dynamics_matrix(dyn, ii) 
    rhs_matrix_terms = [B(dyn_at_t, ii)' * Zₜ₊₁[ii] * A for ii in 1:num_players]
    rhs_matrix = vcat(rhs_matrix_terms...)

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

    # Ensure that all dynamics objects are discretized.
    # Ensure that the number of controls matches number of players at each horizon.
    for tt in 1:horizon
        @assert !is_continuous(dyns[tt]) string("Dynamics object at time ", tt, " should be discretized.")
        @assert(size(all_costs[tt], 1) == num_agents(dyns[tt]))
    end

    # The number of players, states, and control sizes are assumed to be constant over time.
    num_players = num_agents(dyns[1])
    num_states = xhdim(dyns[1])

    # Initialize the feedbacks gains and state costs.
    all_Zs = [zeros(num_states, num_states, horizon) for _ in 1:num_players]
    all_Ps = [zeros(uhdim(dyns[1], ii), num_states, horizon) for ii in 1:num_players]

    # 1. Start at the final timestep (t=T), setting Z^i_T = Q^i_T.
    Q(ii) = get_homogenized_state_cost_matrix(all_costs[horizon][ii]) 
    Zₜ₊₁ = [Q(ii) for ii in 1:num_players]
    Zₜ = [zeros(size(Zₜ₊₁[ii])) for ii in 1:num_players]

    for ii in 1:num_players
        all_Zs[ii][:, :, horizon] = get_homogenized_state_cost_matrix(all_costs[horizon][ii])
    end

    for tt = horizon-1:-1:1

        # Get the dynamics and costs at the current time.
        dyn = dyns[tt]
        costs = all_costs[tt]

        # 2. Compute P^{i*}_t and Z^i_t for all players at time t and store them.
        Ps = compute_P_at_t(dyn, costs, Zₜ₊₁)
        num_inputs = 0
        for ii in 1:num_players
            index_range = num_inputs + 1 : num_inputs + uhdim(dyn, ii)
            num_inputs += uhdim(dyn, ii)

            all_Ps[ii][:, :, tt] = reshape(Ps[index_range, :], (uhdim(dyn, ii), xhdim(dyn), 1))
        end

        for ii in 1:num_players
            # Extract other values.
            Qₜ = [get_homogenized_state_cost_matrix(costs[ii]) for i in 1:num_players]
            Pⁱₜ = all_Ps[ii][:, :, tt]

            # Compute Z terms. There are no nonzero off diagonal Rij terms, so we just need to compute the terms with Rii.
            summation_1_terms = [Pⁱₜ' * get_homogenized_control_cost_matrix(costs[ii], ii) * Pⁱₜ]
            summation_1 = sum(summation_1_terms)

            summation_2_terms = [get_homogenized_control_dynamics_matrix(dyn, jj) * all_Ps[jj][:, :, tt] for jj in 1:num_players]
            A = get_homogenized_state_dynamics_matrix(dyn)
            summation_2 = A - sum(summation_2_terms)

            Zₜ[ii] = Qₜ[ii] + summation_1 + summation_2' * Zₜ₊₁[ii] * summation_2
        end

        # Update Z_{t+1}
        for ii in 1:num_players
            all_Zs[ii][:, :, tt] = Zₜ[ii]
        end
        Zₜ₊₁ = Zₜ

        # 3. Go to (2) until t=1.
    end

    # Cut off the extra dimension of the homogenized coordinates system.
    extra_dim = xhdim(dyns[1])
    Ks = [all_Ps[ii][1:udim(dyns[1], ii), 1:xdim(dyns[1]), :] for ii in 1:num_players]
    ks = [all_Ps[ii][1:udim(dyns[1], ii), extra_dim, :] for ii in 1:num_players]

    Qs = [all_Zs[ii][1:xdim(dyns[1]), 1:xdim(dyns[1]), :] for ii in 1:num_players]
    qs = [all_Zs[ii][extra_dim, 1:xdim(dyns[1]), :] for ii in 1:num_players]
    cqs = [all_Zs[ii][extra_dim, extra_dim, :] for ii in 1:num_players]

    Ps_strategies = FeedbackGainControlStrategy(Ks, ks)
    Zs_future_costs = [[QuadraticCost(Qs[ii][:,:,tt], qs[ii][:, tt], cqs[ii][tt]) for tt in 1:horizon] for ii in 1:num_players]

    return Ps_strategies, Zs_future_costs
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

export solve_lq_nash_feedback
