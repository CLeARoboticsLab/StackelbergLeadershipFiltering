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
                                       leader_idx::Int)

    # Ensure the number of dynamics and costs are the same as the horizon.
    @assert !isempty(dyns) && size(dyns, 1) == horizon
    @assert !isempty(all_costs) && size(all_costs, 1) == horizon

    # TODO(hamzah) If we ever go beyond a 2-player game, figure out multiple followers.
    follower_idx = (leader_idx == 2) ? 1 : 2
    num_players = num_agents(dyns[1])
    num_states = xhdim(dyns[1])

    # Define recursive variables and initialize variables - the number of players, states, and control sizes are assumed
    # to be constant over time.
    all_Ss = [zeros(uhdim(dyns[1], ii), num_states, horizon) for ii in 1:num_players]
    all_Ls = [zeros(num_states, num_states, horizon) for _ in 1:num_players]
    all_Ls[leader_idx][:, :, horizon] = all_costs[horizon][leader_idx].Q
    all_Ls[follower_idx][:, :, horizon] = all_costs[horizon][follower_idx].Q

    # t will increment from 1 ... K-1. k will decrement from K-1 ... 1.
    for tt = horizon-1:-1:1

        # Get the dynamics and costs at the current time.
        dyn = dyns[tt]
        costs = all_costs[tt]

        # Define control variables which are the same over all horizon.
        A = dyn.A
        B_leader = dyn.Bs[leader_idx]
        B_follower = dyn.Bs[follower_idx]

        Q_leader = costs[leader_idx].Q
        Q_follower = costs[follower_idx].Q

        R₁₁ = costs[leader_idx].Rs[leader_idx]
        R₂₂ = costs[follower_idx].Rs[follower_idx]
        R₁₂ = costs[leader_idx].Rs[follower_idx]
        R₂₁ = costs[follower_idx].Rs[leader_idx]

        Lₖ₊₁ = [all_Ls[leader_idx][:, :, tt+1], all_Ls[follower_idx][:, :, tt+1]]

        # Run recursive computation for one step.
        outputs = compute_stackelberg_recursive_step(A, B_leader, B_follower, Q_leader, Q_follower, Lₖ₊₁[leader_idx], Lₖ₊₁[follower_idx], R₁₁, R₂₂, R₁₂, R₂₁)

        all_Ss[leader_idx][:, :, tt] = outputs[1]
        all_Ss[follower_idx][:, :, tt] = outputs[2]
        all_Ls[leader_idx][:, :, tt] = outputs[3]
        all_Ls[follower_idx][:, :, tt] = outputs[4]
    end

    # Adjust the matrices so the output has xdim/udim number of dimensions, not xhdim/uhdim number of dims.
    out_Ss = [all_Ss[ii][1:udim(dyns[1], ii),:,:] for ii in 1:num_players]
    # out_Ls = [all_Ls[ii][1:xdim(dyns[1]),:,:] for ii in 1:num_players]
    L_future_costs = [[QuadraticCost(all_Ls[ii][:, :, tt]) for tt in 1:horizon] for ii in 1:num_players]
    return out_Ss, L_future_costs
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


# A function which accepts non-linear dynamics and non-quadratic costs and solves an LQ approximation at each timestep.
function solve_approximated_lq_stackelberg_feedback(dyn::Dynamics,
                                                    costs::AbstractVector{<:Cost},
                                                    horizon::Int,
                                                    t0::Float64,
                                                    x_refs::AbstractArray{Float64},
                                                    u_refs::AbstractVector{<:AbstractArray{Float64}},
                                                    leader_idx::Int)
    T = horizon
    N = num_agents(dyn)

    lin_dyns = Vector{LinearDynamics}(undef, T)
    all_quad_costs = Vector{Vector{QuadraticCost}}(undef, T)

    for tt in 1:T
        prev_time = t0 + ((tt == 1) ? 0 : tt-1)
        current_time = t0 + tt
        time_range = (prev_time, current_time)

        quad_costs = Vector{QuadraticCost}(undef, N)
        u_refs_at_tt = [u_refs[ii][:, tt] for ii in 1:N]

        # Linearize and quadraticize the dynamics/costs.
        lin_dyns[tt] = linearize_dynamics(dyn, time_range, x_refs[:, tt], u_refs_at_tt)
        for ii in 1:N
            quad_costs[ii] = quadraticize_costs(costs[ii], time_range, x_refs[:, tt], u_refs_at_tt)
        end
        all_quad_costs[tt] = quad_costs
    end

    return solve_lq_stackelberg_feedback(lin_dyns, all_quad_costs, T, leader_idx)
end

export compute_stackelberg_recursive_step, solve_lq_stackelberg_feedback, solve_approximated_lq_stackelberg_feedback
