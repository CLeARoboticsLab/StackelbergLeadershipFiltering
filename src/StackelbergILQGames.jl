# An algorithm that implements ILQGames for non-LQ Stackelberg games.

# struct StackelbergControlStrategy
#     T
#     Ss
#     Ls
# end

# function strategy_distance(strategy1::StackelbergControlStrategy, strategy2::StackelbergControlStrategy)

#     # Normalize so we avoid numerical issues. Q: Or does this introduce new numerical issues?
#     S1s = normalize.(strategy1.Ss)
#     L1s = normalize.(strategy1.Ls)
#     S2s = normalize.(strategy2.Ss)
#     L2s = normalize.(strategy2.Ls)

#     return sum(norm.(S1s - S2s) + norm.(L1s - L2s))
# end

# function stackelberg_2p_ilqgames_iteration(leader_idx::Int,
#                                            dyn::Dynamics,
#                                            costs::AbstractVector{<:Cost},
#                                            x0,
#                                            t0,
#                                            current_ctrl_strats::StackelbergControlStrategy,
#                                            x_refs::AbstractArray{Float64},
#                                            u_refs::AbstractVector{<:AbstractArray{Float64}})
#     N = num_agents(dyn)
#     T = current_ctrl_strats.T

#     # 1. Propagate the current control strategy forward in time using the nonlinear dynamics to get the state and
#     #    control trajectories.
#     x_hats, u_hats = unroll_feedback(dyn, times, FeedbackGainControlStrategy(current_ctrl_strats.Ss), x0)

#     # 2. Compute errors relative to reference trajectories.
#     dxs = x_refs - x_hats
#     dus = u_refs - u_hats

#     # 3. Create an LQ approximation of the game about the error state and controls.
#     # 4. Solve an LQ Stackelberg game given the approximated LQ conditions.
#     Ss, Ls = solve_approximated_lq_stackelberg_feedback(dyn, costs, T, t0, x_hats, u_hats, leader_idx)

#     # 5. Adjust the control strategies based on the approximate solution.
#     new_control_strats = StackelbergControlStrategy(T, Ss, Ls)

#     for tt in 1:T
#         for ii in 1:N
#             u_hats[ii][:, tt] -= Ss[ii][:, :, tt] * dxs[:, tt]
#         end
#     end

#     return new_control_strats, u_hats

# end

MAX_ITERS=100
THRESHOLD = 0.1
# function stackelberg_ilqgames(leader_idx::Int,
#                               T::Int,
#                               x0,
#                               t0,
#                               dyn::Dynamics,
#                               costs::AbstractVector{QuadraticCost},
#                               initial_ctrl_strats::StackelbergControlStrategy,
#                               x_refs::AbstractArray{Float64},
#                               u_refs::AbstractVector{<:AbstractArray{Float64}};
#                               threshold::Float64 = THRESHOLD)
#     MAX_ITERS = 100
#     num_iters = 0

#     N = num_agents(dyn)

#     prev_ctrl_strats = initial_ctrl_strats
#     current_ctrl_strats = initial_ctrl_strats
#     is_converged = false

#     function simple_iter(ctrl_strats::StackelbergControlStrategy, x_references, u_references)
#         return stackelberg_2p_ilqgames_iteration(leader_idx, dyn, costs, x0, t0, current_ctrl_strats, x_references, u_references)
#     end

#     while !is_converged && num_iters < MAX_ITERS
#         prev_ctrl_strats = current_ctrl_strats

#         # Q: What do I do with u_hat here?
#         current_ctrl_strats, u_hats = simple_iter(current_ctrl_strats, x_refs, u_refs)

#         is_converged = !(strategy_distance(current_ctrl_strats, prev_ctrl_strats) > threshold)
#         num_iters += 1
#     end

#     return current_ctrl_strats, is_converged, num_iters
# end


function stackelberg_ilqgames(leader_idx::Int,
              T::Int,
              t0,
              times,
              dyn::Dynamics,
              costs::AbstractVector{<:Cost},
              x₁::AbstractVector{Float64},
              us_1::AbstractVector{<:AbstractArray{Float64}};
              threshold::Float64 = THRESHOLD,
              max_iters = MAX_ITERS,
              step_size=1.0)

    num_players = num_agents(dyn)

    # TODO(hamzah) - Don't require the use of references as initial control trajectory.
    # TODO(hamzah) - update for affine ness
    xs_1 = unroll_raw_controls(dyn, times, us_1, x₁)

    xs_im1 = xs_1
    @assert all(xs_im1[:, 1] .== x₁)
    us_im1 = us_1

    costs = zeros(max_iters+1)
    costs[1] = evaluate(cost, xs_1, [us_1])

    num_x = xdim(dyn)
    num_us = [udim(dyn, ii) for ii in 1:num_players]
    num_iters = 0
    is_converged = false
    while !is_converged && num_iters < max_iters

        # I. Backwards pass

        # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
        lin_dyns = Array{LinearDynamics}(undef, T)
        quad_costs = [Array{PureQuadraticCost}(undef, T) for ii in 1:num_players]

        for tt in 1:T
            prev_time = (tt == 1) ? t0 : times[tt-1]
            curr_time = times[tt]
            time_range = (prev_time, curr_time)

            us_im1_tt = [us_im1[ii][:, tt] for ii in 1:num_players]
            lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_im1[:, tt], us_im1_tt)
            for ii in 1:num_players
                quad_costs[ii][tt] = quadraticize_costs(cost, time_range, xs_im1[:, tt], us_im1_tt)
            end
        end

        # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
        S̃s, _ = solve_lq_stackelberg_feedback(lin_dyns, quad_costs, T, leader_idx)

        # 3. Extract the feedback matrices from the homogeneous matrix.
        Ks = [S̃s[ii][1:num_us[ii], 1:num_x, :] for ii in 1:num_players]
        ks = [S̃s[ii][1:num_us[ii], num_x+1, :] for ii in 1:num_players]

        # II. Forward pass
        # TODO(hamzah) - turn this into a control strategy/generalize the other one.
        xs_i = zeros(size(xs_im1))
        xs_i[:, 1] = x₁
        us_i = zeros(size(us_im1))
        for tt in 1:T-1
            ttp1 = tt + 1
            prev_time = (tt == 1) ? t0 : times[tt]
            curr_time = times[ttp1]

            for jj in 1:num_players
                us_i[jj][:, tt] = us_im1[jj][:, tt] - Ks[jj][:, :, tt] * (xs_i[:, tt] - xs_im1[:, tt]) - step_size * ks[jj][:, tt]
            end
            us_i_tt = [us_i[ii][:, tt] for ii in 1:num_players]
            time_range = (prev_time, curr_time)
            xs_i[:, ttp1] = propagate_dynamics(dyn, time_range, xs_i[:, tt], us_i_tt)
        end

        # Final controls because why not...
        for jj in 1:num_players
            us_i[jj][:, T-1] = us_im1[jj][:, T-1] - Ks[jj][:, :, T-1] * (xs_i[:, T-1] - xs_im1[:, T-1]) - step_size * ks[jj][:, T-1]
        end

        x_norm_val = norm(xs_i - xs_im1)
        u_norm_val = norm(sum(us_i[ii] - us_im1[ii] for ii in 1:num_players))
        # is_converged = x_norm_val < threshold && u_norm_val < threshold

        # TODO(hmzh): Do we perform convergence based on quadratic cost in dx or quadratic cost in x^k?
        new_cost = evaluate(cost, xs_i, us_i)
        new_cost = evaluate(cost, xs_i - xs_im1, [us_i[jj] for jj in 1:num_players - us_im1[jj] for jj in 1:num_players])

        cost_im1 = costs[num_iters + 1]
        costs[num_iters + 2] = new_cost
        cost_diff = new_cost - cost_im1
        is_converged = abs(new_cost - cost_im1) < threshold

        # println("iteration ", num_iters, ": norms (xs, us): ", x_norm_val, " ", u_norm_val)
        println("iteration ", num_iters, ": costs (difference, new, old): ", new_cost - cost_im1, " ", new_cost, " ", cost_im1)

        xs_im1 = xs_i
        us_im1 = us_i

        num_iters +=1
    end

    xs_i = xs_im1
    us_i = us_im1

    return xs_i, us_i, is_converged, num_iters, costs
end
export ilqr

export StackelbergControlStrategy, stackelberg_ilqgames
