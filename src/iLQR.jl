

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
#     x_hats, u_hats = unroll_feedback(dyn, current_ctrl_strats.Ss, x0)

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

THRESHOLD = 0.01
MAX_ITERS = 100
function ilqr(T::Int,
              x0,
              t0,
              times,
              dyn::Dynamics,
              cost::Cost,
              xs_1::AbstractArray{Float64},
              us_1::AbstractArray{Float64};
              threshold::Float64 = THRESHOLD,
              max_iters = MAX_ITERS,
              step_size=1.0)
    num_iters = 0

    @assert num_agents(dyn) == 1

    # TODO(hamzah) - Don't require the use of references as initial control trajectory.
    # TODO(hamzah) - update for affine ness
    xs_im1 = vcat(xs_1, ones(1, T))
    xs_i = xs_im1
    us_im1 = vcat(us_1, ones(1, T))
    us_i = us_im1

    is_converged = false
    while !is_converged && num_iters < max_iters

        num_x = size(xs_i, 1)
        num_u = size(us_i, 1)

        # I. Backwards pass

        # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
        lin_dyns = Array{LinearDynamics}(undef, T)
        quad_costs = Array{QuadraticCost}(undef, T)

        for tt in 1:T
            prev_time = (tt == 1) ? t0 : times[tt-1]
            curr_time = times[tt]
            time_range = (prev_time, curr_time)
            lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_i[:, tt], [us_i[:, tt]]; should_homogenize=true)
            quad_costs[tt] = quadraticize_costs(cost, time_range, xs_i[:, tt], [us_i[:, tt]])
        end

        # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
        K̃s, _ = solve_lqr_feedback(lin_dyns, quad_costs, T)

        # 3. Extract the feedback matrices from the homogeneous matrix.
        Ks = K̃s[1:num_x, 1:num_x, :]
        ks = K̃s[num_x+1, 1:num_u, :][:, :]


        # II. Forward pass
        # TODO(hamzah) - turn this into a control strategy/generalize the other one.
        for tt in 2:T
            ttm1 = tt - 1
            prev_time = (tt == 1) ? t0 : times[ttm1]
            curr_time = times[tt]

            us_i[:, ttm1] = us_i[:, ttm1] - K[:, :, ttm1] * (xs_i[:, ttm1] - xs_im1[:, ttm1]) - step_size * ks[:, ttm1]
            time_range = (prev_time, curr_time)
            xs_i[:, tt] = propagate_dynamics(dyn, time_range, xs_i[:, ttm1], [us_i[:, ttm1]])
        end


        # 1. Can't use this because it doesn't allow for quadraticizing/linearizing given state and forward pass with dx, du.
        # 2. Actually, we can, beacuse the LQR solver deals with feedback matrices that don't use x, u.
        # Ps, Zs = solve_approximated_lqr_feedback(dyn, cost, T, t0, xs_i, us_i)
        # ctrl_strat = FeedbackGainControlStrategy(Ps)


        # Compute the differences.
        # dxs = x_hats - xs_i
        # dus = u_hats - us_i

        # Apply feedback in forward direction.
        # dx0 = dxs[:, 1]
        # xs_i_new, us_i_new = unroll_feedback(dyn, ctrl_strat, xs_i[:, 1])

        # # Q: What do I do with u_hat here?
        # current_ctrl_strats, u_hats = simple_iter(current_ctrl_strats, x_refs, u_refs)

        is_converged = !(norm(xs_i - xs_im1) > threshold && norm(us_i - us_im1) > threshold)
        num_iters += 1

        xs_im1 = xs_i
        us_im1 = us_i
    end

    return xs_i, us_i, is_converged, num_iters
end
export ilqr
