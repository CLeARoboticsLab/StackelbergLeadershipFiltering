
THRESHOLD = 1e-1
MAX_ITERS = 100
function ilqr(T::Int,
              t0,
              times,
              dyn::Dynamics,
              cost::Cost,
              x₁::AbstractVector{Float64},
              us_1::AbstractArray{Float64};
              threshold::Float64 = THRESHOLD,
              max_iters = MAX_ITERS,
              step_size=1.0)

    @assert num_agents(dyn) == 1

    # TODO(hamzah) - Don't require the use of references as initial control trajectory.
    # TODO(hamzah) - update for affine ness
    xs_1 = unroll_raw_controls(dyn, times, [us_1], x₁)

    xs_im1 = xs_1
    @assert all(xs_im1[:, 1] .== x₁)
    us_im1 = us_1

    costs = zeros(max_iters+1)
    costs[1] = evaluate(cost, xs_1, [us_1])

    num_x = size(xs_im1, 1)
    num_u = size(us_im1, 1)
    num_iters = 0
    is_converged = false
    while !is_converged && num_iters < max_iters

        # I. Backwards pass

        # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
        lin_dyns = Array{LinearDynamics}(undef, T)
        quad_costs = Array{PureQuadraticCost}(undef, T)

        for tt in 1:T
            prev_time = (tt == 1) ? t0 : times[tt-1]
            curr_time = times[tt]
            time_range = (prev_time, curr_time)
            lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_im1[:, tt], [us_im1[:, tt]])
            quad_costs[tt] = quadraticize_costs(cost, time_range, xs_im1[:, tt], [us_im1[:, tt]])
        end

        # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
        K̃s, _ = solve_lqr_feedback(lin_dyns, quad_costs, T)

        # 3. Extract the feedback matrices from the homogeneous matrix.
        Ks = K̃s[1:num_u, 1:num_x, :]
        ks = K̃s[1:num_u, num_x+1, :]

        # II. Forward pass
        # TODO(hamzah) - turn this into a control strategy/generalize the other one.
        xs_i = zeros(size(xs_im1))
        xs_i[:, 1] = x₁
        us_i = zeros(size(us_im1))
        for tt in 1:T-1
            ttp1 = tt + 1
            prev_time = (tt == 1) ? t0 : times[tt]
            curr_time = times[ttp1]

            # println("iter ", num_iters, " - dx @t=", tt, ": \n", xs_i[:, tt], "\n", xs_im1[:, tt], "\n", xs_i[:, tt] - xs_im1[:, tt])
            us_i[:, tt] = us_im1[:, tt] - Ks[:, :, tt] * (xs_i[:, tt] - xs_im1[:, tt]) - step_size * ks[:, tt]
            time_range = (prev_time, curr_time)
            xs_i[:, ttp1] = propagate_dynamics(dyn, time_range, xs_i[:, tt], [us_i[:, tt]])
        end

        # Final controls because why not...
        us_i[:, T] = us_im1[:, T-1] - Ks[:, :, T-1] * (xs_i[:, T-1] - xs_im1[:, T-1]) - step_size * ks[:, T-1]

        x_norm_val = norm(xs_i - xs_im1)
        u_norm_val = norm(us_i - us_im1)
        # is_converged = x_norm_val < threshold && u_norm_val < threshold

        # TODO(hmzh): Do we perform convergence based on quadratic cost in dx or quadratic cost in x^k?
        new_cost = evaluate(cost, xs_i, [us_i])
        new_cost = norm(ks)

        cost_im1 = costs[num_iters + 1]
        costs[num_iters + 2] = new_cost
        cost_diff = new_cost - cost_im1
        
        is_converged = abs(cost_diff) < threshold
        is_converged = norm(ks) < threshold

        # if cost_diff > 0
        #     step_size /= 2.
        # end

        # println("iteration ", num_iters, ": norms (xs, us): ", x_norm_val, " ", u_norm_val)
        println("iteration ", num_iters, ": costs (difference, new, old): ", cost_diff, " ", new_cost, " ", cost_im1)
        # println("norm of ks: ", norm(ks))

        xs_im1 = xs_i
        us_im1 = us_i

        num_iters += 1
    end

    xs_i = xs_im1
    us_i = us_im1

    return xs_i, us_i, is_converged, num_iters, costs
end
export ilqr


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
