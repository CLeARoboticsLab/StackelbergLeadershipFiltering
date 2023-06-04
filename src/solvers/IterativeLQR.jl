
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
              step_size=1.0,
              verbose=false)

    @assert num_agents(dyn) == 1

    # Extract initial reference trajectory based on provided controls.
    xs_1 = unroll_raw_controls(dyn, times, [us_1], x₁)

    # Initialize variables to passed states/controls, validate initial state.
    xs_im1 = xs_1
    @assert all(xs_im1[:, 1] .== x₁)
    us_im1 = us_1

    # Initialize the cost J of the initial trajectory.
    convergence_metrics = zeros(max_iters+1)
    convergence_metrics[1] = 0.0

    evaluated_costs = zeros(max_iters+1)
    evaluated_costs[1] = evaluate(cost, xs_1, [us_1])

    num_x = size(xs_im1, 1)
    num_u = size(us_im1, 1)
    num_iters = 0
    is_converged = false
    while !is_converged && num_iters < max_iters

        # I. Backwards pass

        # 1. Extract linear dynamics and quadratic costs wrt to the current iteration of state and controls.
        lin_dyns = Array{LinearDynamics}(undef, T)
        quad_costs = Array{QuadraticCost}(undef, T)

        for tt in 1:T
            prev_time = (tt == 1) ? t0 : times[tt-1]
            curr_time = times[tt]
            time_range = (prev_time, curr_time)
            lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_im1[:, tt], [us_im1[:, tt]])
            quad_costs[tt] = quadraticize_costs(cost, time_range, xs_im1[:, tt], [us_im1[:, tt]])
        end

        # 2. Solve the optimal control problem wrt the linearizations to produce the homogeneous feedback and cost matrices.
        ctrl_strats, _ = solve_lqr_feedback(lin_dyns, quad_costs, T)

        # 3. Extract the feedback matrices from the homogeneous feedback matrix.
        Ks = ctrl_strats.Ps[1]
        ks = ctrl_strats.ps[1]

        # II. Forward pass
        xs_i = zeros(size(xs_im1))
        xs_i[:, 1] = x₁
        us_i = zeros(size(us_im1))
        for tt in 1:T-1
            ttp1 = tt + 1
            prev_time = (tt == 1) ? t0 : times[tt]
            curr_time = times[ttp1]

            # Run the update step for the controls.
            us_i[:, tt] = us_im1[:, tt] - Ks[:, :, tt] * (xs_i[:, tt] - xs_im1[:, tt]) - step_size * ks[:, tt]

            # Propagate the state at the next time step based on the new controls.
            time_range = (prev_time, curr_time)
            xs_i[:, ttp1] = propagate_dynamics(dyn, time_range, xs_i[:, tt], [us_i[:, tt]])
        end

        # Final controls because why not...
        us_i[:, T] = us_im1[:, T-1] - Ks[:, :, T-1] * (xs_i[:, T-1] - xs_im1[:, T-1]) - step_size * ks[:, T-1]

        # Check convergence based on the norm of the constant feedback term, which should satisfy the KKT conditions for optimality.
        evaluated_costs[num_iters+2] = evaluate(cost, xs_i, [us_i])
        cost_diff = evaluated_costs[num_iters+2] - evaluated_costs[num_iters+1]

        # Use norm of ks as the convergence metric.
        k_norm_conv_metric = norm(ks)

        # New convergence metric: maximum infinite norm difference between current and previous trajectory iteration.
        max_inf_norm_conv_metric = maximum(abs.(xs_i - xs_im1))
        
        # Another possible convergence metric is cost difference.
        conv_metric = max_inf_norm_conv_metric
        is_converged = conv_metric < threshold
        convergence_metrics[num_iters+2] = conv_metric

        # TODO(hmzh) - Implement line search here once derivatives are ready.

        if verbose
            conv_diff = convergence_metrics[num_iters+2] - convergence_metrics[num_iters+1]
            println("iteration ", num_iters, ": convergence metric (difference, new, old): ", conv_diff, " ", convergence_metrics[num_iters+2], " ", convergence_metrics[num_iters+1])
        end

        xs_im1 = xs_i
        us_im1 = us_i

        num_iters += 1
    end

    xs_i = xs_im1
    us_i = us_im1

    return xs_im1, us_im1, is_converged, num_iters, convergence_metrics, evaluated_costs
end
export ilqr
