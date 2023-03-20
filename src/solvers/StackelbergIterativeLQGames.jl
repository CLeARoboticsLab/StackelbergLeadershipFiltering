# Implements the Stackelberg ILQGames algorithm.

THRESHOLD = 1e-1
MAX_ITERS = 100
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
                              step_size=1.0,
                              verbose=false)

    num_players = num_agents(dyn)

    # Compute the initial states based on the initial controls.
    xs_1 = unroll_raw_controls(dyn, times, us_1, x₁)

    # Initialize the state iterations
    xs_km1 = xs_1
    @assert all(xs_km1[:, 1] .== x₁)
    us_km1 = us_1

    # Keeps track of the evaluated cost, though we use the sum of the constant feedback term's norms for the convergence.
    evaluated_costs = zeros(num_players, max_iters+1)
    evaluated_costs[:, 1] = [evaluate(costs[ii], xs_1, us_1) for ii in 1:num_players]

    convergence_metrics = zeros(num_players, max_iters+1)

    num_x = xdim(dyn)
    num_us = [udim(dyn, ii) for ii in 1:num_players]
    num_iters = 0
    is_converged = false

    while !is_converged && num_iters < max_iters

        ###########################
        #### I. Backwards pass ####
        ###########################
        # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
        lin_dyns = Array{LinearDynamics}(undef, T)
        quad_costs = [Array{QuadraticCost}(undef, num_players) for tt in 1:T]

        for tt in 1:T
            prev_time = (tt == 1) ? t0 : times[tt-1]
            curr_time = times[tt]
            time_range = (prev_time, curr_time)

            us_km1_tt = [us_km1[ii][:, tt] for ii in 1:num_players]
            lin_dyns[tt] = linearize_dynamics(dyn, time_range, xs_km1[:, tt], us_km1_tt)
            for ii in 1:num_players
                quad_costs[tt][ii] = quadraticize_costs(costs[ii], time_range, xs_km1[:, tt], us_km1_tt)
            end
            # println(norm(dyn.A - lin_dyns[1].A), norm(dyn.a - lin_dyns[1].a))
            # println(norm(costs[1].Q - quad_costs[1][1].Q), norm(costs[1].q - quad_costs[1][1].q), norm(costs[1].cq - quad_costs[1][1].cq))
            # println(norm(costs[2].Q - quad_costs[1][2].Q), norm(costs[2].q - quad_costs[1][2].q), norm(costs[2].cq - quad_costs[1][2].cq))
        end

         # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
        ctrl_strats, _ = solve_lq_stackelberg_feedback(lin_dyns, quad_costs, T, leader_idx)
        # ctrl_strats, _ = solve_lq_nash_feedback(lin_dyns, quad_costs, T)
        Ks = get_linear_feedback_gains(ctrl_strats)
        ks = get_constant_feedback_gains(ctrl_strats)
        # println("big K norms: ", norm(Ks[1]), " ", norm(Ks[2]))
        # println("little k norms: ", norm(ks[1]), " ", norm(ks[2]))

        # DELETE later - this shows that until this point, the algorithm works.
        # xs_km1, us_km1 = unroll_feedback(dyn, times, ctrl_strats, x₁)
        # is_converged = true
        # num_iters = 1
        # evaluated_costs[:, 2] = [evaluate(costs[ii], xs_km1, us_km1) for ii in 1:num_players]

        # convergence_metrics[1, 2] = norm(ks[1])^2
        # convergence_metrics[2, 2] = norm(ks[2])^2

        # return xs_km1, us_km1, is_converged, num_iters, convergence_metrics, evaluated_costs

        # ##########################
        # #### II. Forward pass ####
        # ##########################

        # TODO(hamzah) - turn this into a control strategy/generalize the other one.
        xs_k = zeros(size(xs_km1))
        xs_k[:, 1] = x₁
        us_k = [zeros(size(us_km1[ii])) for ii in 1:num_players]
        for tt in 1:T-1
            ttp1 = tt + 1
            prev_time = (tt == 1) ? t0 : times[tt]
            curr_time = times[ttp1]

            for ii in 1:num_players
                us_k[ii][:, tt] = us_km1[ii][:, tt] - Ks[ii][:, :, tt] * (xs_k[:, tt] - xs_km1[:, tt]) - step_size * ks[ii][:, tt]
            end
            us_k_tt = [us_k[ii][:, tt] for ii in 1:num_players]
            time_range = (prev_time, curr_time)
            xs_k[:, ttp1] = propagate_dynamics(dyn, time_range, xs_k[:, tt], us_k_tt)
        end

        # Final controls because why not...
        for ii in 1:num_players
            us_k[ii][:, T-1] = us_km1[ii][:, T-1] - Ks[ii][:, :, T-1] * (xs_k[:, T-1] - xs_km1[:, T-1]) - step_size * ks[ii][:, T-1]
        end

        #############################################
        #### III. Compute convergence and costs. ####
        #############################################

        # Compute the convergence metric to understand whether we are converged.
        for ii in 1:num_players
            convergence_metrics[ii, num_iters+1] = norm(ks[ii])^2
            # println("little k for player ", ii, ": ", norm(ks[ii])^2)
            # println("iter ", num_iters+1, " - player ", ii, ": ", convergence_metrics[ii, num_iters+1])
        end

        is_converged = sum(convergence_metrics[:, num_iters+1]) < threshold
        # is_converged = false

        # Evaluate and store the costs.
        evaluated_costs[:, num_iters+2] = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]

        if verbose
            old_metric = (num_iters == 0) ? 0. : sum(convergence_metrics[:, num_iters])
            new_metric = sum(convergence_metrics[:, num_iters+1])
            println("iteration ", num_iters, ": convergence metric (difference, new, old): ", new_metric - old_metric, " ", new_metric, " ", old_metric)
        end

        # if verbose
        #     old_metric = (num_iters == 0) ? 0. : sum(evaluated_costs[:, num_iters])
        #     new_metric = sum(evaluated_costs[:, num_iters+1])
        #     println("iteration ", num_iters, ": convergence metric (difference, new, old): ", new_metric - old_metric, " ", new_metric, " ", old_metric)
        # end

        xs_km1 = xs_k
        us_km1 = us_k

        num_iters +=1
    end

    return xs_km1, us_km1, is_converged, num_iters, convergence_metrics, evaluated_costs
end

export stackelberg_ilqgames