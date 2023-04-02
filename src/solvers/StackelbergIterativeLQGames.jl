# Implements the Stackelberg ILQGames algorithm.

# A particle filter that can run iteratively on command, with new arguments (below) each round:
# - Particle Filter object
# - measurement + uncertainty z, R
# - f_dynamics
# - h_measurement model
# - discrete state transition
# - u_inputs for the iteration
# TODO(hamzah) - add particle set class?
# It should contain all the context and derivative data from the filter from time 1 until the most recent time it was
# stepped to.
mutable struct SILQGamesObject
    # information that allows multiple runs
    num_runs::Int       # anticipated number of runs; will be used for allocating space for the data
    current_idx::Int

    # inputs that won't change
    leader_idx::Int
    horizon::Int
    dyn::Dynamics
    costs::AbstractVector{<:Cost}

    # configuration
    threshold::Float64
    max_iters
    step_size
    verbose

    # iteration data for states and controls
    xks # num_runs x max_iters x num_states x num_times 4D data on the kth iteration of the run
    uks # [num_runs x max_iters x num_ctrls(ii) x num_times]

    # track the derivative data we are interested in
    Kks  # [num_runs x max_iters x num_ctrls(ii) x num_states x num_times]
    kks  # [num_runs x max_iters x num_ctrls(ii) x num_times]
    convergence_metrics # num_runs x max_iters
    num_iterations      # num_runs
    evaluated_costs     # num_runs x num_players x max_iters+1
end

THRESHOLD = 1e-1
MAX_ITERS = 100
# initialize_silq_games_object - initializes an SILQ Games Object which contains the data required within the algorithm
function initialize_silq_games_object(num_runs, leader_idx, horizon, dyn::Dynamics, costs::AbstractVector{<:Cost};
                                      threshold::Float64 = THRESHOLD, max_iters = MAX_ITERS, step_size=1.0, verbose=false)
    num_players = num_agents(dyn)
    @assert length(costs) == num_players
    num_states = xdim(dyn)

    xks = zeros(num_runs, max_iters+1, num_states, horizon)
    uks = [zeros(num_runs, max_iters+1, udim(dyn, ii), horizon) for ii in 1:num_players]

    Kks = [zeros(num_runs, max_iters, udim(dyn, ii), num_states, horizon) for ii in 1:num_players]
    kks = [zeros(num_runs, max_iters, udim(dyn, ii), horizon) for ii in 1:num_players]
    
    num_iterations = zeros(Int, num_runs)
    evaluated_costs = zeros(num_runs, num_players, max_iters+1)
    convergence_metrics = zeros(num_runs, num_players, max_iters+1)

    current_idx = 0
    return SILQGamesObject(num_runs, current_idx,
                           leader_idx, horizon, dyn, costs,
                           threshold, max_iters, step_size, verbose,
                           xks, uks,
                           Kks, kks, convergence_metrics, num_iterations, evaluated_costs)
end
export initialize_silq_games_object


function stackelberg_ilqgames(sg::SILQGamesObject,
                              t0,
                              times,
                              x₁::AbstractVector{Float64},
                              us_1::AbstractVector{<:AbstractArray{Float64}})
    T = sg.horizon
    sg.current_idx += 1
    if sg.current_idx > sg.num_runs
        error("too many runs - please provide fresh SILQGames object.")
    end

    num_players = num_agents(sg.dyn)
    num_x = xdim(sg.dyn)
    num_us = [udim(sg.dyn, ii) for ii in 1:num_players]

    # Compute the initial states based on the initial controls.
    xs_1 = unroll_raw_controls(sg.dyn, times, us_1, x₁)

    # Initialize the state iterations
    xs_km1 = xs_1
    @assert all(xs_km1[:, 1] .== x₁)
    us_km1 = us_1

    # Keeps track of the evaluated cost, though we use the sum of the constant feedback term's norms for the convergence.
    evaluated_costs = zeros(num_players, sg.max_iters+1)
    evaluated_costs[:, 1] = [evaluate(sg.costs[ii], xs_1, us_1) for ii in 1:num_players]
    convergence_metrics = zeros(num_players, sg.max_iters+1)

    # Fill in debug data.
    sg.xks[sg.current_idx, 1, :, :] = xs_1
    for ii in 1:num_players
        sg.uks[ii][sg.current_idx, 1, :, :] = us_1[ii]
    end

    num_iters = 0
    is_converged = false

    while !is_converged && num_iters < sg.max_iters

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
            lin_dyns[tt] = linearize_dynamics(sg.dyn, time_range, xs_km1[:, tt], us_km1_tt)
            for ii in 1:num_players
                quad_costs[tt][ii] = quadraticize_costs(sg.costs[ii], time_range, xs_km1[:, tt], us_km1_tt)
            end
        end

         # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
        ctrl_strats, _ = solve_lq_stackelberg_feedback(lin_dyns, quad_costs, T, sg.leader_idx)
        Ks = get_linear_feedback_gains(ctrl_strats)
        ks = get_constant_feedback_gains(ctrl_strats)

        ##########################
        #### II. Forward pass ####
        ##########################

        # TODO(hamzah) - turn this into a control strategy/generalize the other one.
        xs_k = zeros(size(xs_km1))
        xs_k[:, 1] = x₁
        us_k = [zeros(size(us_km1[ii])) for ii in 1:num_players]
        for tt in 1:T-1
            ttp1 = tt + 1
            prev_time = (tt == 1) ? t0 : times[tt]
            curr_time = times[ttp1]

            for ii in 1:num_players
                us_k[ii][:, tt] = us_km1[ii][:, tt] - Ks[ii][:, :, tt] * (xs_k[:, tt] - xs_km1[:, tt]) - sg.step_size * ks[ii][:, tt]
            end
            us_k_tt = [us_k[ii][:, tt] for ii in 1:num_players]
            time_range = (prev_time, curr_time)
            xs_k[:, ttp1] = propagate_dynamics(sg.dyn, time_range, xs_k[:, tt], us_k_tt)
        end

        #############################################
        #### III. Compute convergence and costs. ####
        #############################################

        # Compute the convergence metric to understand whether we are converged.
        for ii in 1:num_players
            convergence_metrics[ii, num_iters+1] = norm(ks[ii])^2
        end

        is_converged = sum(convergence_metrics[:, num_iters+1]) < sg.threshold

        # Evaluate and store the costs.
        evaluated_costs[:, num_iters+2] = [evaluate(sg.costs[ii], xs_k, us_k) for ii in 1:num_players]

        if sg.verbose
            old_metric = (num_iters == 0) ? 0. : sum(convergence_metrics[:, num_iters])
            new_metric = sum(convergence_metrics[:, num_iters+1])
            println("iter ", num_iters, ": convergence metric (diff, new, old): ", round(new_metric - old_metric, digits=8), " ", round(new_metric, digits=8), " ", round(old_metric, digits=8))
        end

        # Fill in debug data.
        sg.xks[sg.current_idx, num_iters+2, :, :] = xs_k
        for ii in 1:num_players
            sg.uks[ii][sg.current_idx, num_iters+2, :, :] = us_k[ii]
            sg.Kks[ii][sg.current_idx, num_iters+1, :, :, :] = Ks[ii]
            sg.kks[ii][sg.current_idx, num_iters+1, :, :] = ks[ii]
        end

        xs_km1 = xs_k
        us_km1 = us_k

        num_iters +=1
    end

    # Update debug data.
    sg.num_iterations[sg.current_idx] = num_iters
    sg.evaluated_costs[sg.current_idx, :, :] = evaluated_costs
    sg.convergence_metrics[sg.current_idx, :, :] = convergence_metrics

    # Return the results from the current run of the algorithm for convenience.
    return xs_km1, us_km1, is_converged, num_iters, convergence_metrics, evaluated_costs
end

export stackelberg_ilqgames
