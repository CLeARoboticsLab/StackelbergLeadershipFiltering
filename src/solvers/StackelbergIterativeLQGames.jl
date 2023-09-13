# Implements the Stackelberg ILQGames algorithm.

using BenchmarkTools

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
    horizon::Int
    dyn::Dynamics
    costs::AbstractVector{<:Cost}

    # configuration
    threshold::Float64
    max_iters::Int

    step_size::Float64          # initial step size α₀
    ss_reduce::Float64          # reduction factor τ
    α_min::Float64              # min step size
    max_linesearch_iters::Int   # maximum number of linesearch iterations
    check_valid::Function       # for checking constraints in backstepping, f(xs, us, ts)

    verbose

    # regularization config
    state_reg_param::Float64
    control_reg_param::Float64
    ensure_pd::Bool

    # who was the leader each time we ran
    leader_idxs::AbstractVector{<:Int}

    # iteration data for states and controls
    xks # num_runs x max_iters x num_states x num_times 4D data on the kth iteration of the run
    uks # [num_runs x max_iters x num_ctrls(ii) x num_times]

    # track the derivative data we are interested in
    Kks  # [num_runs x max_iters x num_ctrls(ii) x num_states x num_times]
    kks  # [num_runs x max_iters x num_ctrls(ii) x num_times]
    convergence_metrics # num_runs x num_players x max_iters
    num_iterations      # num_runs
    evaluated_costs     # num_runs x num_players x max_iters+1

    # timing information
    timings # num_runs x max_iters
end

THRESHOLD = 1e-2
MAX_ITERS = 100
# initialize_silq_games_object - initializes an SILQ Games Object which contains the data required within the algorithm
function initialize_silq_games_object(num_runs, horizon, dyn::Dynamics, costs::AbstractVector{<:Cost};
                                      state_reg_param=1e-2, control_reg_param=1e-2, ensure_pd=true,
                                      threshold::Float64 = THRESHOLD, max_iters = MAX_ITERS,
                                      step_size=1.0, ss_reduce=1e-2, α_min=1e-2, max_linesearch_iters=10,
                                      check_valid=(xs, us, ts)->true, verbose=false, ignore_Kks=true, ignore_xkuk_iters=true)
    num_players = num_agents(dyn)
    @assert length(costs) == num_players
    num_states = xdim(dyn)

    if !ignore_xkuk_iters
        xks = zeros(num_runs, max_iters+1, num_states, horizon)
        uks = [zeros(num_runs, max_iters+1, udim(dyn, ii), horizon) for ii in 1:num_players]
    else
        xks = zeros(num_runs, num_states, horizon)
        uks = [zeros(num_runs, udim(dyn, ii), horizon) for ii in 1:num_players]
    end

    Kks = []
    kks = []
    if !ignore_Kks
        Kks = [zeros(num_runs, max_iters, udim(dyn, ii), num_states, horizon) for ii in 1:num_players]
        kks = [zeros(num_runs, max_iters, udim(dyn, ii), horizon) for ii in 1:num_players]
    end

    leader_idxs = zeros(Int, num_runs)

    num_iterations = zeros(Int, num_runs)
    evaluated_costs = zeros(num_runs, num_players, max_iters+1)
    convergence_metrics = zeros(num_runs, num_players, max_iters+1)

    timings = zeros(num_runs, max_iters)

    current_idx = 0
    return SILQGamesObject(num_runs, current_idx,
                           horizon, dyn, costs,
                           threshold, max_iters, 
                           step_size, ss_reduce, α_min, max_linesearch_iters, 
                           check_valid, verbose,
                           state_reg_param, control_reg_param, ensure_pd,
                           leader_idxs,
                           xks, uks,
                           Kks, kks, convergence_metrics, num_iterations, evaluated_costs,
                           timings)
end
export initialize_silq_games_object

function backward_pass(sg, leader_idx, t0, times, xs_km1, us_km1)
    T = sg.horizon

    # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
    lin_dyns = Array{LinearDynamics}(undef, T)
    quad_costs = [Array{QuadraticCost}(undef, num_agents(sg.dyn)) for tt in 1:T]

    for tt in 1:T
        prev_time = (tt == 1) ? t0 : times[tt-1]
        curr_time = times[tt]
        time_range = (prev_time, curr_time)

        us_km1_tt = [us_km1[ii][:, tt] for ii in 1:num_agents(sg.dyn)]

        # Produce a continuous-time linear system from the dynamical system,
        # then discretize it at the sampling time of the original system.
        # Finally, regularize the quadratic cost terms inside.
        cont_lin_dyn = linearize(sg.dyn, time_range, xs_km1[:, tt], us_km1_tt)
        lin_dyns[tt] = discretize(cont_lin_dyn, sampling_time(sg.dyn))
        for ii in 1:num_players
            quad_costs[tt][ii] = quadraticize_costs(sg.costs[ii], time_range, xs_km1[:, tt], us_km1_tt)
        end
    end

    # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.
    ctrl_strats, _ = solve_lq_stackelberg_feedback(lin_dyns, quad_costs, T, leader_idx; state_reg_param=sg.state_reg_param, control_reg_param=sg.control_reg_param, ensure_pd=sg.ensure_pd)
    return ctrl_strats
end


function forward_pass(sg, t0, times, step_size, xs_km1, us_km1, Ks, ks)
    # TODO(hamzah) - turn this into a control strategy/generalize the other one.
    xs_k = zeros(size(xs_km1))
    xs_k[:, 1] = xs_km1[:, 1] # initial state should be same across all iterations
    us_k = [zeros(size(us_km1[ii])) for ii in 1:num_agents(sg.dyn)]

    for tt in 1:sg.horizon-1
        ttp1 = tt + 1
        prev_time = (tt == 1) ? t0 : times[tt]
        curr_time = times[ttp1]

        for ii in 1:num_agents(sg.dyn)
            us_k[ii][:, tt] = us_km1[ii][:, tt] - Ks[ii][:, :, tt] * (xs_k[:, tt] - xs_km1[:, tt]) - step_size * ks[ii][:, tt]
        end
        us_k_tt = [us_k[ii][:, tt] for ii in 1:num_agents(sg.dyn)]
        time_range = (prev_time, curr_time)
        xs_k[:, ttp1] = propagate_dynamics(sg.dyn, time_range, xs_k[:, tt], us_k_tt)
    end
    return xs_k, us_k
end


# # TODO(hamzah) - Implement the Armijo condition. Requires computing the gradient at each time step for each cost.
# function compute_armijo_condition()

#     # NOTE: As of 07/16/23, this function passes in discrete times to compute_cost, not continuous times.
#     adjusted_func_evals = [evaluate(sg.costs[ii], xs, us) for ii in 1:num_agents(sg.dyn)]
#     c₁ = 0.5 # for now, TODO(hamzah): move to be configurable
#     grad_Js = Gus(sg.costs[ii], time_range, xs[:, tt], us_at_tt)
#     sufficient_decreases = [evaluate(sg.costs[ii], xs_km1, us_km1) + c₁ * α *  for ii in 1:num_agents(sg.dyn)]
    
#     for tt in 1:sg.horizon
#         for ii in 1:num_agents(sg.dyn)
#             grad_Js = Gus(sg.costs[ii], )
#             new_us_tt = [us[ii][:, tt] + τ *  for ii in 1:num_agents(sg.dyn)]

#             J[ii](u0 + t * grad_J) > J[ii](u0) + alpha * t * dot(grad_J[ii], grad_J[ii])
#         end
#     end
# end

# TODO(hamzah) - Implement the Armijo condition. Requires computing the gradient at each time step for each cost.
# TODO(hamzah) - Assumes gradient descent direction.
function validating_line_search(sg, t0, times, xs_km1, us_km1, Ks, ks; step_size_override=nothing)
    is_valid = false
    α = (isnothing(step_size_override)) ? sg.step_size : step_size_override

    iters = 0

    # Generate a forward pass trajectory with the initial step size.
    xs, us = forward_pass(sg, t0, times, α, xs_km1, us_km1, Ks, ks)

    while !sg.check_valid(xs, us, times) # if invalid trajectory, repeat
        # Shorten the step size and rerun the forward pass.
        α = max(sg.ss_reduce * α, sg.α_min)
        xs, us = forward_pass(sg, t0, times, α, xs_km1, us_km1, Ks, ks)
        
        # Update iteration count.
        iters += 1
        if iters > sg.max_linesearch_iters
            error("exceeded $(sg.max_linesearch_iters) in linesearch")
        end 
    end

    return xs, us
end

function stackelberg_ilqgames(sg::SILQGamesObject,
                              leader_idx::Int,
                              t0,
                              times,
                              x₁::AbstractVector{Float64},
                              us_1::AbstractVector{<:AbstractArray{Float64}};
                              manual_idx=nothing) # for use with distributed computing
    if isnothing(manual_idx)
        sg.current_idx += 1
    end

    # Setting this variable allows for distributed processing.
    current_idx = (isnothing(manual_idx)) ? sg.current_idx : manual_idx
    if current_idx > sg.num_runs
        error("too many runs - please provide fresh SILQGames object.")
    end

    # Store which actor was assumed to be leader.
    sg.leader_idxs[current_idx] = leader_idx

    num_players = num_agents(sg.dyn)
    num_x = xdim(sg.dyn)
    num_us = [udim(sg.dyn, ii) for ii in 1:num_players]

    # Compute the initial states based on the initial controls.
    xs_1 = unroll_raw_controls(sg.dyn, times, us_1, x₁)
    is_initial_trajectory_valid = sg.check_valid(xs_1, us_1, times)
    @assert is_initial_trajectory_valid "Provided control trajectory does not results in valid state trajectory."

    # Initialize the state iterations
    xs_km1 = xs_1
    if norm(xs_km1 - xs_1) != 0
        println("init diff should be 0: ", xs_km1, " ", xs_1)
    end
    @assert all(xs_km1[:, 1] .== x₁)
    us_km1 = us_1

    # Keeps track of the evaluated cost, though we use the sum of the constant feedback term's norms for the convergence.
    evaluated_costs = zeros(num_players, sg.max_iters+1)
    evaluated_costs[:, 1] = [evaluate(sg.costs[ii], xs_1, us_1) for ii in 1:num_players]
    convergence_metrics = zeros(num_players, sg.max_iters+1)

    # Fill in debug data.
    ignore_xkuk_iters = ndims(sg.xks) == 3 # if we want to save all of them, it should be 4
    should_store_feedback = !isempty(sg.Kks)

    if !ignore_xkuk_iters
        sg.xks[current_idx, 1, :, :] = xs_1
        for ii in 1:num_players
            sg.uks[ii][current_idx, 1, :, :] = us_1[ii]
        end
    else
        sg.xks[current_idx, :, :] = xs_1
        for ii in 1:num_players
            sg.uks[ii][current_idx, :, :] = us_1[ii]
        end
    end

    num_iters = 0
    is_converged = false

    while !is_converged && num_iters < sg.max_iters
        # Save timing information for the iteration.
        time_val = @elapsed begin
            ###########################
            #### I. Backwards pass ####
            ###########################
            # 1. Extract linear dynamics and quadratic costs wrt to the current guess for the state and controls.
                # 2. Solve the optimal control problem wrt δx to produce the homogeneous feedback and cost matrices.

            ctrl_strats = backward_pass(sg, leader_idx, t0, times, xs_km1, us_km1)
            Ks = get_linear_feedback_gains(ctrl_strats)
            ks = get_constant_feedback_gains(ctrl_strats)

            ##########################
            #### II. Forward pass ####
            ##########################

            # On the first iteration, choose a step size of 1.
            step_size_override = iszero(num_iters) ? 1. : nothing
            # xs_k, us_k = forward_pass(sg, t0, times, step_size, xs_km1, us_km1, Ks, ks)
            xs_k, us_k = validating_line_search(sg, t0, times, xs_km1, us_km1, Ks, ks; step_size_override)

            #############################################
            #### III. Compute convergence and costs. ####
            #############################################

            # Compute the convergence metric to understand whether we are converged.
            for ii in 1:num_players
                convergence_metrics[ii, num_iters+1] = norm(ks[ii])^2
            end
            # New convergence metric: maximum infinite norm difference between current and previous trajectory iteration.
            convergence_metrics[:, num_iters+1] .= 1e-8
            max_inf_norm_conv_metric = maximum(abs.(xs_k - xs_km1))
            convergence_metrics[1, num_iters+1] = max_inf_norm_conv_metric

            is_converged = sum(convergence_metrics[:, num_iters+1]) < sg.threshold

            # Evaluate and store the costs.
            evaluated_costs[:, num_iters+2] = [evaluate(sg.costs[ii], xs_k, us_k) for ii in 1:num_players]
            # is_converged = abs(sum(evaluated_costs[:, num_iters+1] - evaluated_costs[:, num_iters+2])) < sg.threshold

            if sg.verbose
                old_metric = (num_iters == 0) ? 0. : sum(convergence_metrics[:, num_iters])
                new_metric = sum(convergence_metrics[:, num_iters+1])
                println("iter ", num_iters, ": convergence metric (diff, new, old): ", round(new_metric - old_metric, digits=8), " ", round(new_metric, digits=8), " ", round(old_metric, digits=8))
            end

            # Fill in debug data, either with the most recent iteration or every.
            if !ignore_xkuk_iters
                sg.xks[current_idx, num_iters+2, :, :] = xs_k
                for ii in 1:num_players
                    sg.uks[ii][current_idx, num_iters+2, :, :] = us_k[ii]
                    if should_store_feedback
                        sg.Kks[ii][current_idx, num_iters+1, :, :, :] = Ks[ii]
                        sg.kks[ii][current_idx, num_iters+1, :, :] = ks[ii]
                    end
                end
            else
                sg.xks[current_idx, :, :] = xs_k
                for ii in 1:num_players
                    sg.uks[ii][current_idx, :, :] = us_k[ii]
                    if should_store_feedback
                        sg.Kks[ii][current_idx, num_iters+1, :, :, :] = Ks[ii]
                        sg.kks[ii][current_idx, num_iters+1, :, :] = ks[ii]
                    end
                end
            end

            xs_km1 = xs_k
            us_km1 = us_k

            num_iters += 1
        end
        sg.timings[current_idx, num_iters] = time_val
    end

    # Update debug data.
    sg.num_iterations[current_idx] = num_iters
    sg.evaluated_costs[current_idx, :, :] = evaluated_costs
    sg.convergence_metrics[current_idx, :, :] = convergence_metrics

    # Return the results from the current run of the algorithm for convenience.
    return xs_km1, us_km1, is_converged, num_iters, convergence_metrics, evaluated_costs
end

export stackelberg_ilqgames
