using Distributions
using Random
using SparseArrays

# TODO(hamzah) - horizon 1 game should be random walk
# TODO(hamzah) - implement with unicycle dynamics, test on SILQGames too if needed
    # dyn = UnicycleDynamics(num_players)
# TODO(hamzah) - add process noise if needed

# used for when previous states don't exist
LARGE_VARIANCE = 1e6

###############################################
### Measurement model 2: Filtering approach ###
###############################################

# Identifies the start indices and end indices for the player games.
# Also returns number of games able to be played given the parameters.
# - tt is current time index
# - Ng_desired is the desired number of games to play
# - Ts is the horizon over which each Stackelberg game is player
# Returns number of played games, indices from most recent game start to oldest
function get_indices_for_playable_games(tt::Int, Ng_desired::Int, Ts::Int)
    # Ensure that every played games will reach time tt during the game.
    @assert Ts > Ng_desired

    # If we are at the first step, no games can be played.
    if tt == 1
        num_games_played = 0
        indices = [(1, Ts)]

    # If this condition is satisfied, there isn't enough history to play the desired number of games.
    # We can play up to tt-1 of them.
    elseif tt ≤ Ng_desired
        num_games_played = tt-1
        indices = [(tt - ii, tt + Ts - ii) for ii in 1:tt-1]

    # In the general case, we play the desired number of games.
    else
        num_games_played = Ng_desired
        indices = [(tt - ii, tt + Ts - ii) for ii in 1:Ng_desired]
    end

    return num_games_played, indices
end

# Produce batched measurements at time times[tt].
function process_measurements_opt2(tt::Int, zs, R, num_games_desired::Int, stack_horizon::Int)

    # Make sure we haven't gone past the end of the measurments in time.
    @assert tt <= size(zs, 2)

    # Extract the number of games we can play.
    num_games_playable, _ = get_indices_for_playable_games(tt, num_games_desired, stack_horizon)

    num_meas = num_games_playable

    # If we can't play any games, then we return one measurement at the current time.
    if iszero(num_games_playable)
        # this situation should only happen at tt = 1
        @assert tt == 1
        num_meas = 1
    end

    zₜ = zs[:, tt]
    Rₜ = R

    # Produce the desired augmented measurement matrix.
    Zₜ = repeat(zₜ, num_meas)

    # Construct a large R matrix that contains the (identical) uncertanties for each of these measurements.
    Rs = [sparse(R) for I in 1:num_meas]
    big_R = Matrix(blockdiag(Rs...))

    return Zₜ, big_R
end


# Measurement model option 2 - Extract the waypoint at time tt in the simulation from the state trajectory xs, which is
#                              indexed from 1 to tt.
function extract_measurements_from_stack_trajectory(xs, start_time_idx::Int, tt::Int)
    translated_time_idx = tt - start_time_idx + 1
    return xs[:, translated_time_idx]
end


function make_stackelberg_meas_model(tt::Int, sg_obj::SILQGamesObject, leader_idx::Int, num_games_desired::Int,
                                     Ts::Int, t0, times, dyn_w_hist::DynamicsWithHistory, costs, us)

    # Extract the history-less dynamics.
    dyn = get_underlying_dynamics(dyn_w_hist)

    # Extract the number of games we can play and the start/end indices.
    num_games_playable, indices_list = get_indices_for_playable_games(tt, num_games_desired, Ts)

    # TODO(hamzah) - update this to work for multiple timesteps
    # Extract the start and end indices.
    @assert num_games_desired == 1
    s_idx, e_idx = indices_list[1]

    # Extract the desired times and controls.
    stack_times = times[s_idx:e_idx]

    # DEBUGGING: This line provides the future control trajectory to the filter - should not generally be enabled.
    # us_1_from_tt = [us[ii][:, s_idx:e_idx] for ii in 1:num_agents(dyn)]

    # Set the initial control estimate to be the initial control repeated into the future for Ts time steps.
    ctrl_len = e_idx - s_idx + 1
    us_prev = [us[ii][:, s_idx] for ii in 1:num_agents(dyn)]
    us_1_from_tt = [repeat(us_prev[ii], 1, ctrl_len) for ii in 1:num_agents(dyn)]

    # TODO(hamzah) - For now assumes 1 game played; fix this
    @assert num_games_desired == 1
    function h(X)
        # TODO(hamzah) - revisit this if it becomes a problem
        # If we are on the first time step, then we can't get a useful history to play a Stackelberg game.
        # Return the current state.
        if tt == 1
            @assert iszero(num_games_playable)
            return get_current_state(dyn_w_hist, X)
        end

        # Get the state at the previous time tt-1. This will be the initial state in the game.
        prev_state = get_state(dyn_w_hist, X, 2)

        # Play a stackelberg games starting at this previous state using the times/controls we extracted.
        xs, us = stackelberg_ilqgames(sg_obj, leader_idx, stack_times[1], stack_times, prev_state, us_1_from_tt)

        # Process the stackelberg trajectory to get the desired output and vectorize.
        return extract_measurements_from_stack_trajectory(xs, tt-1, tt)
    end
    return h
end


# This implementations assumes no history.
function leadership_filter(dyn::Dynamics,
                           costs,
                           t0,
                           times,
                           T,  # simulation horizon
                           Ts, # horizon-length over which the stackelberg game should be played
                           num_games, # number of stackelberg games which should be batched in measurement model
                           x₁, # initial state at the beginning of simulation
                           P₁, # initial covariance at the beginning of simulation
                           us, # the control inputs that the actor takes
                           zs, # the measurements
                           R,  # measurement noise
                           process_noise_distribution,
                           s_init_distrib::Distribution{Univariate, Discrete},
                           discrete_state_transition::Function;
                           threshold,
                           rng,
                           max_iters=1000,
                           step_size=0.01,
                           Ns=1000,
                           verbose=false)
    num_times = T
    num_players = num_agents(dyn)
    dyn_w_hist = DynamicsWithHistory(dyn, num_games+1)

    num_states = xdim(dyn)
    num_states_w_hist = xdim(dyn_w_hist)

    # Store the results of the SILQGames runs for the leader and follower at each time in the simulation.
    # This is important because it exposes the debug data.
    sg_objs = Array{SILQGamesObject}(undef, num_times)

    # Initialize variables for case num_hist == 1.
    X = x₁
    big_P = P₁

    # If multiple points of history included, stack the vector as needed.
    if dyn_w_hist.num_hist > 1
        # size of the historical states
        hist_size = num_states_w_hist - num_states

        X = vcat(zeros(hist_size), X)
        big_P = zeros(num_states_w_hist, num_states_w_hist)
        big_P[hist_size+1:num_states_w_hist, hist_size+1:num_states_w_hist] = P₁

        # Adjust the covariances to make the matrix nonsingular.
        # TODO(hamzah) - how to handle states that don't exist, high uncertainty probably best
        big_P = big_P + 1e-32 * I
    end

    # initialize outputs
    x̂s = zeros(num_states, num_times)
    P̂s = zeros(num_states, num_states, num_times)
    lead_probs = zeros(num_times)

    # compute number of sim_times
    # TODO(hamzah) - extrapolate times to the future as needed for Ts to be used
    #                this is currently done when running it

    # The measurements and state are the same size.
    meas_size = xdim(dyn_w_hist.dyn)
    pf = initialize_particle_filter(X, big_P, s_init_distrib, t0, Ns, num_times, meas_size, rng)

    for tt in 1:num_times
        println("leadership_filter tt ", tt)

        # Get inputs at time tt.
        us_at_tt = [us[ii][:, tt] for ii in 1:num_players]

        # Define Stackelberg measurement models that stack the state results.
        # TODO(hamzah) - get the other things out too
        num_runs_per_game = Ns + 2 # two extra runs for metadata

        # Initialize an SILQ Games Object for this set of runs.
        sg_objs[tt] = initialize_silq_games_object(num_runs_per_game, Ts+1, dyn, costs;
                                              threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)

        # Create the measurement models.
        h₁ = make_stackelberg_meas_model(tt, sg_objs[tt], 1, num_games,
                                         Ts, t0, times, dyn_w_hist, costs, us)
        h₂ = make_stackelberg_meas_model(tt, sg_objs[tt], 2, num_games,
                                         Ts, t0, times, dyn_w_hist, costs, us)

        # TODO(hamzah) - update for multiple historical states
        Zₜ, Rₜ = process_measurements_opt2(tt, zs, R, num_games, Ts)

        f_dynamics(time_range, X, us, rng) = propagate_dynamics(dyn_w_hist, time_range, X, us) + vcat(zeros(xdim(dyn_w_hist) - xdim(dyn)), rand(rng, process_noise_distribution))

        ttm1 = (tt == 1) ? 1 : tt-1
        time_range = (times[ttm1], times[tt])
        step_pf(pf, time_range, [f_dynamics, f_dynamics], [h₁, h₂], discrete_state_transition, us_at_tt, Zₜ, Rₜ)

        # Update the variables.
        X = pf.x̂[:, tt]
        big_P = pf.P[:, :, tt]

        # Store relevant information.
        # TODO(hamzah) - update for multiple states
        x̂s[:, tt] = get_current_state(dyn_w_hist, X)
        s_idx = (dyn_w_hist.num_hist-1) * num_states+1
        e_idx = xdim(dyn_w_hist)
        P̂s[:, :, tt] = big_P[s_idx:e_idx, s_idx:e_idx]

        lead_probs[tt] = pf.ŝ_probs[tt]
    end
    
    # outputs: (1) state estimates, uncertainty estimates, leadership_probabilities over time, debug data
    return x̂s, P̂s, lead_probs, pf, sg_objs
end

export leadership_filter
