###############################################
### Measurement model 1: Smoothing approach ###
###############################################

# Produce windowed measurements
# function process_measurements_opt1(tt, zs, R, num_stack_hist; large_var=LARGE_VARIANCE)
#     num_z = size(zs, 1)
#     meas_hist_count = size(zs, 2)
#     out = zeros(num_z, num_hist)
    
#     # Compute indices for the case where we don't have enough data to fill a history of the desired length.
#     iter_count = min(meas_hist_count, num_hist)
#     start_idx = num_hist-iter_count+1

#     # Put the measurements in the proper order with later ones last.
#     out[:, start_idx:num_hist] = zs

#     # Compute measurement matrix, with placeholder 0 states having high variance.
#     Rs = [sparse(R) for I in 1:num_hist]
#     if meas_hist_count < num_hist
#         m = sparse(large_var * I, num_z, num_z)
#         for tau in 1:num_hist-iter_count
#             Rs[tau] = m
#         end
#     end
#     big_R = Matrix(blockdiag(Rs...))

#     return vec(out), big_R
# end


# function make_stackelberg_meas_model(tt::Int, leader_idx::Int, num_games::Int, num_runs_per_game::Int,
#                                      Ts::Int, t0, times, dyn_w_hist::DynamicsWithHistory, costs, us,
#                                      threshold, max_iters, step_size, verbose)
#     dyn = get_underlying_dynamics(dyn_w_hist)
#     @assert num_games == 1

#     # TODO(hamzah) - update this to work for multiple timesteps

#     # Select actual controls as initial control trajectory estimate for Stackelberg solutions
#     us_1_from_tt = [zeros(udim(dyn, ii), Ts) for ii in 1:num_players]

#     # If we have multiple time steps to draw from, then let's do more.
#     ideal_start_idx = (tt==1) ? tt - num_games + 1 : tt - num_games
#     has_full_history = ideal_start_idx > 0
#     start_idx = (has_full_history) ? ideal_start_idx : 1

#     # number of indices to leave with 0 controls
#     u_times = zeros(Ts)
#     empty_end_idx = (start_idx == 1) ? Ts - tt : 0
#     println("size of u_times: ", size(u_times[empty_end_idx+1:Ts]))
#     println("size of times: ", size(times[start_idx:tt]), " ", times[start_idx:tt])
#     println(empty_end_idx + 1, " ", Ts)
#     println(start_idx, " ", tt)
#     u_times[empty_end_idx+1:Ts] = copy(times[start_idx:tt])
#     for ii in 1:num_players
#         us_1_from_tt[ii][:, empty_end_idx+1:Ts] = us[ii][:, start_idx:tt] 
#     end

#     # Initialize an SILQ Games Object for this set of runs.
#     sg_obj = initialize_silq_games_object(num_runs_per_game, leader_idx, Ts, dyn, costs;
#                                           threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)

#     # For now assumes 1 game played
#     @assert num_games == 1
#     function h(X)
#         # TODO(hamzah) - revisit this if it becomes a problem
#         # If we are on the first time step, then we can't get a useful history to play a Stackelberg game.
#         # Return the current state.
#         if tt == 1
#             return get_current_state(dyn_w_hist, X)
#         end

#         # Get the state at the previous time.
#         prev_state = get_state(dyn_w_hist, X, 2)

#         # Get the time for the previous state.
#         init_time = times[tt-1]
#         stack_times = [times[ii] for ii in tt-1:tt+Ts-1]

#         xs, us = stackelberg_ilqgames(sg_obj, leader_idx, init_time, stack_times, prev_state, us_1_from_tt)

#         # Process the stackelberg trajectory to get the desired output and vectorize.
#         return xs[:, min(Ts, 2)]
#     end
#     return h, sg_obj
# end


# function make_stackelberg_meas_model(tt::Int, leader_idx::Int, num_games_desired::Int, num_runs_per_game::Int,
#                                      Ts::Int, t0, times, dyn_w_hist::DynamicsWithHistory, costs, us,
#                                      threshold, max_iters, step_size, verbose)
#     # Extract the history-less dynamics.
#     dyn = get_underlying_dynamics(dyn_w_hist)

#     # Extract the number of games we can play and the start/end indices.
#     num_games_playable, indices_list = get_indices_for_playable_games(tt, num_games_desired, Ts)

#     # TODO(hamzah) - update this to work for multiple timesteps
#     # Extract the start and end indices.
#     @assert num_games_desired == 1
#     s_idx, e_idx = indices_list[1]

#     # Extract the desired times and controls.
#     stack_times = times[s_idx:e_idx]
#     us_1_from_tt = [us[ii][:, s_idx:e_idx] for ii in 1:num_agents(dyn)]

#     # # Select actual controls as initial control trajectory estimate for Stackelberg solutions
#     # us_1_from_tt = [zeros(udim(dyn, ii), Ts) for ii in 1:num_players]

#     # # If we have multiple time steps to draw from, then let's do more.
#     # ideal_start_idx = (tt==1) ? tt - num_games + 1 : tt - num_games
#     # has_full_history = ideal_start_idx > 0
#     # start_idx = (has_full_history) ? ideal_start_idx : 1

#     # # number of indices to leave with 0 controls
#     # u_times = zeros(Ts)
#     # empty_end_idx = (start_idx == 1) ? Ts - tt : 0
#     # println("size of u_times: ", size(u_times[empty_end_idx+1:Ts]))
#     # println("size of times: ", size(times[start_idx:tt]), " ", times[start_idx:tt])
#     # println(empty_end_idx + 1, " ", Ts)
#     # println(start_idx, " ", tt)
#     # u_times[empty_end_idx+1:Ts] = copy(times[start_idx:tt])
#     # for ii in 1:num_players
#     #     us_1_from_tt[ii][:, empty_end_idx+1:Ts] = us[ii][:, start_idx:tt] 
#     # end

#     # Initialize an SILQ Games Object for this set of runs.
#     sg_obj = initialize_silq_games_object(num_runs_per_game, leader_idx, Ts+1, dyn, costs;
#                                           threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)

#     # TODO(hamzah) - For now assumes 1 game played; fix this
#     @assert num_games_desired == 1
#     function h(X)
#         # TODO(hamzah) - revisit this if it becomes a problem
#         # If we are on the first time step, then we can't get a useful history to play a Stackelberg game.
#         # Return the current state.
#         if tt == 1
#             @assert iszero(num_games_playable)
#             return get_current_state(dyn_w_hist, X)
#         end

#         # Get the state at the previous time tt-1. This will be the initial state in the game.
#         prev_state = get_state(dyn_w_hist, X, 2)

#         # # Get the time for the previous state.
#         # init_time = times[tt-1]
#         # stack_times = [times[ii] for ii in tt-1:tt+Ts-1]

#         xs, us = stackelberg_ilqgames(sg_obj, leader_idx, stack_times[1], stack_times, prev_state, us_1_from_tt)

#         # Process the stackelberg trajectory to get the desired output and vectorize.
#         # return xs[:, min(Ts, 2)]
#         return extract_measurements_from_stack_trajectory(xs, tt-1, tt)
#     end
#     return h, sg_obj
# end