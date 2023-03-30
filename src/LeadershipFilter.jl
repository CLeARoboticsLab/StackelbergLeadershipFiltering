using Distributions
using Random
using SparseArrays

# TODO(hamzah) - horizon 1 game should be 50-50 at all times

# TODO(hamzah) - implement dynamics with memory that stacks states
# TODO(hamzah) - implement with unicycle dynamics, test on SILQGames too if needed
    # dyn = UnicycleDynamics(num_players)
# TODO(hamzah) - add process noise if needed

# used for when previous states don't exist
LARGE_VARIANCE = 1e6


# Produce windowed measurements
function process_measurements(zs, R, num_hist; large_var=LARGE_VARIANCE)
    num_z = size(zs, 1)
    meas_hist_count = size(zs, 2)
    out = zeros(num_z, num_hist)
    
    # Compute indices for the case where we don't have enough data to fill a history of the desired length.
    iter_count = min(meas_hist_count, num_hist)
    start_idx = num_hist-iter_count+1

    # Put the measurements in the proper order with later ones last.
    out[:, start_idx:num_hist] = zs

    # Compute measurement matrix, with placeholder 0 states having high variance.
    Rs = [sparse(R) for I in 1:num_hist]
    if meas_hist_count < num_hist
        m = sparse(large_var * I, num_z, num_z)
        for tau in 1:num_hist-iter_count
            Rs[tau] = m
        end
    end
    big_R = Matrix(blockdiag(Rs...))

    return vec(out), big_R
end


# This implementations assumes no history.
function leadership_filter(dyn::Dynamics,
                           costs,
                           t0,
                           times,
                           Ts, # horizon over which the stackelberg game should be played,
                           x₁, # initial state at the beginning of simulation
                           P₁, # initial covariance at the beginning of simulation
                           us, # the control inputs that the actor takes
                           zs, # the measurements
                           R,
                           s_init_distrib::Distribution{Univariate, Discrete},
                           discrete_state_transition::Function;
                           threshold,
                           rng,
                           max_iters=1000,
                           step_size=0.01,
                           Ns=1000,
                           verbose=false)
    num_times = length(times)
    num_players = num_agents(dyn)
    dyn_w_hist = DynamicsWithHistory(dyn, Ts)

    num_states = xdim(dyn)
    num_states_w_hist = xdim(dyn_w_hist)

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

    # Define the variables that capture outputs of the SILQGames call (i.e. eval_costs). 


    # The measurements and state are the same size.
    meas_size = xdim(dyn_w_hist)
    pf = initialize_particle_filter(X, big_P, s_init_distrib, t0, Ns, num_times, meas_size, rng)

    for tt in 1:num_times
        println("leadership_filter tt ", tt)

        # Get inputs.
        # TODO(hamzah) - update this to work for multiple timesteps
        start_idx = max(tt-Ts+1, 1)

        # stack_times = times[tt-Ts+1:tt]
        ttm1 = (tt == 1) ? tt : tt-1
        stack_times = t0 * ones(Ts)

        if dyn_w_hist.num_hist == 1
            stack_times = [times[ttm1], times[tt]]
        end

        us_at_tt = [us[ii][:, tt] for ii in 1:num_players]

        # Initial control trajectory estimate for Stackelberg solutions
        us_1_from_tt = [zeros(udim(dyn, ii), Ts) for ii in 1:num_players]

        # number of indices to leave with 0 controls
        empty_end_idx = (start_idx == 1) ? Ts - tt : 0
        times[empty_end_idx+1:Ts] = times[start_idx:tt]
        for ii in 1:num_players
            us_1_from_tt[ii][:, empty_end_idx+1:Ts] = us[ii][:, start_idx:tt] 
        end

        # Define Stackelberg measurement models that stack the state results.
        # TODO(hamzah) - get the other things out too
        h₁(X) = vec(stackelberg_ilqgames(1, Ts, stack_times[1], stack_times, dyn, costs, get_state(dyn_w_hist, X, Ts), us_1_from_tt; threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)[1])
        h₂(X) = vec(stackelberg_ilqgames(2, Ts, stack_times[1], stack_times, dyn, costs, get_state(dyn_w_hist, X, Ts), us_1_from_tt; threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)[1])

        # TODO(hamzah) - update for multiple historical states
        # Assumes Ts=1
        # start_idx = max(1, tt-Ts+1)
        Zₜ, Rₜ = process_measurements(zs[:, start_idx:tt], R, dyn_w_hist.num_hist)
        # Zₜ = zs[:, tt]
        # Rₜ = R

        f_dynamics(time_range, X, us, rng) = propagate_dynamics(dyn_w_hist, time_range, X, us)
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
        println(tt, " x - ", x̂s[:, tt])
        println(tt, " P - ", norm(P̂s[:, :, tt]))

        lead_probs[tt] = pf.ŝ_probs[tt]
        println(tt, " prob - ", lead_probs[tt])
    end
    
    # outputs: (1) state estimates, uncertainty estimates, leadership_probabilities over time, debug data
    return x̂s, P̂s, lead_probs
end

export leadership_filter
