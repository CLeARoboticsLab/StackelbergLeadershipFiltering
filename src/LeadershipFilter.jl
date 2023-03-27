using Distributions
using Random

# TODO(hamzah) - horizon 1 game should be 50-50 at all times

# TODO(hamzah) - implement dynamics with memory that stacks states
# TODO(hamzah) - implement with unicycle dynamics, test on SILQGames too if needed
    # dyn = UnicycleDynamics(num_players)
# TODO(hamzah) - add process noise if needed


# Produce windowed measurements
# function process_measurements(Zs, R, num_hist, num_states)
#     out = zeros(num_states, num_hist)
#     dim = size(Zs, 2)
#     iter_count = min(dim, num_hist)

#     # Put the measurements in the proper order with later ones last.
#     out[:, num_hist-iter_count+1:num_hist] = Zs
#     return vec(out)
# end


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

    # Initialize variables.
    X = x₁
    big_P = P₁

    @assert Ts == 1

    # # If multiple points of history included, stack the vector as needed.
    # if dyn_w_hist.num_hist > 1
    #     X = vcat(zeros((dyn_w_hist.num_hist - 1) * xdim(dyn)),
    #              X)
    #     P = 3 # TODO(hamzah) - fix this ...
    # end

    # initialize outputs
    x̂s = zeros(xdim(dyn), num_times)
    P̂s = zeros(xdim(dyn), xdim(dyn), num_times)
    lead_probs = zeros(num_times)

    # compute number of sim_times

    # Define the variables that capture outputs of the SILQGames call (i.e. eval_costs). 

    # The measurements and state are the same size.
    meas_size = xdim(dyn_w_hist)
    pf = initialize_particle_filter(X, big_P, s_init_distrib, t0, Ns, num_times, meas_size, rng)

    for tt in 1:num_times

        # Get inputs.
        # TODO(hamzah) - update this to work for multiple timesteps
        # stack_times = times[tt-Ts+1:tt]
        ttm1 = (tt == 1) ? tt : tt-1
        stack_times = [times[ttm1], times[tt]]
        us_at_tt = [us[ii][:, tt] for ii in 1:num_players]

        # Initial control trajectory estimate for Stackelberg solutions
        us_1_from_tt = [us[ii][:, tt-Ts+1:tt] for ii in 1:num_players]

        # Define Stackelberg measurement models that stack the state results.
        # TODO(hamzah) - get the other things out too
        h₁(X) = vec(stackelberg_ilqgames(1, Ts, stack_times[1], stack_times, dyn, costs, get_state(dyn_w_hist, X, Ts), us_1_from_tt; threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)[1])
        h₂(X) = vec(stackelberg_ilqgames(2, Ts, stack_times[1], stack_times, dyn, costs, get_state(dyn_w_hist, X, Ts), us_1_from_tt; threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)[1])

        # TODO(hamzah) - update for multiple historical states
        # Assumes Ts=1
        # start_idx = max(1, tt-Ts+1)
        # Zₜ = process_measurements(zs[start_idx:tt])
        Zₜ = zs[:, tt]
        Rₜ = R

        f_dynamics(time_range, X, us, rng) = propagate_dynamics(dyn_w_hist, time_range, X, us)
        ttm1 = (tt == 1) ? 1 : tt-1

        time_range = (times[ttm1], times[tt])
        step_pf(pf, time_range, [f_dynamics, f_dynamics], [h₁, h₂], discrete_state_transition, us_at_tt, Zₜ, Rₜ)

        # Update the variables.
        X = pf.x̂[:, 2]
        big_P = pf.P[:, :, 2]

        # Store relevant information.
        # TODO(hamzah) - update for multiple states
        x̂s[:, tt] = X
        P̂s[:, :, tt] = big_P
        lead_probs[tt] = pf.ŝ_probs[tt]
    end
    
    # outputs: (1) state estimates, uncertainty estimates, leadership_probabilities over time, debug data
    return x̂s, P̂s, lead_probs
end

export leadership_filter
