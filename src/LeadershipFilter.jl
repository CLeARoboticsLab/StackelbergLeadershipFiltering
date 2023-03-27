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
                           max_iters=1000,
                           step_size=0.01,
                           seed=1,
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

    for tt in 1:num_times

        # Get inputs.
        # TODO(hamzah) - update this to work for multiple timesteps
        # stack_times = times[tt-Ts+1:tt]
        ttm1 = (tt == 1) ? tt : tt-1
        stack_times = [times[ttm1], times[tt]]
        us_by_time = Dict(tau => [us[ii][:, tau] for ii in 1:num_players] for tau in 1:Ts+1)

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
        Zₜ = zs'
        Rₜ = R

        f_dynamics(time_range, X, us, rng) = propagate_dynamics(dyn_w_hist, time_range, X, us)
        ttm1 = (tt == 1) ? 1 : tt-1

        x̂, P, z̄, P̄_zz, ϵ_bar, ϵ_hat, N̂s, s_particles, ŝ_probs, particles = two_state_PF(X,
             P₁, # TODO(hamzah): Update this to run properly.
             us_by_time,
             s_init_distrib,
             [times[ttm1], times[tt]],
             t0,
             Zₜ,
             Rₜ,
             discrete_state_transition,
             [f_dynamics, f_dynamics],
             [h₁, h₂];
             seed=seed,
             Ns=Ns)

        # Update the variables.
        X = x̂[:, 2]
        big_P = P[:, :, 2]

        # Store relevant information.
        # TODO(hamzah) - update for multiple states
        x̂s[:, tt] = X
        P̂s[:, :, tt] = big_P
        # println(size(ŝ_probs))
        lead_probs[tt] = ŝ_probs[2]
    end
    
    # outputs: (1) state estimates, uncertainty estimates, leadership_probabilities over time, debug data
    return x̂s, P̂s, lead_probs
end

export leadership_filter
