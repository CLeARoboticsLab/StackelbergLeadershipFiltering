# This file sets up a simple scenario and runs it on a Stackelberg game (with or without noise).
using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Random

include("leadfilt_LQ_parameters.jl")

# Solve an LQ Stackelberg game based on the shepherd and sheep example.
Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)


# CONFIG: 

# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 4. * I
zs = zeros(xdim(dyn), T)
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(xs[:, tt], R))
end
Ts = 2

p = 0.4
p_init = 0.6


# TODO(hamzah) - vectorize this better
function generate_discrete_state_transition(p₁₁, p₂₂)

    distribs = [Bernoulli(p₁₁), Bernoulli(p₂₂)]

    # state transition matrix of state
    P = [ p₁₁  1-p₂₂;
         1-p₁₁  p₂₂]

    # The discrete state transition stays in state i with probability pᵢ.
    function discrete_state_transition(time_range, s_prev, s_probs, 𝒳_prev, s_actions, rng)

        @assert length(s_prev) == 1
        s_prev = s_prev[1]
        sample = rand(rng, distribs[s_prev], 1)

        # use markov chain to adjust over time
        other_state = (s_prev == 1) ? 2 : 1
        s_new = (isone(sample[1])) ? s_prev : other_state

        return [s_new]
    end
    return discrete_state_transition, P
end

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p, 1-p)
s_init_distrib = Bernoulli(p_init)


x̂s, P̂s, probs = leadership_filter(dyn, costs,
                           t0,
                           times,
                           Ts, # horizon over which the stackelberg game should be played,
                           x₁, # initial state at the beginning of simulation
                           P₁, # initial covariance at the beginning of simulation
                           us, # the control inputs that the actor takes
                           zs, # the measurements
                           R,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=1.,
                           rng,
                           max_iters=1000,
                           step_size=0.01,
                           Ns=10,
                           verbose=false)

plot(times, probs, xlabel="t (s)", ylabel="prob. leadership")
