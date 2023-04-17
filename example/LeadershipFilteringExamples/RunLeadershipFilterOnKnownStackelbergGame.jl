# This file sets up a simple scenario and runs it on a Stackelberg game (with or without noise).
using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Random
using Plots

include("leadfilt_LQ_parameters.jl")

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.001 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 100

p = 0.95
p_init = 0.3


threshold = 0.1
max_iters = 25
step_size = 0.01


# Solve an LQ Stackelberg game based on the shepherd and sheep example.
Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T+Ts-1, leader_idx)
xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)


# Fill in z as noisy state measurements.
for tt in 1:T
    zs[:, tt] = rand(rng, MvNormal(xs[:, tt], R))
end


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

discrete_state_transition, state_trans_P = generate_discrete_state_transition(p, p)
s_init_distrib = Bernoulli(p_init)


x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t0, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us,        # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=false)

true_xs = xs

# Plot positions, other two states, controls, and convergence.
# q = @layout [a b; c d; e f]
p = @layout [a; b]

q1 = plot(legend=:outertopright, ylabel="y (m)", xlabel="x (m)", title=string("LF on Noisy LQ Stack., leader=", leader_idx,
                                                                              ", Ts=", Ts,
                                                                              ", Ns=", num_particles))
plot!(q1, true_xs[1, :], true_xs[3, :], label="P1 pos", ylimit=(-2.0, 2.0), xlimit=(-2.0, 2.0))
plot!(q1, true_xs[5, :], true_xs[7, :], label="P2 pos")

plot!(q1, zs[1, :], zs[3, :], label="P1 meas pos", color=:blue, linewidth=0.15)
plot!(q1, zs[5, :], zs[7, :], label="P2 meas pos", color=:red, linewidth=0.15)

# q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Leadership Filter")
q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

# Plot the stackelberg trajectories.
PLOT_WHOLE_TRAJECTORY = false # if true, plot the whole trajectory at each cycle. If not, plot the relevant Stack. state.

# This plots the line formed by the measurement model outputs (for when we have 1 game played).
# However, it switches between whatever the leadership state of the particle is.
# println(size(pf.z_models), " ", pf.z_models)
for t in 2:T
    # println(pf.z_models[1,:,t], pf.z_models[3,:,t])
    num_iters = [0, 0]
    for n in 1:num_particles

        num_iter_1 = sg_objs[t].num_iterations[n]
        num_iter_2 = sg_objs[t].num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])

        x1_idx = 1
        y1_idx = 3
        x2_idx = 5
        y2_idx = 7

        # TODO(hamzah) - change color based on which agent is leader
        scatter!(q1, sg_objs[t].xks[n, num_iter_1, x1_idx, :], sg_objs[t].xks[n, num_iter_1, y1_idx, :], color=:black, markersize=0.15, label="")
        scatter!(q1, sg_objs[t].xks[n, num_iter_2, x2_idx, :], sg_objs[t].xks[n, num_iter_2, y2_idx, :], color=:black, markersize=0.15, label="")
    end

    # Plot the h(X) for each of the particles.
    # scatter!(q1, pf.z_models[1,:,t], pf.z_models[3,:,t], color=:black, markersize=0.15, label="")
    # scatter!(q1, pf.z_models[5,:,t], pf.z_models[7,:,t], color=:black, markersize=0.15, label="")
end
    # t, particles[2, :, :][:, :]', color=:black, markersize=0.15, label="", yrange=(-plot_mult*5.0,plot_mult*5.0), legend=:outertopright, ylabel="position (m)")

# q2 = plot(times, xs_k[1,:], label="P1 px", legend=:outertopright)
# plot!(times, xs_k[3,:], label="P1 py")
# plot!(times, xs_k[5,:], label="P2 px", legend=:outertopright)
# plot!(times, xs_k[7,:], label="P2 py")

# q3 = plot(times, xs_k[2,:], label="vel1 x", legend=:outertopright)
# plot!(times, xs_k[4,:], label="vel1 y")
# plot!(times, xs_k[6,:], label="vel2 x")
# plot!(times, xs_k[8,:], label="vel2 y")

# q4 = plot(times, us_k[1][1, :], label="P1 accel x", legend=:outertopright)
# plot!(times, us_k[1][2, :], label="P1 accel y")
# plot!(times, us_k[2][1, :], label="P2 accel x", legend=:outertopright)
# plot!(times, us_k[2][2, :], label="P2 accel y")

# # Plot convergence.
# conv_x = cumsum(ones(num_iters)) .- 1
# q5 = plot(conv_x, conv_metrics[1, 1:num_iters], title="convergence (||k||^2) by player", label="p1", yaxis=:log)
# plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)

# q6 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log)
# plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)




# TODO(hamzah) - Next step, make these plots better so I can debug what's going on.
q5 = plot(times[1:T], probs, xlabel="t (s)", ylabel="prob. leadership", ylimit=(-0.1, 1.1))

# plot(q1, q2, q3, q4, q5, q6, layout = q)
plot(q1, q5, layout = p)


