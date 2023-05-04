# This file sets up a simple scenario and runs it on a Stackelberg game (with or without noise).
using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Random
using Plots

gr()

include("leadfilt_LQ_parameters.jl")

# CONFIG: 
# We define an uncertainty for the measurements R arbitrarily - easy for now.
# 
rng = MersenneTwister(0)

R = zeros(xdim(dyn), xdim(dyn)) + 0.001 * I
zs = zeros(xdim(dyn), T)
Ts = 20
num_games = 1
num_particles = 50

p = 0.95
p_init = 0.7


threshold = 0.1
max_iters = 25
step_size = 0.01


# Solve an LQ Stackelberg game based on the shepherd and sheep example.
Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)
xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)

# Augment the remaining states so we have T+Ts-1 of them.
xs = hcat(xs, zeros(xdim(dyn), Ts-1))
us = [hcat(us[ii], zeros(udim(dyn, ii), Ts-1)) for ii in 1:num_players]

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

process_noise_distribution = MvNormal(zeros(xdim(dyn)), Q)


x̂s, P̂s, probs, pf, sg_objs = leadership_filter(dyn, costs, t0, times,
                           T,         # simulation horizon
                           Ts,        # horizon over which the stackelberg game should be played,
                           num_games, # number of stackelberg games played for measurement
                           x₁,        # initial state at the beginning of simulation
                           P₁,        # initial covariance at the beginning of simulation
                           us,        # the control inputs that the actor takes
                           zs,        # the measurements
                           R,
                           process_noise_distribution,
                           s_init_distrib,
                           discrete_state_transition;
                           threshold=threshold,
                           rng,
                           max_iters=max_iters,
                           step_size=step_size,
                           Ns=num_particles,
                           verbose=false)

true_xs = xs

using Dates
using Plots
using Printf
using ProgressBars
gr()

# N = Int(sg_obj.num_iterations[1]+1)
iter = ProgressBar(2:T)
anim = @animate for t in iter
    p = @layout [a b; c d; e f]

    title = string("LF (", t, ") on Stack(", leader_idx, "), Ts=", Ts,"\n, Ns=", num_particles)
    p1 = plot(legend=:outertopright, ylabel="y (m)", xlabel="x (m)", title=title)
    plot!(p1, true_xs[1, 1:T], true_xs[3, 1:T], label="P1 pos", ylimit=(-2.0, 2.0), xlimit=(-2.0, 2.0))
    plot!(p1, true_xs[5, 1:T], true_xs[7, 1:T], label="P2 pos")

    plot!(p1, zs[1, 1:T], zs[3, 1:T], label="P1 meas pos", color=:blue, linewidth=0.15)
    plot!(p1, zs[5, 1:T], zs[7, 1:T], label="P2 meas pos", color=:red, linewidth=0.15)

    p1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
    p1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

    # plot 2
    p2 = plot(times[1:T], x̂s[1,1:T], label="P1 px", legend=:outertopright)
    plot!(p2, times[1:T], x̂s[3,1:T], label="P1 py")
    plot!(p2, times[1:T], x̂s[5,1:T], label="P2 px", legend=:outertopright)
    plot!(p2, times[1:T], x̂s[7,1:T], label="P2 py")
    plot!(p2, [times[t], times[t]], [-1, 2], label="", color=:black)

    # plot 3
    p3 = plot(times[1:T], x̂s[2,1:T], label="P1 vx", legend=:outertopright)
    plot!(p3, times[1:T], x̂s[4,1:T], label="P1 vy")
    plot!(p3, times[1:T], x̂s[6,1:T], label="P2 vx")
    plot!(p3, times[1:T], x̂s[8,1:T], label="P2 vy")
    plot!(p3, [times[t], times[t]], [-1, 1], label="", color=:black)

    # Add particles
    num_iters = [0, 0]
    for n in 1:num_particles

        num_iter = sg_objs[t].num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])

        x1_idx = 1
        y1_idx = 3
        x2_idx = 5
        y2_idx = 7

        xks = sg_objs[t].xks[n, num_iter, :, :]

        # TODO(hamzah) - change color based on which agent is leader
        scatter!(p1, xks[x1_idx, :], xks[y1_idx, :], color=:black, markersize=0.5, label="")

        color = (sg_objs[t].leader_idxs[n] == 1) ? :blue : :red
        scatter!(p1, [xks[x1_idx, 2]], [xks[y1_idx, 2]], color=color, markersize=3., label="")

        scatter!(p1, xks[x2_idx, :], xks[y2_idx, :], color=:black, markersize=0.5, label="")
        scatter!(p1, [xks[x2_idx, 2]], [xks[y2_idx, 2]], color=color, markersize=3., label="")


        # plot 2 particles
        # scatter!(p2, t:t+Ts-1, xks[x1_idx, :], color=:black, markersize=0.15, label="")

        # plot 3 particles


    end



    # plot 4
    p4 = plot(times[1:T], us[1][1, 1:T], label="P1 ax", legend=:outertopright)
    plot!(times[1:T], us[1][2, 1:T], label="P1 ay")
    plot!(times[1:T], us[2][1, 1:T], label="P2 ax", legend=:outertopright)
    plot!(times[1:T], us[2][2, 1:T], label="P2 ay")
    plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)

    # probability plot - plot 5
    p5 = plot(times[1:T], probs[1:T], xlabel="t (s)", ylabel="prob. leadership", ylimit=(-0.1, 1.1), label="", legend=:outertopright)
    plot!(p5, times[1:T], (leader_idx%2) * ones(T), label="truth")
    plot!(p5, [times[t], times[t]], [0, 1], label="", color=:black)

    plot(p1, p2, p3, p4, p5, layout = p, size=(1080, 860))
end

# Speeds up call to gif - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = string("lq_leadfilt_",string(Dates.now()),".gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
