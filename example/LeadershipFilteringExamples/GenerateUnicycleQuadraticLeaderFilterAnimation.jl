# This file sets up a simple scenario and runs the leadership filter on a Stackelberg game with unicycle dynamics and quadratic cost.
using StackelbergControlHypothesesFiltering

using Distributions
using LinearAlgebra
using Random
using Plots

gr()

include("leadfilt_unicycle_quadratic_parameters.jl")


discrete_state_transition, state_trans_P = generate_discrete_state_transition(p_transition, p_transition)
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

    plot_title = string("LF (", t, "/", T, ") on Stack(L=P", leader_idx, "), Ts=", Ts, ", Ns=", num_particles, ", p(transition)=", p_transition, ", #games: ", num_games)
    println(plot_title)
    title1="x-y plot of agent positions over time"
    p1 = plot(title=title1, legend=:outertopright, ylabel="y (m)", xlabel="x (m)", ylimit=(-2.0, 2.0), xlimit=(-2.0, 2.0))
    plot!(p1, true_xs[1, 1:T], true_xs[2, 1:T], label="P1 true pos")
    plot!(p1, true_xs[5, 1:T], true_xs[6, 1:T], label="P2 true pos")

    plot!(p1, zs[1, 1:T], zs[2, 1:T], label="P1 meas pos", color=:blue, linewidth=0.15)
    plot!(p1, zs[5, 1:T], zs[6, 1:T], label="P2 meas pos", color=:red, linewidth=0.15)

    p1 = scatter!([x₁[1]], [x₁[2]], color="blue", label="start P1")
    p1 = scatter!([x₁[5]], [x₁[6]], color="red", label="start P2")

    # plot 2
    title2 = "LF estimated states (x̂) over time"
    p2 = plot(legend=:outertopright, xlabel="t (s)", ylabel="pos (m)", title=title2)
    plot!(p2, times[1:T], x̂s[1,1:T], label="P1 px")
    plot!(p2, times[1:T], x̂s[2,1:T], label="P1 py")
    plot!(p2, times[1:T], x̂s[5,1:T], label="P2 px")
    plot!(p2, times[1:T], x̂s[6,1:T], label="P2 py")
    plot!(p2, [times[t], times[t]], [-1, 2], label="", color=:black)

    # plot 3
    title3 = "LF estimated heading (x̂) over time"
    p3 = plot(legend=:outertopright, xlabel="t (s)", ylabel="θ (rad)", title=title3)
    plot!(p3, times[1:T], x̂s[3,1:T], label="P1 θ")
    plot!(p3, times[1:T], x̂s[7,1:T], label="P2 θ")
    plot!(p3, [times[t], times[t]], [-pi, pi], label="", color=:black)

    # plot 3
    title3 = "LF estimated velocity (x̂) over time"
    p4 = plot(legend=:outertopright, xlabel="t (s)", ylabel="vel (m/2)", title=title3)
    plot!(p4, times[1:T], x̂s[4,1:T], label="P1 v")
    plot!(p4, times[1:T], x̂s[8,1:T], label="P2 v")
    plot!(p4, [times[t], times[t]], [-1, 1], label="", color=:black)


    # Add particles
    num_iters = [0, 0]
    for n in 1:num_particles

        num_iter = sg_objs[t].num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])
        # println("num iters 1, 2: ", sg_objs[t].num_iterations, " ", sg_objs[t].num_iterations[n])

        x1_idx = 1
        y1_idx = 2
        x2_idx = 5
        y2_idx = 6

        xks = sg_objs[t].xks[n, num_iter, :, :]

        # TODO(hamzah) - change color based on which agent is leader
        scatter!(p1, xks[x1_idx, :], xks[y1_idx, :], color=:black, markersize=0.5, label="")

        color = (sg_objs[t].leader_idxs[n] == 1) ? :blue : :red
        scatter!(p1, [xks[x1_idx, 2]], [xks[y1_idx, 2]], color=color, markersize=3., label="")

        scatter!(p1, xks[x2_idx, :], xks[y2_idx, :], color=:black, markersize=0.5, label="")
        scatter!(p1, [xks[x2_idx, 2]], [xks[y2_idx, 2]], color=color, markersize=3., label="")
    end

    # plot 4
    title5 = "Input acceleration controls (u) over time"
    p5 = plot(legend=:outertopright, xlabel="t (s)", ylabel="accel. (m/s^2)", title=title5)
    plot!(p5, times[1:T], us[1][1, 1:T], label="P1 ω")
    plot!(p5, times[1:T], us[2][1, 1:T], label="P2 ω")
    plot!(p5, times[1:T], us[1][2, 1:T], label="P1 a")
    plot!(p5, times[1:T], us[2][2, 1:T], label="P2 a")
    plot!(p5, [times[t], times[t]], [-1, 1], label="", color=:black)

    # probability plot - plot 5
    title6 = "Probability over time"
    p6 = plot(xlabel="t (s)", ylabel="prob. leadership", ylimit=(-0.1, 1.1), label="", legend=:outertopright, title=title6)
    plot!(p6, times[1:T], probs[1:T])
    plot!(p6, times[1:T], (leader_idx%2) * ones(T), label="truth")
    plot!(p6, [times[t], times[t]], [0, 1], label="", color=:black)

    plot(p1, p2, p3, p4, p5, p6, plot_title=plot_title, layout = p, size=(1260, 1080))
end

# Speeds up call to gif (p.1/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
previous_GKSwstype = get(ENV, "GKSwstype", "")
ENV["GKSwstype"] = "100"

println("giffying...")
filename = string("unicycle_quadratic_leadfilt_",string(Dates.now()),".gif")
gif(anim, filename, fps=10)
println("done")

# Speeds up call to gif (p.2/2) - https://discourse.julialang.org/t/why-is-my-animate-loop-super-slow/43685/4
ENV["GKSwstype"] = previous_GKSwstype
