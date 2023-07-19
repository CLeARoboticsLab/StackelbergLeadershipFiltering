# TODO(hamzah) Implement the following plots in here if not trivial. To be rolled into a plots package for the lab later on.
# - probabilites (multiple actors)
# - state vs. time (multiple actors)
# - position in two dimensions (multiple actors)
using LaTeXStrings
using Plots

# TODO(hamzah) - refactor this to be tied DoubleIntegrator Dynamics instead of Linear Dynamics.
function plot_states_and_controls(dyn::LinearDynamics, times, xs, us)
    @assert num_agents(dyn) == 2
    @assert xdim(dyn) == 8
    @assert udim(dyn, 1) == 2
    @assert udim(dyn, 2) == 2
    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[3, :], label="P1 pos", color=:red)
    plot!(q1, xs[5, :], xs[7, :], label="P2 pos", color=:blue)

    q1 = scatter!([x₁[1]], [x₁[3]], color=:red, label="start P1")
    q1 = scatter!([x₁[5]], [x₁[7]], color=:blue, label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[1,:], label="P1 px")
    plot!(times, xs[5,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[3,:], label="P1 py")
    plot!(times, xs[7,:], label="P2 py")

    title3 = "x-vel"
    q3 = plot(legend=:outertopright, title=title3, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[2,:], label="P1 vx")
    plot!(times, xs[6,:], label="P2 vx")

    title4 = "y-vel"
    q4 = plot(legend=:outertopright, title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label="P1 vy")
    plot!(times, xs[8,:], label="P2 vy")

    title5 = "x-accel"
    q5 = plot(legend=:outertopright, title=title5, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][1, :], label="P1 ax")
    plot!(times, us[2][1, :], label="P2 ax")

    title6 = "y-accel"
    q6 = plot(legend=:outertopright, title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label="P1 ay")
    plot!(times, us[2][2, :], label="P2 ay")

    return q1, q2a, q2b, q3, q4, q5, q6
end

# TODO(hamzah) - refactor this to adjust based on number of players instead of assuming 2.
function plot_states_and_controls(dyn::UnicycleDynamics, times, xs, us)
    @assert num_agents(dyn) == 2

    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[2, :], label="P1 pos", color=:red)
    plot!(q1, xs[5, :], xs[6, :], label="P2 pos", color=:blue)

    q1 = scatter!([x₁[1]], [x₁[2]], color=:red, label="start P1")
    q1 = scatter!([x₁[5]], [x₁[6]], color=:blue, label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[1,:], label="P1 px")
    plot!(times, xs[5,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[2,:], label="P1 py")
    plot!(times, xs[6,:], label="P2 py")

    title3 = "θ"
    q3 = plot(legend=:outertopright, title=title3, xlabel="t (s)", ylabel="θ (rad)")
    plot!(times, wrap_angle.(xs[3,:]), label="P1 θ")
    plot!(times, wrap_angle.(xs[7,:]), label="P2 θ")

    title4 = "vel"
    q4 = plot(legend=:outertopright, title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label="P1 v")
    plot!(times, xs[8,:], label="P2 v")

    title5 = "ang vel"
    q5 = plot(legend=:outertopright, title=title5, xlabel="t (s)", ylabel="ang. vel. (rad/s)")
    plot!(times, us[1][1, :], label="P1 ω")
    plot!(times, us[2][1, :], label="P2 ω")

    title6 = "accel"
    q6 = plot(legend=:outertopright, title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label="P1 accel")
    plot!(times, us[2][2, :], label="P2 accel")

    return q1, q2a, q2b, q3, q4, q5, q6
end

export plot_states_and_controls


function plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)


    # Plot convergence metric max absolute state difference between iterations.
    conv_x = cumsum(ones(num_iters)) .- 1
    title8 = "convergence"
    q8 = plot(title=title8, yaxis=:log, xlabel="# Iterations", ylabel="Max Absolute State Difference")

    conv_sum = conv_metrics[1, 1:num_iters] #+ conv_metrics[2, 1:num_iters]
    plot!(q8, conv_x, conv_sum, label="Merit Fn", color=:green)
    plot!(q8, [0, num_iters-1], [threshold, threshold], label="Threshold", color=:purple, linestyle=:dot)


    costs_1 = evaluated_costs[1, 1:num_iters]
    costs_2 = evaluated_costs[2, 1:num_iters]

    # # Shift the cost if any are negative to ensure they become all positive for the log-scaled plot.
    min_cost1 = minimum(evaluated_costs[1, 1:num_iters-1])
    min_cost2 = minimum(evaluated_costs[2, 1:num_iters-1])

    if min_cost1 < 0
        costs_1 = costs_1 .+ 2 * abs(min_cost1)
    end
    if min_cost2 < 0
        costs_2 = costs_2 .+ 2 * abs(min_cost1)
    end

    title9 = "evaluated costs"
    q9 = plot(title=title9, yaxis=:log, xlabel="# Iterations", ylabel="Cost")
    plot!(conv_x, costs_1[1:num_iters], label="P1", color=:red)
    plot!(conv_x, costs_2[1:num_iters], label="P2", color=:blue)

    cost_sum = costs_1[1:num_iters] + costs_2[1:num_iters]
    plot!(conv_x, cost_sum, label="Total", color=:green, linestyle=:dash, linewidth=2)

    return q8, q9
end
export plot_convergence_and_costs


# This function plots an x-y plot along with measurement data of the positions and particle trajectories representing
# the games played within the Stackelberg measurement model.
function plot_leadership_filter_position(num_particles, sg_t, true_xs, est_xs, zs)
    x₁ = true_xs[:, 1]

    p1 = plot(ylabel=L"$y$ m", xlabel=L"$x$ m", ylimit=(-2.5, 2.5), xlimit=(-2.5, 2.5))
    plot!(p1, true_xs[1, :], true_xs[3, :], label="True P1", color=:red)
    plot!(p1, true_xs[5, :], true_xs[7, :], label="True P2", color=:blue)

    plot!(p1, est_xs[1, :], est_xs[3, :], label="Est. P1", color=:lightred)
    plot!(p1, est_xs[5, :], est_xs[7, :], label="Est. P2", color=:lightblue)

    scatter!(p1, zs[1, :], zs[3, :], color=:red, marker=:plus, ms=3, markerstrokewidth=0, label="Meas. P1")
    scatter!(p1, zs[5, :], zs[7, :], color=:blue, marker=:plus, ms=3, markerstrokewidth=0, label="Meas. P2")

    # Add particles
    for n in 1:num_particles

        num_iter = sg_t.num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_t.num_iterations, " ", sg_t.num_iterations[n])
        # println("num iters 1, 2: ", sg_t.num_iterations, " ", sg_t.num_iterations[n])

        x1_idx = xidx(dyn, 1)
        y1_idx = yidx(dyn, 1)
        x2_idx = xidx(dyn, 2)
        y2_idx = yidx(dyn, 2)

        xks = sg_t.xks[n, num_iter, :, :]

        # TODO(hamzah) - change color based on which agent is leader
        color = (sg_t.leader_idxs[n] == 1) ? "red" : "blue"
        scatter!(p1, xks[x1_idx, :], xks[y1_idx, :], color=color, markersize=0.5, markerstrokewidth=0, label="")
        scatter!(p1, [xks[x1_idx, 2]], [xks[y1_idx, 2]], color=color, markersize=3., markerstrokewidth=0, label="")

        scatter!(p1, xks[x2_idx, :], xks[y2_idx, :], color=color, markersize=0.5, markerstrokewidth=0, label="")
        scatter!(p1, [xks[x2_idx, 2]], [xks[y2_idx, 2]], color=color, markersize=3., markerstrokewidth=0, label="")
    end

    return p1
end
export plot_leadership_filter_position

# This function generates two probability plots (both lines on one plot is too much to see), one for the probablity of
# each agent as leader.
function make_probability_plots(times, t_idx, probs)
    t = times[t_idx]

    # probability plot for P1 - plot 5
    p5 = plot(xlabel="t (s)", ylabel=L"""$\mathbb{P}(L=\mathcal{A}_1)$""", ylimit=(-0.1, 1.1), label="")
    plot!(p5, times[1:T], probs[1:T], color=:red, label="P1")
    plot!(p5, times[1:T], (leader_idx%2) * ones(T), label="Truth", color=:green, linestyle=:dash, linewidth=2)
    plot!(p5, [t, t], [-0.05, 1.05], label="t=$(round.(t, sigdigits=3)) s", color=:black, linestyle=:dot, linewidth=3)

     # probability plot for P2 - plot 6
    p6 = plot(xlabel="t (s)", ylabel=L"""$\mathbb{P}(L=\mathcal{A}_2)$""", ylimit=(-0.1, 1.1), label="")
    plot!(p6, times[1:T], 1 .- probs[1:T], color=:blue, label="P2")
    plot!(p6, times[1:T], ((leader_idx+1)%2) * ones(T), label="Truth", color=:green, linestyle=:dash, linewidth=2)
    plot!(p6, [t, t], [-0.05, 1.05], label="t=$(round.(t, sigdigits=3)) s", color=:black, linestyle=:dot, linewidth=3)

    return p5, p6
end
export make_probability_plots
