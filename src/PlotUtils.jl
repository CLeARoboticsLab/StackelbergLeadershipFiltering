# TODO(hamzah) Implement the following plots in here if not trivial. To be rolled into a plots package for the lab later on.
# - probabilites (multiple actors)
# - state vs. time (multiple actors)
# - position in two dimensions (multiple actors)
using LaTeXStrings
using Plots

# By default rotates clockwise 90 degrees
function rotate_state(dyn::UnicycleDynamics, xs)
    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    rotated_xs = deepcopy(xs)
    rotated_xs[x1_idx, :] = xs[y1_idx, :]
    rotated_xs[y1_idx, :] = -xs[x1_idx, :]
    rotated_xs[x2_idx, :] = xs[y2_idx, :]
    rotated_xs[y2_idx, :] = -xs[x2_idx, :]

    return rotated_xs
end
export rotate_state

# TODO(hamzah) - refactor this to be tied DoubleIntegrator Dynamics instead of Linear Dynamics.
function plot_states_and_controls(dyn::LinearDynamics, times, xs, us)
    @assert num_agents(dyn) == 2
    @assert xdim(dyn) == 8
    @assert udim(dyn, 1) == 2
    @assert udim(dyn, 2) == 2
    x₁ = xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel=L"$x$ (m)", ylabel=L"$y$ (m)")
    plot!(q1, xs[x1_idx, :], xs[y1_idx,:], label="P1 pos", color=:red)
    plot!(q1, xs[x2_idx,:], xs[y2_idx, :], label="P2 pos", color=:blue)

    q1 = scatter!([x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="P1 start")
    q1 = scatter!([x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="P2 start")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[x1_idx,:], label="P1 px")
    plot!(times, xs[x2_idx,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[y1_idx,:], label="P1 py")
    plot!(times, xs[y2_idx,:], label="P2 py")

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

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[x1_idx, :], xs[y1_idx, :], label="P1 pos", color=:red)
    plot!(q1, xs[x2_idx,:], xs[y2_idx, :], label="P2 pos", color=:blue)

    q1 = scatter!([x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="start P1")
    q1 = scatter!([x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[x1_idx,:], label="P1 px")
    plot!(times, xs[x2_idx,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[y1_idx,:], label="P1 py")
    plot!(times, xs[y2_idx,:], label="P2 py")

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


# This function makes an x-y plot containing (1) the ground truth trajectory,
#                                            (2) simulated measured positions of the trajectory, and 
#                                            (3) the estimated position trajectory produced by the leadership filter.
function plot_leadership_filter_positions(dyn::Dynamics, true_xs, est_xs, zs)
    x₁ = true_xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    p1 = plot(ylabel=L"$y$ m", xlabel=L"$x$ m", ylimit=(-2.5, 2.5), xlimit=(-2.5, 2.5))
    plot!(p1, true_xs[x1_idx, :], true_xs[y1_idx, :], label="True P1", color=:red, linewidth=2, ls=:dash)
    plot!(p1, true_xs[x2_idx, :], true_xs[y2_idx, :], label="True P2", color=:blue, linewidth=2, ls=:dash)

    plot!(p1, est_xs[x1_idx, :], est_xs[y1_idx, :], label="Est. P1", color=:orange)
    plot!(p1, est_xs[x2_idx, :], est_xs[y2_idx, :], label="Est. P2", color=:turquoise2)

    scatter!(p1, zs[x1_idx, :], zs[y1_idx, :], color=:red, marker=:plus, ms=3, markerstrokewidth=0, label="Meas. P1")
    scatter!(p1, zs[x2_idx, :], zs[y2_idx, :], color=:blue, marker=:plus, ms=3, markerstrokewidth=0, label="Meas. P2")

    scatter!(p1, [x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="P1 Start")
    scatter!(p1, [x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="P2 Start")

    return p1
end
export plot_leadership_filter_positions

# This function makes an x-y plot (1) the ground truth trajectories (in black to avoid color conflict),
#                                 (2) the estimated position trajectory produced by the leadership filter, and
#                                 (3) particles representing the solutions to the games played within the Stackelberg
#                                     measurement model. The waypoint at the current time is highlighted. Measurement
#                                     data and particles are colored to match the agent assumed by the particle to be
#                                     leader.
function plot_leadership_filter_measurement_details(num_particles, sg_t, true_xs, est_xs; transform_particle_fn=(xs)->xs)
    x₁ = true_xs[:, 1]

    x1_idx = xidx(sg_t.dyn, 1)
    y1_idx = yidx(sg_t.dyn, 1)
    x2_idx = xidx(sg_t.dyn, 2)
    y2_idx = yidx(sg_t.dyn, 2)

    p2 = plot(ylabel=L"$y$ m", xlabel=L"$x$ m", ylimit=(-2.5, 2.5), xlimit=(-2.5, 2.5))
    plot!(p2, true_xs[x1_idx, :], true_xs[y1_idx, :], label="True P1", color=:black, linewidth=3)
    plot!(p2, true_xs[x2_idx, :], true_xs[y2_idx, :], label="True P2", color=:black, linewidth=3)

    plot!(p2, est_xs[x1_idx, :], est_xs[y1_idx, :], label="Est. P1", color=:orange)
    plot!(p2, est_xs[x2_idx, :], est_xs[y2_idx, :], label="Est. P2", color=:turquoise2)

    scatter!(p2, [x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="P1 Start")
    scatter!(p2, [x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="P2 Start")

    # Add particles
    for n in 1:num_particles

        num_iter = sg_t.num_iterations[n]

        # println("particle n thinks leader is: ", n)
        # println("num iters 1, 2: ", sg_t.num_iterations, " ", sg_t.num_iterations[n])
        # println("num iters 1, 2: ", sg_t.num_iterations, " ", sg_t.num_iterations[n])

        xks = transform_particle_fn(sg_t.xks[n, num_iter, :, :])

        # TODO(hamzah) - change color based on which agent is leader
        color = (sg_t.leader_idxs[n] == 1) ? "red" : "blue"
        scatter!(p2, xks[x1_idx, :], xks[y1_idx, :], color=color, markersize=0.3, markerstrokewidth=0, label="")
        scatter!(p2, [xks[x1_idx, 2]], [xks[y1_idx, 2]], color=color, markersize=2., markerstrokewidth=0, label="")

        scatter!(p2, xks[x2_idx, :], xks[y2_idx, :], color=color, markersize=0.3, markerstrokewidth=0, label="")
        scatter!(p2, [xks[x2_idx, 2]], [xks[y2_idx, 2]], color=color, markersize=3., markerstrokewidth=0, label="")
    end

    return p2
end
export plot_leadership_filter_measurement_details

# This function generates two probability plots (both lines on one plot is too much to see), one for the probablity of
# each agent as leader.
function make_probability_plots(leader_idx, times, t_idx, probs; include_gt=true)
    t = times[t_idx]
    T = length(times)

    # probability plot for P1 - plot 5
    p5 = plot(xlabel="t (s)", ylabel=L"""$\mathbb{P}(L=\mathcal{A}_1)$""", ylimit=(-0.1, 1.1), label="")
    plot!(p5, times, probs, color=:red, label="P1")
    if !include_gt
        plot!(p5, times, (leader_idx%2) * ones(T), label="Truth", color=:green, linestyle=:dash, linewidth=2)
    end
    plot!(p5, [t, t], [-0.05, 1.05], label="t=$(round.(t, digits=3)) s", color=:black, linestyle=:dot, linewidth=3)

     # probability plot for P2 - plot 6
    p6 = plot(xlabel="t (s)", ylabel=L"""$\mathbb{P}(L=\mathcal{A}_2)$""", ylimit=(-0.1, 1.1), label="")
    plot!(p6, times, 1 .- probs, color=:blue, label="P2")
    if !include_gt
        plot!(p6, times, ((leader_idx+1)%2) * ones(T), label="Truth", color=:green, linestyle=:dash, linewidth=2)
    end
    plot!(p6, [t, t], [-0.05, 1.05], label="t=$(round.(t, digits=3)) s", color=:black, linestyle=:dot, linewidth=3)

    return p5, p6
end
export make_probability_plots
