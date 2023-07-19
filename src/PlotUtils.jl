# TODO(hamzah) Implement the following plots in here if not trivial. To be rolled into a plots package for the lab later on.
# - probabilites (multiple actors)
# - state vs. time (multiple actors)
# - position in two dimensions (multiple actors)
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
