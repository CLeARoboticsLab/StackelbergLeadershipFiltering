# TODO(hamzah) Implement the following plots in here if not trivial. To be rolled into a plots package for the lab later on.
# - probabilites (multiple actors)
# - state vs. time (multiple actors)
# - position in two dimensions (multiple actors)
using LaTeXStrings
using Plots

function get_standard_plot(;include_legend=:outertop, columns=-1, legendfontsize=18)
    return plot(legendfontsize=legendfontsize, tickfontsize=18, fontsize=18, labelfontsize=18, legend=include_legend, legend_columns=columns, fg_legend = :transparent)
end
export get_standard_plot

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
function plot_states_and_controls(dyn::LinearDynamics, times, xs, us; include_legend=:none, ms0=6)
    @assert num_agents(dyn) == 2
    @assert xdim(dyn) == 8
    @assert udim(dyn, 1) == 2
    @assert udim(dyn, 2) == 2
    x₁ = xs[:, 1]

    num_times = length(times)

    # Indicates that only the first time has a marker.
    marksize = vcat([ms0], zeros(num_times-1))

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    title1 = "pos. traj."
    q1 = get_standard_plot(;include_legend)
    plot!(title=title1, xlabel="Horizontal Position (m)", ylabel="Vertical Position (m)")
    plot!(q1, xs[x1_idx, :], xs[y1_idx,:], label=L"\mathcal{A}_1", linewidth=3, color=:red, marker=:circle,  markersize=marksize, markerstrokewidth=0)
    plot!(q1, xs[x2_idx,:], xs[y2_idx, :], label=L"\mathcal{A}_2", linewidth=3, color=:blue, marker=:circle,  markersize=marksize, markerstrokewidth=0)

    q1 = scatter!([x₁[x1_idx]], [x₁[y1_idx]], color=:red, label=L"$\mathcal{A}_1$ start")
    q1 = scatter!([x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label=L"$\mathcal{A}_2$ start")

    title2a = "x-pos"
    q2a = get_standard_plot(;include_legend)
    plot!(title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[x1_idx,:], label=L"\mathcal{A}_1~p_x")
    plot!(times, xs[x2_idx,:], label=L"\mathcal{A}_2~p_x")

    title2b = "y-pos"
    q2b = get_standard_plot(;include_legend)
    plot!(title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[y1_idx,:], label=L"\mathcal{A}_1~p_y")
    plot!(times, xs[y2_idx,:], label=L"\mathcal{A}_2~p_y")

    title3 = "x-vel"
    q3 = get_standard_plot(;include_legend)
    plot!(title=title3, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[2,:], label=L"\mathcal{A}_1~v_x")
    plot!(times, xs[6,:], label=L"\mathcal{A}_2~v_x")

    title4 = "y-vel"
    q4 = get_standard_plot(;include_legend)
    plot!(title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label=L"\mathcal{A}_1~v_y")
    plot!(times, xs[8,:], label=L"\mathcal{A}_2~v_y")

    title5 = "x-accel"
    q5 = get_standard_plot(;include_legend)
    plot!(title=title5, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][1, :], label=L"\mathcal{A}_1~a_x")
    plot!(times, us[2][1, :], label=L"\mathcal{A}_2~a_x")

    title6 = "y-accel"
    q6 = get_standard_plot(;include_legend)
    plot!(title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label=L"\mathcal{A}_1~a_y")
    plot!(times, us[2][2, :], label=L"\mathcal{A}_2~a_y")

    return q1, q2a, q2b, q3, q4, q5, q6
end

# TODO(hamzah) - refactor this to adjust based on number of players instead of assuming 2.
function plot_states_and_controls(dyn::UnicycleDynamics, times, xs, us; include_legend=:none)
    @assert num_agents(dyn) == 2

    x₁ = xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    title1 = "pos. traj."
    q1 = get_standard_plot(;include_legend)
    plot!(title=title1, xlabel="Horizontal Position (m)", ylabel="Vertical Position (m)")
    plot!(q1, xs[x1_idx, :], xs[y1_idx, :], label=L"\mathcal{A}_1", color=:red)
    plot!(q1, xs[x2_idx,:], xs[y2_idx, :], label=L"\mathcal{A}_2", color=:blue)

    q1 = scatter!([x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="start P1")
    q1 = scatter!([x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="start P2")

    title2a = "x-pos"
    q2a = get_standard_plot(;include_legend)
    plot!(title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[x1_idx,:], label=L"\mathcal{A}_1~p_x")
    plot!(times, xs[x2_idx,:], label=L"\mathcal{A}_2~p_x")

    title2b = "y-pos"
    q2b = get_standard_plot(;include_legend)
    plot!(title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[y1_idx,:], label=L"\mathcal{A}_1~p_y")
    plot!(times, xs[y2_idx,:], label=L"\mathcal{A}_2~p_y")

    title3 = "θ"
    q3 = get_standard_plot(;include_legend)
    plot!(title=title3, xlabel="t (s)", ylabel="θ (rad)")
    plot!(times, wrap_angle.(xs[3,:]), label=L"\mathcal{A}_1~θ")
    plot!(times, wrap_angle.(xs[7,:]), label=L"\mathcal{A}_2~θ")

    title4 = "vel"
    q4 = get_standard_plot(;include_legend)
    plot!(title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label=L"\mathcal{A}_1~v")
    plot!(times, xs[8,:], label=L"\mathcal{A}_2~v")

    title5 = "ang vel"
    q5 = get_standard_plot(;include_legend)
    plot!(title=title5, xlabel="t (s)", ylabel="ang. vel. (rad/s)")
    plot!(times, us[1][1, :], label=L"\mathcal{A}_1~ω")
    plot!(times, us[2][1, :], label=L"\mathcal{A}_2~ω")

    title6 = "accel"
    q6 = get_standard_plot(;include_legend)
    plot!(title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label=L"$\mathcal{A}_1$ accel")
    plot!(times, us[2][2, :], label=L"$\mathcal{A}_2$ accel")

    return q1, q2a, q2b, q3, q4, q5, q6
end

export plot_states_and_controls


function plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)
    # Plot convergence metric max absolute state difference between iterations.
    conv_x = cumsum(ones(num_iters)) .- 1
    title8 = "convergence"
    q8 = get_standard_plot()
    plot!(title=title8, yaxis=:log, xlabel="# Iterations", ylabel="Max Abs. State Difference")

    conv_sum = conv_metrics[1, 1:num_iters] #+ conv_metrics[2, 1:num_iters]
    if num_iters > 2
        plot!(q8, conv_x, conv_sum, label=L"$\ell_{\infty}$ Merit", color=:green, linewidth=3)
    else
        scatter!(q8, conv_x, conv_sum, label=L"$\ell_{\infty}$ Merit", color=:green)
    end
    plot!(q8, [0, num_iters-1], [threshold, threshold], label="Threshold", color=:purple, linestyle=:dot, linewidth=3)


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
    q9 = get_standard_plot()
    plot!(title=title9, yaxis=:log, xlabel="# Iterations", ylabel="Cost")

    if num_iters > 2
        plot!(conv_x, costs_1[1:num_iters], label=L"\mathcal{A}_1", color=:red, linewidth=2)
        plot!(conv_x, costs_2[1:num_iters], label=L"\mathcal{A}_2", color=:blue, linewidth=2)

        cost_sum = costs_1[1:num_iters] + costs_2[1:num_iters]
        plot!(conv_x, cost_sum, label="Total", color=:purple, linewidth=3)
    else
        scatter!(conv_x, costs_1[1:num_iters], label=L"\mathcal{A}_1", color=:red, marker=:plus, ms=8)
        scatter!(conv_x, costs_2[1:num_iters], label=L"\mathcal{A}_2", color=:blue, marker=:plus, ms=8)

        cost_sum = costs_1[1:num_iters] + costs_2[1:num_iters]
        scatter!(conv_x, cost_sum, label="Total", color=:purple, xaxis=[-0.1, 1.1], xticks=[0, 1])
    end

    return q8, q9
end
export plot_convergence_and_costs


# This function makes an x-y plot containing (1) the ground truth trajectory and
#                                            (2) the estimated position trajectory produced by the leadership filter.
function plot_leadership_filter_positions(dyn::Dynamics, true_xs, est_xs)
    x₁ = true_xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    p1 = get_standard_plot(;columns=2, legendfontsize=18)
    plot!(ylabel="Vertical Position (m)", xlabel="Horizontal Position (m)")
    plot!(p1, true_xs[x1_idx, :], true_xs[y1_idx, :], label=L"$\mathcal{A}_1$ Ground Truth", color=:red, linewidth=2, ls=:dash)
    plot!(p1, est_xs[x1_idx, :], est_xs[y1_idx, :], label=L"$\mathcal{A}_1$ Estimate", color=:orange)
    # scatter!(p1, zs[x1_idx, :], zs[y1_idx, :], color=:red, marker=:plus, ms=6, markerstrokewidth=0, label=L"\mathcal{A}_1 Measurements")
    scatter!(p1, [x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="")# L"$\mathcal{A}_1$ Start")

    plot!(p1, true_xs[x2_idx, :], true_xs[y2_idx, :], label=L"$\mathcal{A}_2$ Ground Truth", color=:blue, linewidth=2, ls=:dash)
    plot!(p1, est_xs[x2_idx, :], est_xs[y2_idx, :], label=L"$\mathcal{A}_2$ Estimate", color=:turquoise2)
    # scatter!(p1, zs[x2_idx, :], zs[y2_idx, :], color=:blue, marker=:plus, ms=6, markerstrokewidth=0, label=L"\mathcal{A}_2 Measurements")
    scatter!(p1, [x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="")# L"$\mathcal{A}_2$ Start")

    return p1
end
export plot_leadership_filter_positions

# This function makes an x-y plot containing (1) the ground truth trajectory and
#                                            (2) simulated measured positions of the trajectory.
function plot_leadership_filter_measurements(dyn::Dynamics, true_xs, zs; show_meas_annotation=nothing)
    x₁ = true_xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    p1 = get_standard_plot(;columns=1, legendfontsize=18)
    
    plot!(ylabel="Vertical Position (m)", xlabel="Horizontal Position (m)")
    if !isnothing(show_meas_annotation)
        # Remove axis and grid.
        plot!(p1, axis=([], false), grid=true, xlabel="", ylabel="")
        annotate!(p1, 1.2, 1.9, text("($(show_meas_annotation)) observations", 30))
    end

    
    plot!(p1, true_xs[x1_idx, :], true_xs[y1_idx, :], color=:black, linewidth=2, ls=:solid, label="")#L"$\mathcal{A}_1$ Ground Truth")
    # plot!(p1, est_xs[x1_idx, :], est_xs[y1_idx, :], label=L"\mathcal{A}_1 Estimate", color=:orange)
    scatter!(p1, zs[x1_idx, :], zs[y1_idx, :], color=:red, marker=:plus, ms=6, markerstrokewidth=0, label=L"$\mathcal{A}_1$ Measurements")
    scatter!(p1, [x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="")#L"$\mathcal{A}_1$ Start")

    plot!(p1, true_xs[x2_idx, :], true_xs[y2_idx, :], color=:black, linewidth=2, ls=:solid, label="")#L"$\mathcal{A}_2$ Ground Truth")
    # plot!(p1, est_xs[x2_idx, :], est_xs[y2_idx, :], label=L"\mathcal{A}_2 Estimate", color=:turquoise2)
    scatter!(p1, zs[x2_idx, :], zs[y2_idx, :], color=:blue, marker=:plus, ms=6, markerstrokewidth=0, label=L"$\mathcal{A}_2$ Measurements")
    scatter!(p1, [x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="")#L"$\mathcal{A}_2$ Start")

    return p1
end
export plot_leadership_filter_measurements

# This function makes an x-y plot (1) the ground truth trajectories (in black to avoid color conflict),
#                                 (2) the estimated position trajectory produced by the leadership filter, and
#                                 (3) particles representing the solutions to the games played within the Stackelberg
#                                     measurement model. The waypoint at the current time is highlighted. Measurement
#                                     data and particles are colored to match the agent assumed by the particle to be
#                                     leader.
function plot_leadership_filter_measurement_details(num_particles, sg_t::SILQGamesObject, true_xs, est_xs; transform_particle_fn=(xs)->xs, include_all_labels=false)
    plot_leadership_filter_measurement_details(sg_t.dyn, sg_t.leader_idxs, num_particles, sg_t.num_iterations, sg_t.xks, true_xs, est_xs; transform_particle_fn=transform_particle_fn, include_all_labels=include_all_labels)
end

function plot_leadership_filter_measurement_details(dyn::Dynamics, particle_leader_idxs_t, num_particles, particle_num_iterations_t, particle_traj_xs_t, true_xs, est_xs; transform_particle_fn=(xs)->xs, t=nothing, letter=nothing, include_all_labels=false)
    x₁ = true_xs[:, 1]

    x1_idx = xidx(dyn, 1)
    y1_idx = yidx(dyn, 1)
    x2_idx = xidx(dyn, 2)
    y2_idx = yidx(dyn, 2)

    if include_all_labels
        p2 = get_standard_plot(;columns=3, legendfontsize=12)
        p1_est_label = L"$\mathcal{A}_1$ Estimate"
        # p1_truth_label = L"$\mathcal{A}_1$ Truth"
        p1_truth_label = "Truth"
        p2_est_label = L"$\mathcal{A}_2$ Estimate"
        # p2_truth_label = L"$\mathcal{A}_2$ Truth"
        p2_truth_label = ""
    else
        p2 = get_standard_plot(;columns=2)
        p1_est_label = ""
        p1_truth_label = ""
        p2_est_label = ""
        p2_truth_label = ""
        # Remove axis and grid.
        plot!(axis=([], false), grid=true)
    end
    plot!(ylabel="Vertical Position (m)", xlabel="Horizontal Position (m)")

    # If t is provided, annotate the plot.
    if !isnothing(t) && !isnothing(letter)
        plot!(ylabel="", xlabel="")
        annotate!(p2, 1.2, 1.8, text("($(letter)) meas. model\ntime step $(t)", 30))
    end

    plot!(p2, true_xs[x1_idx, :], true_xs[y1_idx, :], color=:black, linewidth=3, label=p1_truth_label)
    plot!(p2, est_xs[x1_idx, :], est_xs[y1_idx, :], color=:orange, label=p1_est_label)
    scatter!(p2, [x₁[x1_idx]], [x₁[y1_idx]], color=:red, label="") # L"$\mathcal{A}_1$ Start")

    plot!(p2, true_xs[x2_idx, :], true_xs[y2_idx, :], color=:black, linewidth=3, label=p2_truth_label)
    plot!(p2, est_xs[x2_idx, :], est_xs[y2_idx, :], color=:turquoise2, label=p2_est_label)
    scatter!(p2, [x₁[x2_idx]], [x₁[y2_idx]], color=:blue, label="") # L"$\mathcal{A}_2$ Start")

    # Add particles
    has_labeled_p1 = false
    has_labeled_p2 = false
    for n in 1:num_particles

        num_iter = particle_num_iterations_t[n]
        xks = transform_particle_fn(particle_traj_xs_t[n, :, :])

        # TODO(hamzah) - change color based on which agent is leader
        does_p1_lead = (particle_leader_idxs_t[n] == 1)

        color = (does_p1_lead) ? "red" : "blue"
        label_1 = (!has_labeled_p1 && does_p1_lead) ? L"$\mathcal{A}_1$ Meas. Model" : ""
        label_2 = (!has_labeled_p2 && !does_p1_lead) ? L"$\mathcal{A}_2$ Meas. Model" : ""

        if label_1 != ""
            has_labeled_p1 = true
        end
        if label_2 != ""
            has_labeled_p2 = true
        end

        scatter!(p2, xks[x1_idx, :], xks[y1_idx, :], color=color, markersize=1., markerstrokewidth=0, label="")
        scatter!(p2, [xks[x1_idx, 2]], [xks[y1_idx, 2]], color=color, markersize=3., markerstrokewidth=0, label=label_1)

        scatter!(p2, xks[x2_idx, :], xks[y2_idx, :], color=color, markersize=1., markerstrokewidth=0, label="")
        scatter!(p2, [xks[x2_idx, 2]], [xks[y2_idx, 2]], color=color, markersize=3., markerstrokewidth=0, label=label_2)
    end

    return p2
end
export plot_leadership_filter_measurement_details

# This function generates two probability plots (both lines on one plot is too much to see), one for the probablity of
# each agent as leader. Plot both by default.
function make_probability_plots(times, probs; player_to_plot=nothing, t_idx=nothing, include_gt=nothing, stddevs=nothing)
    T = length(times)
    lower_p1, upper_p1 = (isnothing(stddevs)) ? (zeros(T), zeros(T)) : stddevs

    # probability plot for P1 - plot 5
    plot = get_standard_plot(columns=4)
    plot!(xlabel="t (s)", ylabel="Leadership Probability", ylimit=(-0.1, 1.1), label="")
    if player_to_plot == 1 || isnothing(player_to_plot)
        # L"""$\mathbb{P}(H_t=\mathcal{A}_1)$"""
        if !isnothing(include_gt)
            plot!(plot, times, (include_gt%2) * ones(T), label="Truth", color=:red, linestyle=:dot, linewidth=2)
        end

        # Bound the stddevs to avoid going above 1 or below 0.
        plot!(plot, times, probs, color=:red, label=L"\mathcal{A}_1", ribbon=(lower_p1, upper_p1), fillalpha=0.3)
    end

    # probability plot for P2 - plot 6
    # p6 = get_standard_plot()
    if player_to_plot == 2 || isnothing(player_to_plot)
        # plot!(xlabel="t (s)", ylabel="Leadership Probability", ylimit=(-0.1, 1.1), label="")
        # L"""$\mathbb{P}(H_t=\mathcal{A}_2)$"""
        if !isnothing(include_gt)
            plot!(plot, times, ((include_gt+1)%2) * ones(T), label="Truth", color=:blue, linestyle=:dot, linewidth=2)
        end

        lower_p2, upper_p2 = upper_p1, lower_p1
        plot!(plot, times, 1 .- probs, color=:blue, label=L"\mathcal{A}_2", ribbon=(lower_p2, upper_p2), fillalpha=0.3)
    end

    # Draw the lines of interest.
    if !isnothing(t_idx)
        label_str = join(t_idx, ",")
        for idx in t_idx
            t = times[idx]
            plot!(plot, [t, t], [-0.05, 1.0], color=:black, linestyle=:solid, linewidth=3, label="")#"t=$(label_str)",)
            annotate!(plot, times[idx], 1.1, idx, 24)
            # label_str = ""
            # vline!(plot, [t], label="Max Iterations", color=:black, linewidth=3)
        end
        annotate!(plot, 8.5, 1.1, "steps", 18)
    end


    # if isnothing(player_to_plot)
    #     plot!(ylabel=L"""$\mathbb{P}(H_t=\mathcal{A}_i)$""", label="")
    # end

    return plot
end
export make_probability_plots
