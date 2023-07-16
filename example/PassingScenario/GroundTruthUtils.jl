# A file for utilities for generating the ground truth that is passed into the leadership filter.
# There are multiple ways to do this, but for now we use SILQGames to generate it.

using StackelbergControlHypothesesFiltering
using Plots

function generate_gt_from_silqgames(sg_obj, leader_idx::Int, times, x₁, us_1)

    # ground truth generation config variables
    leader_idx = 1

    t₀ = times[1]

    xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, t₀, times, x₁, us_1)

    println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
    final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
    println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))

    return xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs
end

function plot_silqgames_gt(dyn, times, xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs)
    # PLOTS A: Plot states/controls.
    l = @layout [
        a{0.3h}; [grid(2,3)]
    ]

    # q = @layout [a b; c d ;e f; g h]
    pos_plot, p2, p3, p4, p5, p6, p7 = plot_states_and_controls(dyn, times, xs_k, us_k)
    plot(pos_plot, p2, p3, p4, p5, p6, p7, layout = l)


    # # PLOTS B: Plot convergence metrics/costs separately.
    # m = @layout [a; b]

    # conv_x = cumsum(ones(num_iters)) .- 1
    # title8 = "conv. (|⋅|∞)"
    # q8 = plot(title=title8, yaxis=:log, legend=:outertopright)
    # plot!(conv_x, conv_metrics[1, 1:num_iters], label="p1")
    # plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2")

    # conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
    # plot!(conv_x, conv_sum, label="total")

    # title9 = "evaluated costs"
    # q9 = plot(title=title9, yaxis=:log, legend=:outertopright)
    # plot!(conv_x, evaluated_costs[1, 1:num_iters], label="p1")
    # plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2")

    # cost_sum = evaluated_costs[1, 1:num_iters] + evaluated_costs[2, 1:num_iters]
    # plot!(conv_x, cost_sum, label="total")

    # plot(q8, q9, layout = m)
end
