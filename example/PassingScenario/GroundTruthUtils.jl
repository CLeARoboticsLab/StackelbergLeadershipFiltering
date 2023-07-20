# A file for utilities for generating the ground truth that is passed into the leadership filter.
# There are multiple ways to do this, but for now we use SILQGames to generate it.

using StackelbergControlHypothesesFiltering

using JLD
using Plots

# For saving the trajectory - in case we want to use it later.
function save_trajectory(filename, xs, us, times, T, dt)
    # Create some data to save
    data = Dict("xs" => xs, "us" => us, "times" => times, "T" => T, "dt" => dt)

    # Save the data to a file
    @save "$(filename).jld" data
end

function load_trajectory(filename)
    @load "$(filename).jld" data

    return data["xs"], data["us"], data["times"], data["T"], data["dt"]
end


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
    # conv_plot, cost_plot = plot_convergence_and_costs(num_iters, threshold, conv_metrics, evaluated_costs)
    # m = @layout [a; b]
    # plot(conv_plot, cost_plot, layout = m)
end
