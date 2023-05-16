using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("quadratic_nonlinear_parameters.jl")

costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

leader_idx=2
num_runs=1

# config variables
threshold=0.001
max_iters=1000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, T, dyn, costs;
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))


using ElectronDisplay
using Plots

# Plot positions, other two states, controls, and convergence.
q = @layout [a b c; d e f; g h i]

q1, q2, q3, q4, q5, q6, q7 = plot_states_and_controls(dyn, times, xs_k, us_k)

# Plot convergence.
conv_x = cumsum(ones(num_iters)) .- 1
title8 = "||k||^2 by player"
q8 = plot(title=title8, yaxis=:log, legend=:outertopright)
plot!(conv_x, conv_metrics[1, 1:num_iters], label="p1")
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2")

conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
plot!(conv_x, conv_sum, label="total")

q9 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log, legend=:outertopright)
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)

cost_sum = evaluated_costs[1, 1:num_iters] + evaluated_costs[2, 1:num_iters]
plot!(conv_x, cost_sum, label="total", yaxis=:log)

plot(q1, q2, q3, q4, q5, q6, q7, q8, q9, layout = q)



