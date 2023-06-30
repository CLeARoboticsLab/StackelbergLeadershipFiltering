using StackelbergControlHypothesesFiltering

using LinearAlgebra

# includes linear dynamics and quadratic costs
include("nonLQ_parameters.jl")

num_runs=1

# config variables
threshold=1e-3
max_iters=1000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, T, dyn, costs;
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals)
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))


using ElectronDisplay
using Plots

# Plot positions, other two states, controls, and convergence.
q = @layout [a b; c d; e f]
# q = @layout [a b; c d]#; e f]

# Indices for shepherd and sheep game.
p1x_idx = xidx(dyn, 1)
p1y_idx = yidx(dyn, 1)
p2x_idx = xidx(dyn, 2)
p2y_idx = yidx(dyn, 2)

q1 = plot(legend=:outertopright)
plot!(q1, xs_k[p1x_idx, :], xs_k[p1y_idx, :], label="P1 pos")
plot!(q1, xs_k[p2x_idx, :], xs_k[p2y_idx, :], label="P2 pos")

q1 = scatter!([x₁[p1x_idx]], [x₁[p1y_idx]], color="blue", label="start P1")
q1 = scatter!([x₁[p2x_idx]], [x₁[p2y_idx]], color="red", label="start P2")

q2 = plot(times, xs_k[p1x_idx,:], label="P1 px", legend=:outertopright)
plot!(times, xs_k[p1y_idx,:], label="P1 py")
plot!(times, xs_k[p2x_idx,:], label="P2 px", legend=:outertopright)
plot!(times, xs_k[p2y_idx,:], label="P2 py")

q3 = plot(times, xs_k[3,:], label="P1 θ", legend=:outertopright)
plot!(times, xs_k[4,:], label="P1 v")
plot!(times, xs_k[7,:], label="P2 θ")
plot!(times, xs_k[8,:], label="P2 v")

q4 = plot(times, us_k[1][1, :], label="P1 ω", legend=:outertopright)
plot!(times, us_k[1][2, :], label="P1 accel")
plot!(times, us_k[2][1, :], label="P2 ω", legend=:outertopright)
plot!(times, us_k[2][2, :], label="P2 accel")

# Plot convergence.
conv_x = cumsum(ones(num_iters)) .- 1
# q5 = plot(title="convergence")
# plot!(conv_x, conv_metrics[1, 1:num_iters], label="p1", yaxis=:log)
# plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2")
q5 = plot(conv_x, conv_metrics[1, 1:num_iters], title="conv. metric", label="p1", yaxis=:log, legend=:outertopright)
plot!(q5, conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)
plot!(q5, conv_x, threshold * ones(length(conv_x)), label="threshold")

conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
plot!(conv_x, conv_sum, label="total")

# q6 = plot(title="costs")
# plot!(conv_x, evaluated_costs[1, 1:num_iters], label="p1", yaxis=:log)
# plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2")

# Shift the cost to ensure they are positive.
costs_1 = evaluated_costs[1, 1:num_iters] .+ (abs(minimum(evaluated_costs[1, 1:num_iters])) + 1e-2)
costs_2 = evaluated_costs[2, 1:num_iters] .+ (abs(minimum(evaluated_costs[2, 1:num_iters])) + 1e-2)

q6 = plot(conv_x, costs_1, title="evaluated costs", label="p1", yaxis=:log, legend=:outertopright)
plot!(q6, conv_x, costs_2, label="p2", yaxis=:log)

cost_sum = costs_1 + costs_2
plot!(conv_x, cost_sum, label="total", yaxis=:log)

# plot(q1, q2, q3, q4, layout = q)
plot(q1, q2, q3, q4, q5, q6, layout = q)



