using StackelbergControlHypothesesFiltering

using LinearAlgebra

# includes linear dynamics and quadratic costs
include("nonquadratic_linear_parameters.jl")

costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

# P1 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
indices1 = [1, 3] # x, y
x_offset_1 = zeros(xdim(dyn))
x_offset_1[indices1] .= -(1.0/2)
x_zero_int_1 = zeros(xdim(dyn))
x_zero_int_1[indices1] .= 10.
log_cost_p1 = LogBarrierCost(indices1, x_offset_1, x_zero_int_1)

# P2 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
indices2 = [5, 7] # x, y
x_offset_2 = zeros(xdim(dyn))
x_offset_2[indices2] .= -(1.0/2)
x_zero_int_2 = zeros(xdim(dyn))
x_zero_int_2[indices2] .= 10.
log_cost_p2 = LogBarrierCost(indices2, x_offset_2, x_zero_int_2)

# Make the weighted cost.
p1_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[1]), log_cost_p1])
# new_costs = [p1_new_cost, costs[2]]

p2_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[2]), log_cost_p2])
# new_costs = [costs[1], p2_new_cost]

new_costs = [p1_new_cost, p2_new_cost]

leader_idx=1
num_runs=1

# config variables
threshold=0.0001
max_iters=10000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, T, dyn, new_costs;
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, leader_idx, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(new_costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals)
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))


using ElectronDisplay
using Plots

# Plot positions, other two states, controls, and convergence.
q = @layout [a b; c d; e f]

q1 = plot(legend=:outertopright)
plot!(q1, xs_k[1, :], xs_k[3, :], label="leader pos")
plot!(q1, xs_k[5, :], xs_k[7, :], label="follower pos")

# q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Iterative LQR")
q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

q2 = plot(times, xs_k[1,:], label="P1 px", legend=:outertopright)
plot!(times, xs_k[3,:], label="P1 py")
plot!(times, xs_k[5,:], label="P2 px", legend=:outertopright)
plot!(times, xs_k[7,:], label="P2 py")

q3 = plot(times, xs_k[2,:], label="vel1 x", legend=:outertopright)
plot!(times, xs_k[4,:], label="vel1 y")
plot!(times, xs_k[6,:], label="vel2 x")
plot!(times, xs_k[8,:], label="vel2 y")

q4 = plot(times, us_k[1][1, :], label="P1 accel x", legend=:outertopright)
plot!(times, us_k[1][2, :], label="P1 accel y")
plot!(times, us_k[2][1, :], label="P2 accel x", legend=:outertopright)
plot!(times, us_k[2][2, :], label="P2 accel y")

# Plot convergence.
conv_x = cumsum(ones(num_iters)) .- 1
q5 = plot(conv_x, conv_metrics[1, 1:num_iters], title="||k||^2 metric", label="p1", yaxis=:log)
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)

q6 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log)
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)

plot(q1, q2, q3, q4, q5, q6, layout = q)



