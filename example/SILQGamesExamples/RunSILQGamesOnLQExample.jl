using StackelbergControlHypothesesFiltering

using LinearAlgebra

include("LQ_parameters.jl")

costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

leader_idx=1
num_runs=1

# config variables
threshold=1.
max_iters=1000
step_size=1e-2
verbose=true

sg_obj = initialize_silq_games_object(num_runs, leader_idx, T, dyn, costs;
                                      threshold=threshold, max_iters=max_iters, step_size=step_size, verbose=verbose)
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(sg_obj, times[1], times, x₁, us_1)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals, " sum: ", sum(final_cost_totals))
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
q5 = plot(conv_x, conv_metrics[1, 1:num_iters], title="convergence (||k||^2) by player", label="p1", yaxis=:log, legend=:outertopright)
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)

conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
plot!(conv_x, conv_sum, label="total", yaxis=:log)

q6 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log, legend=:outertopright)
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)

cost_sum = evaluated_costs[1, 1:num_iters] + evaluated_costs[2, 1:num_iters]
plot!(conv_x, cost_sum, label="total", yaxis=:log)

plot(q1, q2, q3, q4, q5, q6, layout = q)



