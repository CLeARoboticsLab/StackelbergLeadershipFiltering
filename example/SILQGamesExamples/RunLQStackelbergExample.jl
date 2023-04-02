using Plots

include("LQ_parameters.jl")

Ps_strategies, Zs_future_costs = solve_lq_stackelberg_feedback(dyn, costs, T, leader_idx)

xs, us = unroll_feedback(dyn, times, Ps_strategies, x₁)

final_cost_totals = [evaluate(costs[ii], xs, us) for ii in 1:num_players]
println("final: ", xs[:, T], " with trajectory costs: ", final_cost_totals)
println(size(xs), " ", size(us[1]), " ", size(us[2]))


using ElectronDisplay

# Plot positions, other two states, controls, and convergence.
q = @layout [a b; c d]

q1 = plot(legend=:outertopright)
plot!(q1, xs[1, :], xs[3, :], label="leader pos")
plot!(q1, xs[5, :], xs[7, :], label="follower pos")

# q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Iterative LQR")
q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

q2 = plot(times, xs[1,:], label="P1 px", legend=:outertopright)
plot!(times, xs[3,:], label="P1 py")
plot!(times, xs[5,:], label="P2 px", legend=:outertopright)
plot!(times, xs[7,:], label="P2 py")

q3 = plot(times, xs[2,:], label="vel x", legend=:outertopright)
plot!(times, xs[4,:], label="vel y")

q4 = plot(times, us[1][1, :], label="P1 accel x", legend=:outertopright)
plot!(times, us[1][2, :], label="P1 accel y")
plot!(times, us[2][1, :], label="P2 accel x", legend=:outertopright)
plot!(times, us[2][2, :], label="P2 accel y")

plot(q1, q2, q3, q4, layout = q)