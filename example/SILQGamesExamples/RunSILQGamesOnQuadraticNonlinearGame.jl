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


function produce_state_and_control_plots(dyn::UnicycleDynamics, times, xs, us)
    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[2, :], label="P1 pos")
    plot!(q1, xs[5, :], xs[6, :], label="P2 pos")

    q1 = scatter!([x₁[1]], [x₁[2]], color="blue", label="start P1")
    q1 = scatter!([x₁[5]], [x₁[6]], color="red", label="start P2")

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
    plot!(times, xs[3,:], label="P1 θ")
    plot!(times, xs[7,:], label="P2 θ")

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

# Plot positions, other two states, controls, and convergence.
q = @layout [a b c; d e f; g h i]

q1, q2, q3, q4, q5, q6, q7 = produce_state_and_control_plots(dyn, times, xs_k, us_k)

# Plot convergence.
conv_x = cumsum(ones(num_iters)) .- 1
q8 = plot(conv_x, conv_metrics[1, 1:num_iters], title="convergence (||k||^2) by player", label="p1", yaxis=:log, legend=:outertopright)
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2", yaxis=:log)

conv_sum = conv_metrics[1, 1:num_iters] + conv_metrics[2, 1:num_iters]
plot!(conv_x, conv_sum, label="total", yaxis=:log)

q9 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1", yaxis=:log, legend=:outertopright)
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2", yaxis=:log)

cost_sum = evaluated_costs[1, 1:num_iters] + evaluated_costs[2, 1:num_iters]
plot!(conv_x, cost_sum, label="total", yaxis=:log)

plot(q1, q2, q3, q4, q5, q6, q7, q8, q9, layout = q)



