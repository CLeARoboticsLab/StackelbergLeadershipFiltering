using StackelbergControlHypothesesFiltering

include("LQ_parameters.jl")

costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

const_multiplier = 1.0
max_vel = 5.0
max_accel = 1.0
example_ilqr_cost1 = ExampleILQRCost(costs[1], const_multiplier, max_vel, max_accel, costs[1].x_dest, true, 1)
example_ilqr_cost2 = ExampleILQRCost(costs[2], const_multiplier, max_vel, max_accel, costs[2].x_dest, true, 2)
costs = [example_ilqr_cost1, example_ilqr_cost2]

leader_idx=1
xs_k, us_k, is_converged, num_iters, conv_metrics, evaluated_costs = stackelberg_ilqgames(
                              leader_idx,
                              T,
                              times[1],
                              times,
                              dyn,
                              costs,
                              x₁,
                              us_1;
                              threshold=1.,
                              max_iters=680,
                              step_size=0.01,
                              verbose=true)

println("Converged status (", is_converged, ") after ", num_iters, " iterations.")
final_cost_totals = [evaluate(costs[ii], xs_k, us_k) for ii in 1:num_players]
println("final: ", xs_k[:, T], " with trajectory costs: ", final_cost_totals)
println(size(xs_k), " ", size(us_k[1]), " ", size(us_k[2]))



using ElectronDisplay
using Plots


# Plot positions, other two states, controls, and convergence.
q = @layout [o; a b; c d; e f]

title = plot(title = "SILQGames on Shepherd and Sheep (nonlinear cost)", grid=false, showaxis=false, bottom_margin = -100Plots.px, framestyle=nothing, xticks=false, yticks=false)

q1 = plot(legend=:outertopright)
plot!(q1, xs_k[1, :], xs_k[3, :], label="leader pos", xlabel="x (m)", ylabel="y (m)")
plot!(q1, xs_k[5, :], xs_k[7, :], label="follower pos")

# q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Iterative LQR")
q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

q2 = plot(times, xs_k[1,:], label="P1 px", legend=:outertopright, xlabel="t (s)", ylabel="dist. (m)")
plot!(times, xs_k[3,:], label="P1 py")
plot!(times, xs_k[5,:], label="P2 px", legend=:outertopright)
plot!(times, xs_k[7,:], label="P2 py")

q3 = plot(times, xs_k[2,:], label="P1 vx", legend=:outertopright, xlabel="t (s)", ylabel="vel. (mps)")
plot!(times, xs_k[4,:], label="P1 vy")
plot!(times, xs_k[6,:], label="P2 vx")
plot!(times, xs_k[8,:], label="P2 vy")

q4 = plot(times, us_k[1][1, :], label="P1 accel x", legend=:outertopright, xlabel="t (s)", ylabel="accel. (mpss)")
plot!(times, us_k[1][2, :], label="P1 accel y")
plot!(times, us_k[2][1, :], label="P2 accel x", legend=:outertopright)
plot!(times, us_k[2][2, :], label="P2 accel y")

# Plot convergence.
conv_x = cumsum(ones(num_iters)) .- 1
q5 = plot(conv_x, conv_metrics[1, 1:num_iters], title="convergence (||k||^2) by player", label="p1")
plot!(conv_x, conv_metrics[2, 1:num_iters], label="p2")

q6 = plot(conv_x, evaluated_costs[1, 1:num_iters], title="evaluated costs", label="p1")
plot!(conv_x, evaluated_costs[2, 1:num_iters], label="p2")

plot(title, q1, q2, q3, q4, q5, q6, layout=q)
