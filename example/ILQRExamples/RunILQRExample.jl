using Plots
include("LQ_parameters.jl")

selected_cost = quad_cost

xs_i, us_i, is_converged, num_iters, costs = ilqr(T, t0, times, dyn, selected_cost, x0, us_1; max_iters=1000, step_size=1., threshold=1., verbose=true)
final_cost_total = evaluate(selected_cost, xs_i, [us_i])

println("final: ", xs_i[:, T], " with trajectory cost: ", final_cost_total)
println(size(xs_i), " ", size(us_i), " ", num_iters, " ", is_converged)


# Plot positions, other two states, controls, and convergence.
q = @layout [a b; c d; e]

q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="Iterative LQR")
q1 = scatter!([x0[1]], [x0[2]], color="red", label="start")
q1 = scatter!([xf[1]], [xf[2]], color="blue", label="goal")

q2 = plot(times, xs_i[1,:], label="px", legend=:outertopright)
plot!(times, xs_i[2,:], label="py")

q3 = plot(times, xs_i[3,:], label="theta/vel x", legend=:outertopright)
plot!(times, xs_i[4,:], label="velocity/vel y")

q4 = plot(times, us_i[1, :], label="rot-vel/accel x", legend=:outertopright)
plot!(times, us_i[2, :], label="accel/accel y")

conv_x = cumsum(ones(num_iters+1)) .- 1
q5 = plot(conv_x, costs[1:num_iters+1], title="convergence (||k||)")


plot(q1, q2, q3, q4, q5, layout = q)
