using Plots
include("LQ_parameters.jl")

# Try out unicycle dynamics.
dyn = UnicycleDynamics(num_players)
xf = [5.; 5.; -pi/2; 0.]

# TODO(hmzh) - Implement this particular non-quadratic cost and test.
# const_multiplier = 1.0
# max_accel = 5.
# max_omega = 1.
# cost = ExampleILQRCost(pure_quad_cost, const_multiplier, max_accel, max_omega, xf, true)

us_1 = zeros(udim(dyn), T)
us_1[1,:] .= 0.1
us_1[2,:] .= 0.01
duration = (T-1) * dt
us_1[1, :] .= (xf[3] - x0[3]) / duration # omega
us_1[2, :] .= (xf[4] - x0[4]) / duration # accel

# lin_dyn_0 = linearize_dynamics(dyn, (t0, t0+dt), x0, [zeros(udim(dyn))])
# ctrl_strats, _ = solve_lqr_feedback(lin_dyn_0, quad_cost, T)
# _, us_1 = unroll_feedback(dyn, times, ctrl_strats, x0)
# us_1 = us_1[1] #+ randn(size(us_1[1]))

selected_cost = quad_w_offset_cost

xs_i, us_i, is_converged, num_iters, costs = ilqr(T, t0, times, dyn, selected_cost, x0, us_1; max_iters=1000, step_size=1., threshold=0.1, verbose=true)
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
