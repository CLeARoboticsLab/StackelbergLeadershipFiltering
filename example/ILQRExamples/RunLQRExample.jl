using Plots

include("params_time.jl")
include("params_doubleintegrator_quadoffset.jl")

time_range = (0.0, horizon)
dummy_us = [zeros(udim(dyn, ii)) for ii in 1:num_agents(dyn)]
lqr_quad_cost_x0 = quadraticize_costs(quad_w_offset_cost, time_range, x0, dummy_us)

# Solve optimal control problem.
println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(horizon, sigdigits=4), " seconds.")

ctrl_strats, _ = solve_lqr_feedback(dyn, lqr_quad_cost_x0, T)

xs_i, us_i = unroll_feedback(dyn, times, ctrl_strats, x0)

final_cost_total = evaluate(quad_w_offset_cost, xs_i, us_i)
println("final: ", xs_i[:, T], " with trajectory cost: ", final_cost_total)


# Plot positions, velocities, and controls.
q = @layout [a b; c d]

q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright, title="LQR")
q1 = scatter!([x0[1]], [x0[2]], color="red", label="start")
q1 = scatter!([xf[1]], [xf[2]], color="blue", label="goal")

q2 = plot(times, xs_i[1,:], label="px", legend=:outertopright)
plot!(times, xs_i[2,:], label="py")

q3 = plot(times, xs_i[3,:], label="vel x", legend=:outertopright)
plot!(times, xs_i[4,:], label="vel y")

q4 = plot(times, us_i[1][1, :], label="accel x", legend=:outertopright)
plot!(times, us_i[1][2, :], label="accel y")

plot(q1, q2, q3, q4, layout = q)
