using StackelbergControlHypothesesFiltering

using LinearAlgebra
using Plots

# TODO(hamzah) - Run iLQR and get outputs.
num_players = 1
T = 101
x0 = zeros(4)
# x0[1] = -2.0
# x0[2] = -1.0
# x0[3] = 0.0
x0[4] = pi/4
t0 = 0.0

dt = 0.05
times = dt * cumsum(ones(T)) .- dt

xf = [5.;5.;0.;0.]       # for the double integrator dynamics
xf = [5.; 5.; -pi/2; 0.] # for the unicycle dynamics

println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(T*dt, sigdigits=4), " seconds.")


#####################################
#        Define the dynamics.       #
#####################################
# Ensure the dynamics and costs are both homogenized similarly.
dyn = LinearDynamics([1. 0. dt 0.;
                      0. 1. 0. dt;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.],
                     [vcat(zeros(2,2),
                      [dt 0; 0 dt])]) # 2d double integrator [x y xdot ydot]
dyn = UnicycleDynamics(num_players)


#####################################
#         Define the costs.         #
#####################################
Q = Matrix(Diagonal(10*[1., 1., 1., 1.]))
R = Matrix(Diagonal(1*[1., 1.]))
quad_cost = make_quadratic_cost_with_offset(Q, xf)
add_control_cost!(quad_cost, 1, R)
dummy_time_range = (0.0, dt)
dummy_x = zeros(xdim(dyn))
dummy_us = [zeros(udim(dyn, ii)) for ii in 1:num_agents(dyn)]
pure_quad_cost = quadraticize_costs(quad_cost, dummy_time_range, dummy_x, dummy_us)

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
# K̃s, _ = solve_lqr_feedback(lin_dyn_0, pure_quad_cost, T)
# _, us_1 = unroll_feedback(dyn, times, FeedbackGainControlStrategy([K̃s]), x0)
# us_1 = us_1[1] #+ randn(size(us_1[1]))

selected_cost = quad_cost

xs_i, us_i, is_converged, num_iters, costs = ilqr(T, t0, times, dyn, selected_cost, x0, us_1; max_iters=1000, step_size=0.1, threshold=1.)
final_cost_total = evaluate(selected_cost, xs_i, [us_i])

println("final: ", xs_i[:, T], " with trajectory cost: ", final_cost_total)
println(size(xs_i), " ", size(us_i), " ", num_iters, " ", is_converged)




# TODO(hamzah) - Plot time vs. cost

# # Plot x-y state.
# println(1, " ", xs_i[1,:])
# println(2, " ", xs_i[2,:])
# println(3, " ", xs_i[3,:])
# println(4, " ", xs_i[4,:])

# println("u1", " ", us_i[1,:])
# println("u2", " ", us_i[2,:])

q = @layout [a b; c d; e]

q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright)
q1 = scatter!([x0[1]], [x0[2]], color="red", label="start")
q1 = scatter!([xf[1]], [xf[2]], color="blue", label="goal")

q2 = plot(times, xs_i[1,:], label="px", legend=:outertopright)
plot!(times, xs_i[2,:], label="py")

q3 = plot(times, xs_i[3,:], label="theta/vel x", legend=:outertopright)
plot!(times, xs_i[4,:], label="velocity/vel y")

q4 = plot(times, us_i[1, :], label="rot-vel/accel x", legend=:outertopright)
plot!(times, us_i[2, :], label="accel/accel y")

# q4 = plot(times, 100 * ones(size(times)), label="cost")

conv_x = cumsum(ones(num_iters+1)) .- 1
q5 = plot(conv_x, costs[1:num_iters+1])


plot(q1, q2, q3, q4, q5, layout = q)



