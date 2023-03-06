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
# x0[4] = 0.
t0 = 0.0

dt = 0.05
times = dt * cumsum(ones(T)) .- dt

xf = [5.;5.;0.;0.]       # for the double integrator dynamics

println("initial state: ", x0')
println("desired state at time T: ", round.(xf', sigdigits=6), " over ", round(T*dt, sigdigits=4), " seconds.")

# Ensure the dynamics and costs are both homogenized similarly.
dyn = LinearDynamics([1. 0. dt 0.;
                      0. 1. 0. dt;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.],
                     [vcat(zeros(2,2), [dt 0; 0 dt])]) # 2d double integrator [x y xdot ydot]

# Costs
Q = Matrix(Diagonal([1., 1., 1., 1.]))
R = Matrix(Diagonal([1., 1.]))
# quad_cost = make_quadratic_cost_with_offset(Q, xf)
quad_cost = QuadraticCost(Q)
add_control_cost!(quad_cost, 1, R)

# Solve optimal control problem.
ctrl_strats, _ = solve_lqr_feedback(dyn, quad_cost, T)
xs_i, us_i = unroll_feedback(dyn, times, ctrl_strats, x0-xf)

final_cost_total = evaluate(quad_cost, xs_i, us_i)
println("final: ", xs_i[:, T], " with trajectory cost: ", final_cost_total)

# TODO(hamzah) - Plot time vs. cost

# # Plot x-y state.
# println(1, " ", xs_i[1,:])
# println(2, " ", xs_i[2,:])
# println(3, " ", xs_i[3,:])
# println(4, " ", xs_i[4,:])

# println("u1", " ", us_i[1][1,:])
# println("u2", " ", us_i[1][2,:])

q = @layout [a b; c d]

xs_i = xs_i .+ xf

q1 = plot(xs_i[1,:], xs_i[2,:], label="", legend=:outertopright)
q1 = scatter!([x0[1]], [x0[2]], color="red", label="start")
q1 = scatter!([xf[1]], [xf[2]], color="blue", label="goal")

q2 = plot(times, xs_i[1,:], label="px", legend=:outertopright)
plot!(times, xs_i[2,:], label="py")

q3 = plot(times, xs_i[3,:], label="vel x", legend=:outertopright)
plot!(times, xs_i[4,:], label="vel y")

q4 = plot(times, us_i[1][1, :], label="accel x", legend=:outertopright)
plot!(times, us_i[1][2, :], label="accel y")

# q4 = plot(times, 100 * ones(size(times)), label="cost")

plot(q1, q2, q3, q4, layout = q)



